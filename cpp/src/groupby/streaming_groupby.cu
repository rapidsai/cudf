/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/groupby.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <algorithm>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

namespace cudf {
namespace groupby {

namespace {

/**
 * @brief Describes one partial-state column: how to produce it from a batch and how to merge two
 * instances.
 *
 * For example, MEAN stores two partial columns: one with batch_kind=SUM / merge_kind=SUM
 * (running sum) and one with batch_kind=COUNT_VALID / merge_kind=SUM (running count).
 */
struct partial_agg_descriptor {
  aggregation::Kind batch_kind;  ///< Aggregation to run on each incoming batch
  aggregation::Kind merge_kind;  ///< Aggregation to combine two partial-state columns
};

/**
 * @brief How to convert accumulated partial state into the final result.
 */
enum class finalize_kind : int32_t {
  IDENTITY,             ///< Partial state is the final result
  MEAN_FROM_SUM_COUNT,  ///< Divide running sum by running count
  VARIANCE_FROM_M2,     ///< Compute M2 / (count - ddof) from an M2 struct
  STD_FROM_M2,          ///< Compute sqrt(M2 / (count - ddof)) from an M2 struct
};

/**
 * @brief Execution plan for a single requested aggregation.
 *
 * Maps one requested aggregation (e.g. MEAN on column 3) to the batch aggregations,
 * merge aggregations, partial-state layout, and finalization logic needed to
 * implement it incrementally.
 */
struct request_plan {
  size_type value_column_index;                ///< Column index in the input data table
  aggregation::Kind requested_kind;            ///< The originally requested aggregation kind
  std::unique_ptr<aggregation> requested_agg;  ///< Clone of the original agg (for params like ddof)
  std::vector<partial_agg_descriptor> partial_descs;  ///< One entry per partial-state column
  finalize_kind finalization;                         ///< How to produce the final result
  size_type partial_state_offset;                     ///< First partial column index (after keys)
  size_type num_partial_columns;                      ///< Number of partial-state columns
};

/**
 * @brief Create a groupby_aggregation to run on an incoming data batch.
 *
 * Maps the requested aggregation kind (or its decomposed component, e.g. MEAN→SUM)
 * to a concrete groupby_aggregation factory call. COLLECT_SET is mapped to COLLECT_LIST
 * because duplicate removal happens during the merge step via MERGE_SETS.
 */
std::unique_ptr<groupby_aggregation> make_batch_agg(aggregation::Kind kind)
{
  switch (kind) {
    case aggregation::SUM: return make_sum_aggregation<groupby_aggregation>();
    case aggregation::PRODUCT: return make_product_aggregation<groupby_aggregation>();
    case aggregation::MIN: return make_min_aggregation<groupby_aggregation>();
    case aggregation::MAX: return make_max_aggregation<groupby_aggregation>();
    case aggregation::COUNT_VALID:
      return make_count_aggregation<groupby_aggregation>(null_policy::EXCLUDE);
    case aggregation::COUNT_ALL:
      return make_count_aggregation<groupby_aggregation>(null_policy::INCLUDE);
    case aggregation::SUM_OF_SQUARES: return make_sum_of_squares_aggregation<groupby_aggregation>();
    case aggregation::M2: return make_m2_aggregation<groupby_aggregation>();
    case aggregation::MEAN: return make_sum_aggregation<groupby_aggregation>();
    case aggregation::VARIANCE: return make_m2_aggregation<groupby_aggregation>();
    case aggregation::STD: return make_m2_aggregation<groupby_aggregation>();
    case aggregation::TDIGEST:
    case aggregation::MERGE_TDIGEST: return make_tdigest_aggregation<groupby_aggregation>();
    case aggregation::HISTOGRAM: return make_histogram_aggregation<groupby_aggregation>();
    case aggregation::MERGE_HISTOGRAM:
      return make_merge_histogram_aggregation<groupby_aggregation>();
    case aggregation::COLLECT_LIST: return make_collect_list_aggregation<groupby_aggregation>();
    case aggregation::COLLECT_SET: return make_collect_list_aggregation<groupby_aggregation>();
    case aggregation::MERGE_LISTS: return make_merge_lists_aggregation<groupby_aggregation>();
    case aggregation::MERGE_SETS: return make_merge_lists_aggregation<groupby_aggregation>();
    default: CUDF_FAIL("Unsupported batch aggregation kind for streaming groupby");
  }
}

/**
 * @brief Create a groupby_aggregation to merge two partial-state columns.
 *
 * For simple associative aggregations (SUM, MIN, MAX, PRODUCT) the merge agg is the
 * same as the batch agg. For compound types (M2, TDIGEST, HISTOGRAM, LISTS, SETS)
 * the corresponding MERGE_* aggregation is used. @p original_agg is consulted for
 * parameters like max_centroids on TDIGEST.
 */
std::unique_ptr<groupby_aggregation> make_merge_agg(aggregation::Kind kind,
                                                    aggregation const* original_agg = nullptr)
{
  switch (kind) {
    case aggregation::SUM: return make_sum_aggregation<groupby_aggregation>();
    case aggregation::PRODUCT: return make_product_aggregation<groupby_aggregation>();
    case aggregation::MIN: return make_min_aggregation<groupby_aggregation>();
    case aggregation::MAX: return make_max_aggregation<groupby_aggregation>();
    case aggregation::MERGE_M2: return make_merge_m2_aggregation<groupby_aggregation>();
    case aggregation::MERGE_TDIGEST: {
      auto max_centroids = 1000;
      if (original_agg) {
        if (auto const* td = dynamic_cast<cudf::detail::tdigest_aggregation const*>(original_agg)) {
          max_centroids = td->max_centroids;
        }
        if (auto const* mtd =
              dynamic_cast<cudf::detail::merge_tdigest_aggregation const*>(original_agg)) {
          max_centroids = mtd->max_centroids;
        }
      }
      return make_merge_tdigest_aggregation<groupby_aggregation>(max_centroids);
    }
    case aggregation::MERGE_HISTOGRAM:
      return make_merge_histogram_aggregation<groupby_aggregation>();
    case aggregation::MERGE_LISTS: return make_merge_lists_aggregation<groupby_aggregation>();
    case aggregation::MERGE_SETS: return make_merge_sets_aggregation<groupby_aggregation>();
    default: CUDF_FAIL("Unsupported merge aggregation kind for streaming groupby");
  }
}

/**
 * @brief Construct a request_plan from its components.
 */
request_plan make_plan(size_type col_idx,
                       aggregation const& agg,
                       size_type offset,
                       std::vector<partial_agg_descriptor> descs,
                       finalize_kind fin,
                       size_type num_cols)
{
  request_plan plan;
  plan.value_column_index   = col_idx;
  plan.requested_kind       = agg.kind;
  plan.requested_agg        = agg.clone();
  plan.partial_state_offset = offset;
  plan.partial_descs        = std::move(descs);
  plan.finalization         = fin;
  plan.num_partial_columns  = num_cols;
  return plan;
}

/**
 * @brief Convert streaming_aggregation_requests into an ordered list of request_plans.
 *
 * Each aggregation in each request becomes one plan. The plan specifies which batch
 * and merge aggregations to use, how many partial-state columns are needed, and what
 * finalization to apply. Partial-state column offsets are assigned contiguously.
 *
 * @throws cudf::invalid_argument for unsupported aggregation kinds
 */
std::vector<request_plan> build_request_plans(
  host_span<streaming_aggregation_request const> requests)
{
  std::vector<request_plan> plans;
  auto offset = size_type{0};

  for (auto const& req : requests) {
    for (auto const& agg : req.aggregations) {
      auto const col = req.column_index;

      switch (agg->kind) {
        case aggregation::SUM:
          plans.push_back(make_plan(
            col, *agg, offset, {{aggregation::SUM, aggregation::SUM}}, finalize_kind::IDENTITY, 1));
          break;
        case aggregation::PRODUCT:
          plans.push_back(make_plan(col,
                                    *agg,
                                    offset,
                                    {{aggregation::PRODUCT, aggregation::PRODUCT}},
                                    finalize_kind::IDENTITY,
                                    1));
          break;
        case aggregation::MIN:
          plans.push_back(make_plan(
            col, *agg, offset, {{aggregation::MIN, aggregation::MIN}}, finalize_kind::IDENTITY, 1));
          break;
        case aggregation::MAX:
          plans.push_back(make_plan(
            col, *agg, offset, {{aggregation::MAX, aggregation::MAX}}, finalize_kind::IDENTITY, 1));
          break;
        case aggregation::COUNT_VALID:
          plans.push_back(make_plan(col,
                                    *agg,
                                    offset,
                                    {{aggregation::COUNT_VALID, aggregation::SUM}},
                                    finalize_kind::IDENTITY,
                                    1));
          break;
        case aggregation::COUNT_ALL:
          plans.push_back(make_plan(col,
                                    *agg,
                                    offset,
                                    {{aggregation::COUNT_ALL, aggregation::SUM}},
                                    finalize_kind::IDENTITY,
                                    1));
          break;
        case aggregation::MEAN:
          plans.push_back(make_plan(
            col,
            *agg,
            offset,
            {{aggregation::SUM, aggregation::SUM}, {aggregation::COUNT_VALID, aggregation::SUM}},
            finalize_kind::MEAN_FROM_SUM_COUNT,
            2));
          break;
        case aggregation::SUM_OF_SQUARES:
          plans.push_back(make_plan(col,
                                    *agg,
                                    offset,
                                    {{aggregation::SUM_OF_SQUARES, aggregation::SUM}},
                                    finalize_kind::IDENTITY,
                                    1));
          break;
        case aggregation::M2:
          plans.push_back(make_plan(col,
                                    *agg,
                                    offset,
                                    {{aggregation::M2, aggregation::MERGE_M2}},
                                    finalize_kind::IDENTITY,
                                    1));
          break;
        case aggregation::VARIANCE:
          plans.push_back(make_plan(col,
                                    *agg,
                                    offset,
                                    {{aggregation::M2, aggregation::MERGE_M2}},
                                    finalize_kind::VARIANCE_FROM_M2,
                                    1));
          break;
        case aggregation::STD:
          plans.push_back(make_plan(col,
                                    *agg,
                                    offset,
                                    {{aggregation::M2, aggregation::MERGE_M2}},
                                    finalize_kind::STD_FROM_M2,
                                    1));
          break;
        case aggregation::TDIGEST:
          plans.push_back(make_plan(col,
                                    *agg,
                                    offset,
                                    {{aggregation::TDIGEST, aggregation::MERGE_TDIGEST}},
                                    finalize_kind::IDENTITY,
                                    1));
          break;
        case aggregation::MERGE_TDIGEST:
          plans.push_back(make_plan(col,
                                    *agg,
                                    offset,
                                    {{aggregation::MERGE_TDIGEST, aggregation::MERGE_TDIGEST}},
                                    finalize_kind::IDENTITY,
                                    1));
          break;
        case aggregation::HISTOGRAM:
          plans.push_back(make_plan(col,
                                    *agg,
                                    offset,
                                    {{aggregation::HISTOGRAM, aggregation::MERGE_HISTOGRAM}},
                                    finalize_kind::IDENTITY,
                                    1));
          break;
        case aggregation::MERGE_HISTOGRAM:
          plans.push_back(make_plan(col,
                                    *agg,
                                    offset,
                                    {{aggregation::MERGE_HISTOGRAM, aggregation::MERGE_HISTOGRAM}},
                                    finalize_kind::IDENTITY,
                                    1));
          break;
        case aggregation::COLLECT_LIST:
          plans.push_back(make_plan(col,
                                    *agg,
                                    offset,
                                    {{aggregation::COLLECT_LIST, aggregation::MERGE_LISTS}},
                                    finalize_kind::IDENTITY,
                                    1));
          break;
        case aggregation::COLLECT_SET:
          plans.push_back(make_plan(col,
                                    *agg,
                                    offset,
                                    {{aggregation::COLLECT_SET, aggregation::MERGE_SETS}},
                                    finalize_kind::IDENTITY,
                                    1));
          break;
        case aggregation::MERGE_LISTS:
          plans.push_back(make_plan(col,
                                    *agg,
                                    offset,
                                    {{aggregation::MERGE_LISTS, aggregation::MERGE_LISTS}},
                                    finalize_kind::IDENTITY,
                                    1));
          break;
        case aggregation::MERGE_SETS:
          plans.push_back(make_plan(col,
                                    *agg,
                                    offset,
                                    {{aggregation::MERGE_SETS, aggregation::MERGE_SETS}},
                                    finalize_kind::IDENTITY,
                                    1));
          break;
        default:
          CUDF_FAIL(
            "Unsupported aggregation kind for streaming groupby: " + std::to_string(agg->kind),
            std::invalid_argument);
      }
      offset += plans.back().num_partial_columns;
    }
  }
  return plans;
}

/**
 * @brief Determine the data type of a partial-state column after merging.
 *
 * When the batch and merge aggregation kinds differ (e.g. COUNT_VALID batched, SUM
 * merged), the merge output type may be wider than the batch output type (INT32→INT64).
 * This function returns the stable type that the partial-state column should be stored as.
 */
data_type compute_partial_state_type(data_type batch_result_type,
                                     aggregation::Kind batch_kind,
                                     aggregation::Kind merge_kind)
{
  if (batch_kind == merge_kind) { return batch_result_type; }
  return cudf::detail::target_type(batch_result_type, merge_kind);
}

/**
 * @brief Combine key columns and partial-state columns into a single state table.
 *
 * The resulting table layout is [key_col_0, ..., key_col_N, partial_col_0, ...].
 */
std::unique_ptr<table> build_state_table(std::unique_ptr<table>&& keys,
                                         std::vector<std::unique_ptr<column>>& partial_cols)
{
  auto cols = keys->release();
  for (auto& c : partial_cols) {
    cols.push_back(std::move(c));
  }
  return std::make_unique<table>(std::move(cols));
}

/**
 * @brief Merge new partial results into the accumulated state.
 *
 * Concatenates @p accumulated and @p new_partials vertically, then runs a groupby
 * with the appropriate merge aggregations to collapse duplicate keys back to one row
 * per group. The merged result replaces @p accumulated in place.
 */
void merge_partial_states(std::unique_ptr<table>& accumulated,
                          table_view new_partials,
                          size_type num_keys,
                          std::vector<request_plan> const& plans,
                          null_policy null_handling,
                          rmm::cuda_stream_view stream,
                          rmm::device_async_resource_ref mr)
{
  std::vector<table_view> to_concat{accumulated->view(), new_partials};
  auto concatenated = cudf::concatenate(to_concat, stream, mr);

  std::vector<size_type> key_col_indices(num_keys);
  std::iota(key_col_indices.begin(), key_col_indices.end(), 0);
  auto concat_keys = concatenated->view().select(key_col_indices);

  cudf::groupby::groupby merge_groupby{concat_keys, null_handling};

  std::vector<aggregation_request> merge_requests;
  for (auto const& plan : plans) {
    for (size_type i = 0; i < plan.num_partial_columns; ++i) {
      aggregation_request req;
      req.values = concatenated->view().column(num_keys + plan.partial_state_offset + i);
      req.aggregations.push_back(
        make_merge_agg(plan.partial_descs[i].merge_kind, plan.requested_agg.get()));
      merge_requests.push_back(std::move(req));
    }
  }

  auto [merged_keys, merged_results] = merge_groupby.aggregate(merge_requests, stream, mr);

  std::vector<std::unique_ptr<column>> merged_partials;
  for (auto& res : merged_results) {
    for (auto& col : res.results) {
      merged_partials.push_back(std::move(col));
    }
  }

  accumulated = build_state_table(std::move(merged_keys), merged_partials);
}

}  // namespace

struct streaming_groupby::impl {
  std::vector<size_type> _key_indices;
  std::vector<size_type> _aggs_per_request;
  std::vector<request_plan> _plans;
  null_policy _null_handling;
  sorted _keys_are_sorted;
  std::vector<order> _column_order;
  std::vector<null_order> _null_precedence;

  std::unique_ptr<table> _partial_state;
  std::vector<data_type> _partial_column_types;

  /**
   * @brief Whether partial-state column types have been determined from the first batch.
   */
  [[nodiscard]] bool types_initialized() const { return !_partial_column_types.empty(); }

  /**
   * @brief Number of key columns, derived from the key_indices passed at construction.
   */
  [[nodiscard]] size_type num_keys() const { return static_cast<size_type>(_key_indices.size()); }

  /**
   * @brief Total number of partial-state columns across all plans.
   *
   * The internal state table has `num_keys() + total_partial_columns()` columns.
   */
  [[nodiscard]] size_type total_partial_columns() const
  {
    return std::accumulate(
      _plans.begin(), _plans.end(), size_type{0}, [](size_type acc, auto const& p) {
        return acc + p.num_partial_columns;
      });
  }

  impl(host_span<size_type const> key_idx,
       host_span<streaming_aggregation_request const> requests,
       null_policy null_handling,
       sorted keys_are_sorted,
       std::vector<order> const& column_order,
       std::vector<null_order> const& null_precedence)
    : _key_indices{key_idx.begin(), key_idx.end()},
      _null_handling{null_handling},
      _keys_are_sorted{keys_are_sorted},
      _column_order{column_order},
      _null_precedence{null_precedence}
  {
    _plans = build_request_plans(requests);
    for (auto const& req : requests) {
      _aggs_per_request.push_back(static_cast<size_type>(req.aggregations.size()));
    }
  }

  /**
   * @brief Determine the stable partial-state column types from the first batch's results.
   */
  void initialize_types(std::vector<std::unique_ptr<column>> const& batch_partial_cols)
  {
    _partial_column_types.clear();
    auto col_idx = size_type{0};
    for (auto const& plan : _plans) {
      for (size_type i = 0; i < plan.num_partial_columns; ++i) {
        auto batch_type = batch_partial_cols[col_idx]->type();
        auto state_type = compute_partial_state_type(
          batch_type, plan.partial_descs[i].batch_kind, plan.partial_descs[i].merge_kind);
        _partial_column_types.push_back(state_type);
        ++col_idx;
      }
    }
  }

  /**
   * @brief Cast batch result columns to the established partial-state types where they differ
   * (e.g. COUNT INT32 to INT64 for SUM-based merging). Non-fixed-width columns are passed through.
   */
  std::vector<std::unique_ptr<column>> cast_to_state_types(
    std::vector<std::unique_ptr<column>>& cols,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
  {
    std::vector<std::unique_ptr<column>> result;
    for (size_t i = 0; i < cols.size(); ++i) {
      if (cols[i]->type() != _partial_column_types[i] && cudf::is_fixed_width(cols[i]->type()) &&
          cudf::is_fixed_width(_partial_column_types[i])) {
        result.push_back(cudf::cast(cols[i]->view(), _partial_column_types[i], stream, mr));
      } else {
        result.push_back(std::move(cols[i]));
      }
    }
    return result;
  }

  /**
   * @brief Run a standard cudf::groupby on the incoming data batch.
   *
   * Returns the partial result columns (one per partial_agg_descriptor across all plans)
   * and writes the unique keys to @p out_keys.
   */
  std::vector<std::unique_ptr<column>> run_batch_groupby(table_view const& data,
                                                         std::unique_ptr<table>& out_keys,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
  {
    std::vector<column_view> key_cols;
    key_cols.reserve(num_keys());
    for (auto idx : _key_indices) {
      key_cols.push_back(data.column(idx));
    }
    table_view keys_view{key_cols};

    cudf::groupby::groupby batch_groupby{
      keys_view, _null_handling, _keys_are_sorted, _column_order, _null_precedence};

    std::vector<aggregation_request> batch_requests;
    for (auto const& plan : _plans) {
      for (auto const& desc : plan.partial_descs) {
        aggregation_request req;
        req.values = data.column(plan.value_column_index);
        req.aggregations.push_back(make_batch_agg(desc.batch_kind));
        batch_requests.push_back(std::move(req));
      }
    }

    auto [batch_keys, batch_results] = batch_groupby.aggregate(batch_requests, stream, mr);
    out_keys                         = std::move(batch_keys);

    std::vector<std::unique_ptr<column>> partial_cols;
    for (auto& res : batch_results) {
      for (auto& col : res.results) {
        partial_cols.push_back(std::move(col));
      }
    }
    return partial_cols;
  }

  /**
   * @brief Core aggregate logic: batch groupby, cast, merge with accumulated state.
   */
  void do_aggregate(table_view const& data,
                    rmm::cuda_stream_view stream,
                    rmm::device_async_resource_ref mr)
  {
    std::unique_ptr<table> batch_keys;
    auto batch_partials = run_batch_groupby(data, batch_keys, stream, mr);

    if (!types_initialized()) { initialize_types(batch_partials); }

    auto cast_partials = cast_to_state_types(batch_partials, stream, mr);
    auto batch_state   = build_state_table(std::move(batch_keys), cast_partials);

    if (!_partial_state) {
      _partial_state = std::move(batch_state);
      return;
    }

    merge_partial_states(
      _partial_state, batch_state->view(), num_keys(), _plans, _null_handling, stream, mr);
  }

  /**
   * @brief Verify that @p other has the same configuration (key indices, null handling,
   * sort metadata, and aggregation plan) so their partial states can be merged.
   */
  void validate_compatible(impl const& other) const
  {
    CUDF_EXPECTS(_key_indices == other._key_indices,
                 "Cannot merge streaming_groupby objects with different key indices.",
                 std::invalid_argument);
    CUDF_EXPECTS(_null_handling == other._null_handling,
                 "Cannot merge streaming_groupby objects with different null handling.",
                 std::invalid_argument);
    CUDF_EXPECTS(_keys_are_sorted == other._keys_are_sorted,
                 "Cannot merge streaming_groupby objects with different keys_are_sorted.",
                 std::invalid_argument);
    CUDF_EXPECTS(_column_order == other._column_order,
                 "Cannot merge streaming_groupby objects with different column order.",
                 std::invalid_argument);
    CUDF_EXPECTS(_null_precedence == other._null_precedence,
                 "Cannot merge streaming_groupby objects with different null precedence.",
                 std::invalid_argument);
    CUDF_EXPECTS(_plans.size() == other._plans.size(),
                 "Cannot merge streaming_groupby objects with different aggregation requests.",
                 std::invalid_argument);
    for (size_t i = 0; i < _plans.size(); ++i) {
      CUDF_EXPECTS(_plans[i].requested_kind == other._plans[i].requested_kind &&
                     _plans[i].value_column_index == other._plans[i].value_column_index,
                   "Cannot merge streaming_groupby objects with different aggregation requests.",
                   std::invalid_argument);
    }
  }

  /**
   * @brief Validate compatibility, then merge @p other's partial state into this one.
   */
  void do_merge(impl const& other, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
  {
    validate_compatible(other);
    if (!other._partial_state) { return; }

    if (!_partial_state) {
      _partial_state = std::make_unique<table>(other._partial_state->view(), stream, mr);
      if (!types_initialized() && other.types_initialized()) {
        _partial_column_types = other._partial_column_types;
      }
      return;
    }

    merge_partial_states(
      _partial_state, other._partial_state->view(), num_keys(), _plans, _null_handling, stream, mr);
  }

  /**
   * @brief Copy the accumulated state and apply finalization transforms (identity, mean
   * division, variance/std from M2 struct) to produce the final result columns.
   */
  std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> do_finalize(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
  {
    CUDF_EXPECTS(_partial_state != nullptr,
                 "Cannot finalize streaming_groupby with no accumulated data.");

    std::vector<size_type> key_col_indices(num_keys());
    std::iota(key_col_indices.begin(), key_col_indices.end(), 0);
    auto keys_table =
      std::make_unique<table>(_partial_state->view().select(key_col_indices), stream, mr);

    auto req_idx = size_type{0};
    std::vector<aggregation_result> results;

    for (auto num_aggs : _aggs_per_request) {
      aggregation_result agg_result;
      for (size_type agg_idx = 0; agg_idx < num_aggs; ++agg_idx) {
        auto const& plan = _plans[req_idx];

        switch (plan.finalization) {
          case finalize_kind::IDENTITY: {
            auto const& src = _partial_state->view().column(num_keys() + plan.partial_state_offset);
            agg_result.results.push_back(std::make_unique<column>(src, stream, mr));
            break;
          }
          case finalize_kind::MEAN_FROM_SUM_COUNT: {
            auto sum_col = _partial_state->view().column(num_keys() + plan.partial_state_offset);
            auto count_col =
              _partial_state->view().column(num_keys() + plan.partial_state_offset + 1);

            auto count_as_double = cudf::cast(count_col, data_type{type_id::FLOAT64}, stream, mr);
            auto sum_as_double   = cudf::cast(sum_col, data_type{type_id::FLOAT64}, stream, mr);

            agg_result.results.push_back(cudf::binary_operation(sum_as_double->view(),
                                                                count_as_double->view(),
                                                                binary_operator::DIV,
                                                                data_type{type_id::FLOAT64},
                                                                stream,
                                                                mr));
            break;
          }
          case finalize_kind::VARIANCE_FROM_M2: {
            auto m2_struct = _partial_state->view().column(num_keys() + plan.partial_state_offset);
            auto count_col = m2_struct.child(0);
            auto m2_col    = m2_struct.child(2);

            auto ddof = size_type{1};
            if (auto const* var_agg =
                  dynamic_cast<cudf::detail::var_aggregation const*>(plan.requested_agg.get())) {
              ddof = var_agg->_ddof;
            }

            auto count_dbl = cudf::cast(count_col, data_type{type_id::FLOAT64}, stream, mr);
            auto ddof_scalar =
              cudf::numeric_scalar<double>{static_cast<double>(ddof), true, stream};
            auto denom = cudf::binary_operation(count_dbl->view(),
                                                ddof_scalar,
                                                binary_operator::SUB,
                                                data_type{type_id::FLOAT64},
                                                stream,
                                                mr);
            agg_result.results.push_back(cudf::binary_operation(m2_col,
                                                                denom->view(),
                                                                binary_operator::DIV,
                                                                data_type{type_id::FLOAT64},
                                                                stream,
                                                                mr));
            break;
          }
          case finalize_kind::STD_FROM_M2: {
            auto m2_struct = _partial_state->view().column(num_keys() + plan.partial_state_offset);
            auto count_col = m2_struct.child(0);
            auto m2_col    = m2_struct.child(2);

            auto ddof = size_type{1};
            if (auto const* std_agg =
                  dynamic_cast<cudf::detail::std_aggregation const*>(plan.requested_agg.get())) {
              ddof = std_agg->_ddof;
            }

            auto count_dbl = cudf::cast(count_col, data_type{type_id::FLOAT64}, stream, mr);
            auto ddof_scalar =
              cudf::numeric_scalar<double>{static_cast<double>(ddof), true, stream};
            auto denom    = cudf::binary_operation(count_dbl->view(),
                                                ddof_scalar,
                                                binary_operator::SUB,
                                                data_type{type_id::FLOAT64},
                                                stream,
                                                mr);
            auto variance = cudf::binary_operation(
              m2_col, denom->view(), binary_operator::DIV, data_type{type_id::FLOAT64}, stream, mr);
            agg_result.results.push_back(
              cudf::unary_operation(variance->view(), unary_operator::SQRT, stream, mr));
            break;
          }
        }
        ++req_idx;
      }
      results.push_back(std::move(agg_result));
    }

    return {std::move(keys_table), std::move(results)};
  }

  /**
   * @brief Move out the partial-state table as individual columns and reset internal state.
   */
  std::vector<std::unique_ptr<column>> do_release()
  {
    if (!_partial_state) { return {}; }
    auto cols      = _partial_state->release();
    _partial_state = nullptr;
    _partial_column_types.clear();
    return cols;
  }
};

// -- Public method implementations --

streaming_groupby::streaming_groupby(host_span<size_type const> key_indices,
                                     host_span<streaming_aggregation_request const> requests,
                                     null_policy null_handling,
                                     sorted keys_are_sorted,
                                     std::vector<order> const& column_order,
                                     std::vector<null_order> const& null_precedence)
  : _impl{std::make_unique<impl>(
      key_indices, requests, null_handling, keys_are_sorted, column_order, null_precedence)}
{
}

streaming_groupby::streaming_groupby(std::vector<std::unique_ptr<column>>&& partial_state,
                                     host_span<size_type const> key_indices,
                                     host_span<streaming_aggregation_request const> requests,
                                     null_policy null_handling,
                                     sorted keys_are_sorted,
                                     std::vector<order> const& column_order,
                                     std::vector<null_order> const& null_precedence)
  : _impl{std::make_unique<impl>(
      key_indices, requests, null_handling, keys_are_sorted, column_order, null_precedence)}
{
  if (!partial_state.empty()) {
    auto const expected_cols = _impl->num_keys() + _impl->total_partial_columns();
    CUDF_EXPECTS(static_cast<size_type>(partial_state.size()) == expected_cols,
                 "partial_state column count (" + std::to_string(partial_state.size()) +
                   ") does not match expected (" + std::to_string(expected_cols) +
                   "): " + std::to_string(_impl->num_keys()) + " key columns + " +
                   std::to_string(_impl->total_partial_columns()) + " partial columns.",
                 std::invalid_argument);

    _impl->_partial_state = std::make_unique<table>(std::move(partial_state));
    _impl->_partial_column_types.reserve(_impl->total_partial_columns());
    for (auto i = _impl->num_keys(); i < _impl->num_keys() + _impl->total_partial_columns(); ++i) {
      _impl->_partial_column_types.push_back(_impl->_partial_state->view().column(i).type());
    }
    // types_initialized() is now true via non-empty _partial_column_types
  }
}

streaming_groupby::~streaming_groupby() = default;

streaming_groupby::streaming_groupby(streaming_groupby&&) noexcept = default;

streaming_groupby& streaming_groupby::operator=(streaming_groupby&&) noexcept = default;

void streaming_groupby::aggregate(table_view const& data,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  _impl->do_aggregate(data, stream, mr);
}

void streaming_groupby::merge(streaming_groupby const& other,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  _impl->do_merge(*other._impl, stream, mr);
}

std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> streaming_groupby::finalize(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return _impl->do_finalize(stream, mr);
}

std::vector<std::unique_ptr<column>> streaming_groupby::release() { return _impl->do_release(); }

}  // namespace groupby
}  // namespace cudf
