/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
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
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cudf {
namespace groupby {

namespace {

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
 * @brief One deduplicated partial-state column in the state table.
 *
 * Multiple request_plans may reference the same unique_agg_column if they need the
 * same (value_column_index, batch_kind, merge_kind) combination. For example, both
 * a SUM request and a MEAN request on the same column share a single SUM column.
 *
 * The original_agg clone is retained so parameterized aggregations (e.g. tdigest
 * max_centroids, collect_set null/nan equality) can propagate their options through
 * the batch and merge factory paths.
 */
struct unique_agg_column {
  size_type value_column_index;               ///< Column index in the input data table
  aggregation::Kind batch_kind;               ///< Aggregation to run on each incoming batch
  aggregation::Kind merge_kind;               ///< Aggregation to combine two partial-state columns
  std::shared_ptr<aggregation> original_agg;  ///< Clone of the originating agg (carries params)
  size_type column_index;                     ///< Position in the partial-state table (after keys)

  bool operator==(unique_agg_column const& other) const
  {
    return value_column_index == other.value_column_index && batch_kind == other.batch_kind &&
           merge_kind == other.merge_kind;
  }
};

/**
 * @brief Execution plan for a single requested aggregation.
 *
 * Maps one requested aggregation (e.g. MEAN on column 3) to references into the
 * deduplicated unique_agg_column table, plus finalization logic.
 */
struct request_plan {
  size_type value_column_index;                ///< Column index in the input data table
  aggregation::Kind requested_kind;            ///< The originally requested aggregation kind
  std::unique_ptr<aggregation> requested_agg;  ///< Clone of the original agg (for params like ddof)
  std::vector<size_type> agg_column_refs;      ///< Indices into the shared unique_agg_column list
  finalize_kind finalization;                  ///< How to produce the final result
};

/**
 * @brief Composite key for deduplicating partial-state columns:
 * (value_column_index, batch_kind, merge_kind).
 */
using agg_key_t = std::tuple<size_type, aggregation::Kind, aggregation::Kind>;

/**
 * @brief Hash functor for agg_key_t, using boost-style hash combining.
 */
struct agg_key_hash {
  std::size_t operator()(agg_key_t const& k) const noexcept
  {
    auto h = std::hash<size_type>{}(std::get<0>(k));
    h ^=
      std::hash<int32_t>{}(static_cast<int32_t>(std::get<1>(k))) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^=
      std::hash<int32_t>{}(static_cast<int32_t>(std::get<2>(k))) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
  }
};

/**
 * @brief Maps agg_key_t to the index of the corresponding unique_agg_column.
 *
 * Used during build_plans_and_columns to ensure that identical
 * (value_column_index, batch_kind, merge_kind) combinations share a single
 * partial-state column rather than each request plan allocating its own.
 */
using dedup_map_t = std::unordered_map<agg_key_t, size_type, agg_key_hash>;

/**
 * @brief Register a unique_agg_column, reusing an existing one if the same
 * (value_column_index, batch_kind, merge_kind) combination was already registered.
 *
 * @return The index of the (possibly existing) unique_agg_column.
 */
size_type register_agg_column(size_type value_col,
                              aggregation::Kind batch_kind,
                              aggregation::Kind merge_kind,
                              aggregation const& original_agg,
                              std::vector<unique_agg_column>& columns,
                              dedup_map_t& dedup_map)
{
  auto const key = agg_key_t{value_col, batch_kind, merge_kind};
  auto const it  = dedup_map.find(key);
  if (it != dedup_map.end()) { return it->second; }

  auto const idx = static_cast<size_type>(columns.size());
  columns.push_back(
    {value_col, batch_kind, merge_kind, std::shared_ptr<aggregation>(original_agg.clone()), idx});
  dedup_map[key] = idx;
  return idx;
}

/**
 * @brief Build the list of request_plans and the deduplicated unique_agg_columns
 * from streaming_aggregation_requests.
 *
 * @throws cudf::invalid_argument for unsupported aggregation kinds
 */
std::pair<std::vector<request_plan>, std::vector<unique_agg_column>> build_plans_and_columns(
  host_span<streaming_aggregation_request const> requests)
{
  std::vector<request_plan> plans;
  std::vector<unique_agg_column> agg_columns;
  dedup_map_t dedup_map;

  for (auto const& req : requests) {
    for (auto const& agg : req.aggregations) {
      auto const col = req.column_index;

      request_plan plan;
      plan.value_column_index = col;
      plan.requested_kind     = agg->kind;
      plan.requested_agg      = agg->clone();

      switch (agg->kind) {
        case aggregation::SUM:
          plan.agg_column_refs = {register_agg_column(
            col, aggregation::SUM, aggregation::SUM, *agg, agg_columns, dedup_map)};
          plan.finalization    = finalize_kind::IDENTITY;
          break;
        case aggregation::PRODUCT:
          plan.agg_column_refs = {register_agg_column(
            col, aggregation::PRODUCT, aggregation::PRODUCT, *agg, agg_columns, dedup_map)};
          plan.finalization    = finalize_kind::IDENTITY;
          break;
        case aggregation::MIN:
          plan.agg_column_refs = {register_agg_column(
            col, aggregation::MIN, aggregation::MIN, *agg, agg_columns, dedup_map)};
          plan.finalization    = finalize_kind::IDENTITY;
          break;
        case aggregation::MAX:
          plan.agg_column_refs = {register_agg_column(
            col, aggregation::MAX, aggregation::MAX, *agg, agg_columns, dedup_map)};
          plan.finalization    = finalize_kind::IDENTITY;
          break;
        case aggregation::COUNT_VALID:
          plan.agg_column_refs = {register_agg_column(
            col, aggregation::COUNT_VALID, aggregation::SUM, *agg, agg_columns, dedup_map)};
          plan.finalization    = finalize_kind::IDENTITY;
          break;
        case aggregation::COUNT_ALL:
          plan.agg_column_refs = {register_agg_column(
            col, aggregation::COUNT_ALL, aggregation::SUM, *agg, agg_columns, dedup_map)};
          plan.finalization    = finalize_kind::IDENTITY;
          break;
        case aggregation::MEAN:
          plan.agg_column_refs = {
            register_agg_column(
              col, aggregation::SUM, aggregation::SUM, *agg, agg_columns, dedup_map),
            register_agg_column(
              col, aggregation::COUNT_VALID, aggregation::SUM, *agg, agg_columns, dedup_map)};
          plan.finalization = finalize_kind::MEAN_FROM_SUM_COUNT;
          break;
        case aggregation::SUM_OF_SQUARES:
          plan.agg_column_refs = {register_agg_column(
            col, aggregation::SUM_OF_SQUARES, aggregation::SUM, *agg, agg_columns, dedup_map)};
          plan.finalization    = finalize_kind::IDENTITY;
          break;
        case aggregation::M2:
          plan.agg_column_refs = {register_agg_column(
            col, aggregation::M2, aggregation::MERGE_M2, *agg, agg_columns, dedup_map)};
          plan.finalization    = finalize_kind::IDENTITY;
          break;
        case aggregation::VARIANCE:
          plan.agg_column_refs = {register_agg_column(
            col, aggregation::M2, aggregation::MERGE_M2, *agg, agg_columns, dedup_map)};
          plan.finalization    = finalize_kind::VARIANCE_FROM_M2;
          break;
        case aggregation::STD:
          plan.agg_column_refs = {register_agg_column(
            col, aggregation::M2, aggregation::MERGE_M2, *agg, agg_columns, dedup_map)};
          plan.finalization    = finalize_kind::STD_FROM_M2;
          break;
        case aggregation::TDIGEST:
          plan.agg_column_refs = {register_agg_column(
            col, aggregation::TDIGEST, aggregation::MERGE_TDIGEST, *agg, agg_columns, dedup_map)};
          plan.finalization    = finalize_kind::IDENTITY;
          break;
        case aggregation::MERGE_TDIGEST:
          plan.agg_column_refs = {register_agg_column(col,
                                                      aggregation::MERGE_TDIGEST,
                                                      aggregation::MERGE_TDIGEST,
                                                      *agg,
                                                      agg_columns,
                                                      dedup_map)};
          plan.finalization    = finalize_kind::IDENTITY;
          break;
        case aggregation::HISTOGRAM:
          plan.agg_column_refs = {register_agg_column(col,
                                                      aggregation::HISTOGRAM,
                                                      aggregation::MERGE_HISTOGRAM,
                                                      *agg,
                                                      agg_columns,
                                                      dedup_map)};
          plan.finalization    = finalize_kind::IDENTITY;
          break;
        case aggregation::MERGE_HISTOGRAM:
          plan.agg_column_refs = {register_agg_column(col,
                                                      aggregation::MERGE_HISTOGRAM,
                                                      aggregation::MERGE_HISTOGRAM,
                                                      *agg,
                                                      agg_columns,
                                                      dedup_map)};
          plan.finalization    = finalize_kind::IDENTITY;
          break;
        case aggregation::COLLECT_LIST:
          plan.agg_column_refs = {register_agg_column(col,
                                                      aggregation::COLLECT_LIST,
                                                      aggregation::MERGE_LISTS,
                                                      *agg,
                                                      agg_columns,
                                                      dedup_map)};
          plan.finalization    = finalize_kind::IDENTITY;
          break;
        case aggregation::COLLECT_SET:
          plan.agg_column_refs = {register_agg_column(
            col, aggregation::COLLECT_SET, aggregation::MERGE_SETS, *agg, agg_columns, dedup_map)};
          plan.finalization    = finalize_kind::IDENTITY;
          break;
        case aggregation::MERGE_LISTS:
          plan.agg_column_refs = {register_agg_column(
            col, aggregation::MERGE_LISTS, aggregation::MERGE_LISTS, *agg, agg_columns, dedup_map)};
          plan.finalization    = finalize_kind::IDENTITY;
          break;
        case aggregation::MERGE_SETS:
          plan.agg_column_refs = {register_agg_column(
            col, aggregation::MERGE_SETS, aggregation::MERGE_SETS, *agg, agg_columns, dedup_map)};
          plan.finalization    = finalize_kind::IDENTITY;
          break;
        default:
          CUDF_FAIL(
            "Unsupported aggregation kind for streaming groupby: " + std::to_string(agg->kind),
            std::invalid_argument);
      }
      plans.push_back(std::move(plan));
    }
  }
  return {std::move(plans), std::move(agg_columns)};
}

/**
 * @brief Create a groupby_aggregation to run on an incoming data batch.
 *
 * @p original carries parameters from the user-facing aggregation so that parameterized
 * kinds (tdigest max_centroids, collect_set null/nan equality, etc.) are preserved.
 */
std::unique_ptr<groupby_aggregation> make_batch_agg(aggregation::Kind kind,
                                                    aggregation const& original)
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
    case aggregation::TDIGEST: {
      auto const max_c =
        dynamic_cast<cudf::detail::tdigest_aggregation const&>(original).max_centroids;
      return make_tdigest_aggregation<groupby_aggregation>(max_c);
    }
    case aggregation::MERGE_TDIGEST: {
      auto const max_c =
        dynamic_cast<cudf::detail::merge_tdigest_aggregation const&>(original).max_centroids;
      return make_merge_tdigest_aggregation<groupby_aggregation>(max_c);
    }
    case aggregation::HISTOGRAM: return make_histogram_aggregation<groupby_aggregation>();
    case aggregation::MERGE_HISTOGRAM:
      return make_merge_histogram_aggregation<groupby_aggregation>();
    case aggregation::COLLECT_LIST: {
      auto const& src = dynamic_cast<cudf::detail::collect_list_aggregation const&>(original);
      return make_collect_list_aggregation<groupby_aggregation>(src._null_handling);
    }
    case aggregation::COLLECT_SET: {
      auto const& src = dynamic_cast<cudf::detail::collect_set_aggregation const&>(original);
      return make_collect_set_aggregation<groupby_aggregation>(
        src._null_handling, src._nulls_equal, src._nans_equal);
    }
    case aggregation::MERGE_LISTS: return make_merge_lists_aggregation<groupby_aggregation>();
    case aggregation::MERGE_SETS: {
      auto const& src = dynamic_cast<cudf::detail::merge_sets_aggregation const&>(original);
      return make_merge_sets_aggregation<groupby_aggregation>(src._nulls_equal, src._nans_equal);
    }
    default: CUDF_FAIL("Unsupported batch aggregation kind for streaming groupby");
  }
}

/**
 * @brief Create a groupby_aggregation to merge two partial-state columns.
 *
 * @p original carries parameters from the user-facing aggregation so that parameterized
 * merge kinds (merge_tdigest max_centroids, merge_sets null/nan equality) are preserved.
 */
std::unique_ptr<groupby_aggregation> make_merge_agg(aggregation::Kind kind,
                                                    aggregation const& original)
{
  switch (kind) {
    case aggregation::SUM: return make_sum_aggregation<groupby_aggregation>();
    case aggregation::PRODUCT: return make_product_aggregation<groupby_aggregation>();
    case aggregation::MIN: return make_min_aggregation<groupby_aggregation>();
    case aggregation::MAX: return make_max_aggregation<groupby_aggregation>();
    case aggregation::MERGE_M2: return make_merge_m2_aggregation<groupby_aggregation>();
    case aggregation::MERGE_TDIGEST: {
      int max_c = 1000;
      if (auto const* td = dynamic_cast<cudf::detail::tdigest_aggregation const*>(&original)) {
        max_c = td->max_centroids;
      } else if (auto const* mtd =
                   dynamic_cast<cudf::detail::merge_tdigest_aggregation const*>(&original)) {
        max_c = mtd->max_centroids;
      }
      return make_merge_tdigest_aggregation<groupby_aggregation>(max_c);
    }
    case aggregation::MERGE_HISTOGRAM:
      return make_merge_histogram_aggregation<groupby_aggregation>();
    case aggregation::MERGE_LISTS: return make_merge_lists_aggregation<groupby_aggregation>();
    case aggregation::MERGE_SETS: {
      auto null_eq = null_equality::EQUAL;
      auto nan_eq  = nan_equality::ALL_EQUAL;
      if (auto const* cs = dynamic_cast<cudf::detail::collect_set_aggregation const*>(&original)) {
        null_eq = cs->_nulls_equal;
        nan_eq  = cs->_nans_equal;
      } else if (auto const* ms =
                   dynamic_cast<cudf::detail::merge_sets_aggregation const*>(&original)) {
        null_eq = ms->_nulls_equal;
        nan_eq  = ms->_nans_equal;
      }
      return make_merge_sets_aggregation<groupby_aggregation>(null_eq, nan_eq);
    }
    default: CUDF_FAIL("Unsupported merge aggregation kind for streaming groupby");
  }
}

/**
 * @brief Determine the data type of a partial-state column after merging.
 *
 * For non-fixed-width batch results (structs from M2, tdigest; lists from collect/histogram)
 * the batch output is already in the correct form for the merge aggregation, so we return
 * it directly rather than calling target_type (which cannot dispatch on STRUCT/LIST sources).
 */
data_type compute_partial_state_type(data_type batch_result_type,
                                     aggregation::Kind batch_kind,
                                     aggregation::Kind merge_kind)
{
  if (batch_kind == merge_kind) { return batch_result_type; }
  if (!cudf::is_fixed_width(batch_result_type)) { return batch_result_type; }
  return cudf::detail::target_type(batch_result_type, merge_kind);
}

/**
 * @brief Merge new partial results into the accumulated state.
 *
 * Concatenates @p accumulated and @p new_partials vertically, then runs a groupby
 * with the appropriate merge aggregations (one per unique_agg_column) to collapse
 * duplicate keys back to one row per group.
 */
table_view columns_as_table_view(std::vector<std::unique_ptr<column>> const& cols)
{
  std::vector<column_view> views;
  views.reserve(cols.size());
  for (auto const& c : cols) {
    views.push_back(c->view());
  }
  return table_view{views};
}

void merge_partial_states(std::vector<std::unique_ptr<column>>& accumulated,
                          table_view new_partials,
                          size_type num_keys,
                          std::vector<unique_agg_column> const& agg_columns,
                          null_policy null_handling,
                          rmm::cuda_stream_view stream,
                          rmm::device_async_resource_ref mr)
{
  auto const acc_view = columns_as_table_view(accumulated);
  std::vector<table_view> const to_concat{acc_view, new_partials};
  auto const concatenated = cudf::concatenate(to_concat, stream, mr);

  std::vector<size_type> key_col_indices(num_keys);
  std::iota(key_col_indices.begin(), key_col_indices.end(), 0);
  auto const concat_keys = concatenated->view().select(key_col_indices);

  cudf::groupby::groupby merge_groupby{concat_keys, null_handling};

  std::vector<aggregation_request> merge_requests;
  for (auto const& agg_col : agg_columns) {
    aggregation_request req;
    req.values = concatenated->view().column(num_keys + agg_col.column_index);
    req.aggregations.push_back(make_merge_agg(agg_col.merge_kind, *agg_col.original_agg));
    merge_requests.push_back(std::move(req));
  }

  auto [merged_keys, merged_results] = merge_groupby.aggregate(merge_requests, stream, mr);

  std::vector<std::unique_ptr<column>> merged_partials;
  for (auto& res : merged_results) {
    for (auto& col : res.results) {
      merged_partials.push_back(std::move(col));
    }
  }

  accumulated = merged_keys->release();
  for (auto& c : merged_partials) {
    accumulated.push_back(std::move(c));
  }
}

}  // namespace

struct streaming_groupby::impl {
  std::vector<size_type> _key_indices;
  std::vector<size_type> _aggs_per_request;
  std::vector<request_plan> _plans;
  std::vector<unique_agg_column> _agg_columns;
  null_policy _null_handling;
  sorted _keys_are_sorted;
  std::vector<order> _column_order;
  std::vector<null_order> _null_precedence;

  std::vector<std::unique_ptr<column>> _partial_columns;
  std::vector<data_type> _partial_column_types;
  std::vector<data_type> _finalize_output_types;

  /**
   * @brief Whether partial-state column types have been determined from the first batch.
   */
  [[nodiscard]] bool types_initialized() const { return !_partial_column_types.empty(); }

  [[nodiscard]] bool has_state() const { return !_partial_columns.empty(); }

  [[nodiscard]] table_view state_as_table_view() const
  {
    std::vector<column_view> views;
    views.reserve(_partial_columns.size());
    for (auto const& c : _partial_columns) {
      views.push_back(c->view());
    }
    return table_view{views};
  }

  /**
   * @brief Number of key columns.
   */
  [[nodiscard]] size_type num_keys() const { return static_cast<size_type>(_key_indices.size()); }

  /**
   * @brief Number of deduplicated partial-state columns.
   */
  [[nodiscard]] size_type num_agg_columns() const
  {
    return static_cast<size_type>(_agg_columns.size());
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
    auto [plans, agg_columns] = build_plans_and_columns(requests);
    _plans                    = std::move(plans);
    _agg_columns              = std::move(agg_columns);
    for (auto const& req : requests) {
      _aggs_per_request.push_back(static_cast<size_type>(req.aggregations.size()));
    }
  }

  /**
   * @brief Determine partial-state column types and expected finalize output types.
   */
  void initialize_types(std::vector<std::unique_ptr<column>> const& batch_partial_cols,
                        table_view const& data)
  {
    _partial_column_types.clear();
    _finalize_output_types.clear();

    for (size_type i = 0; i < num_agg_columns(); ++i) {
      auto const& agg_col   = _agg_columns[i];
      auto const batch_type = batch_partial_cols[i]->type();
      auto const state_type =
        compute_partial_state_type(batch_type, agg_col.batch_kind, agg_col.merge_kind);
      _partial_column_types.push_back(state_type);
    }

    size_type plan_idx = 0;
    for (auto const& plan : _plans) {
      auto const& state_col = batch_partial_cols[plan.agg_column_refs[0]];
      if (!cudf::is_fixed_width(state_col->type())) {
        _finalize_output_types.push_back(state_col->type());
      } else {
        _finalize_output_types.push_back(cudf::detail::target_type(
          data.column(plan.value_column_index).type(), plan.requested_kind));
      }
      ++plan_idx;
    }
  }

  /**
   * @brief Cast batch result columns to the established partial-state types where they differ.
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
   * Creates one aggregation_request per unique_agg_column (deduplicated).
   * For M2 columns, three separate aggregations (COUNT_VALID, MEAN, M2) are run
   * and assembled into a struct column suitable for MERGE_M2.
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
    table_view const keys_view{key_cols};

    cudf::groupby::groupby batch_groupby{
      keys_view, _null_handling, _keys_are_sorted, _column_order, _null_precedence};

    struct m2_assembly_info {
      size_type request_index;
      size_type count_result_index;
      size_type mean_result_index;
      size_type m2_result_index;
    };
    std::vector<m2_assembly_info> m2_assemblies;

    std::vector<aggregation_request> batch_requests;
    std::vector<size_type> result_to_agg_col;

    for (size_type ac = 0; ac < num_agg_columns(); ++ac) {
      auto const& agg_col = _agg_columns[ac];

      if (agg_col.batch_kind == aggregation::M2) {
        auto const req_idx = static_cast<size_type>(batch_requests.size());
        {
          aggregation_request req;
          req.values = data.column(agg_col.value_column_index);
          req.aggregations.push_back(
            make_count_aggregation<groupby_aggregation>(null_policy::EXCLUDE));
          batch_requests.push_back(std::move(req));
        }
        auto const count_idx = static_cast<size_type>(batch_requests.size() - 1);
        {
          aggregation_request req;
          req.values = data.column(agg_col.value_column_index);
          req.aggregations.push_back(make_mean_aggregation<groupby_aggregation>());
          batch_requests.push_back(std::move(req));
        }
        auto const mean_idx = static_cast<size_type>(batch_requests.size() - 1);
        {
          aggregation_request req;
          req.values = data.column(agg_col.value_column_index);
          req.aggregations.push_back(make_m2_aggregation<groupby_aggregation>());
          batch_requests.push_back(std::move(req));
        }
        auto const m2_idx = static_cast<size_type>(batch_requests.size() - 1);

        m2_assemblies.push_back({req_idx, count_idx, mean_idx, m2_idx});
        result_to_agg_col.push_back(ac);
      } else {
        aggregation_request req;
        req.values = data.column(agg_col.value_column_index);
        req.aggregations.push_back(make_batch_agg(agg_col.batch_kind, *agg_col.original_agg));
        batch_requests.push_back(std::move(req));
        result_to_agg_col.push_back(ac);
      }
    }

    auto [batch_keys, batch_results] = batch_groupby.aggregate(batch_requests, stream, mr);
    out_keys                         = std::move(batch_keys);

    if (m2_assemblies.empty()) {
      std::vector<std::unique_ptr<column>> partial_cols;
      for (auto& res : batch_results) {
        for (auto& col : res.results) {
          partial_cols.push_back(std::move(col));
        }
      }
      return partial_cols;
    }

    std::unordered_set<size_type> m2_consumed_indices;
    for (auto const& info : m2_assemblies) {
      m2_consumed_indices.insert(info.count_result_index);
      m2_consumed_indices.insert(info.mean_result_index);
      m2_consumed_indices.insert(info.m2_result_index);
    }

    std::vector<std::unique_ptr<column>> partial_cols;
    size_type m2_assembly_idx = 0;
    for (size_type ri = 0; ri < static_cast<size_type>(batch_results.size()); ++ri) {
      if (m2_consumed_indices.count(ri) > 0) {
        if (m2_assembly_idx < static_cast<size_type>(m2_assemblies.size()) &&
            m2_assemblies[m2_assembly_idx].count_result_index == ri) {
          auto const& info = m2_assemblies[m2_assembly_idx];
          auto count_col   = std::move(batch_results[info.count_result_index].results[0]);
          auto mean_col    = std::move(batch_results[info.mean_result_index].results[0]);
          auto m2_col      = std::move(batch_results[info.m2_result_index].results[0]);
          auto const nrows = count_col->size();

          auto count_f64 = cudf::cast(count_col->view(), data_type{type_id::FLOAT64}, stream, mr);

          std::vector<std::unique_ptr<column>> children;
          children.push_back(std::move(count_f64));
          children.push_back(std::move(mean_col));
          children.push_back(std::move(m2_col));
          partial_cols.push_back(
            make_structs_column(nrows, std::move(children), 0, rmm::device_buffer{}, stream, mr));
          ++m2_assembly_idx;
        }
        continue;
      }
      for (auto& col : batch_results[ri].results) {
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

    if (!types_initialized()) { initialize_types(batch_partials, data); }

    auto cast_partials = cast_to_state_types(batch_partials, stream, mr);

    auto batch_cols = batch_keys->release();
    for (auto& c : cast_partials) {
      batch_cols.push_back(std::move(c));
    }

    if (!has_state()) {
      _partial_columns = std::move(batch_cols);
      return;
    }

    auto batch_state = std::make_unique<table>(std::move(batch_cols));
    merge_partial_states(
      _partial_columns, batch_state->view(), num_keys(), _agg_columns, _null_handling, stream, mr);
  }

  /**
   * @brief Verify that @p other has the same configuration so their partial states can be merged.
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
    CUDF_EXPECTS(_agg_columns.size() == other._agg_columns.size(),
                 "Cannot merge streaming_groupby objects with different aggregation layouts.",
                 std::invalid_argument);
    for (size_t i = 0; i < _agg_columns.size(); ++i) {
      CUDF_EXPECTS(_agg_columns[i] == other._agg_columns[i],
                   "Cannot merge streaming_groupby objects with different aggregation layouts.",
                   std::invalid_argument);
    }
  }

  /**
   * @brief Validate compatibility, then merge @p other's partial state into this one.
   */
  void do_merge(impl const& other, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
  {
    validate_compatible(other);
    if (!other.has_state()) { return; }

    if (!has_state()) {
      auto const other_view = other.state_as_table_view();
      auto copied           = std::make_unique<table>(other_view, stream, mr);
      _partial_columns      = copied->release();
      if (!types_initialized() && other.types_initialized()) {
        _partial_column_types  = other._partial_column_types;
        _finalize_output_types = other._finalize_output_types;
      }
      return;
    }

    auto const other_view = other.state_as_table_view();
    merge_partial_states(
      _partial_columns, other_view, num_keys(), _agg_columns, _null_handling, stream, mr);
  }

  /**
   * @brief Copy the accumulated state and apply finalization transforms to produce final results.
   */
  std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> do_finalize(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
  {
    CUDF_EXPECTS(has_state(), "Cannot finalize streaming_groupby with no accumulated data.");

    std::vector<column_view> key_views;
    key_views.reserve(num_keys());
    for (size_type i = 0; i < num_keys(); ++i) {
      key_views.push_back(_partial_columns[i]->view());
    }
    auto keys_table = std::make_unique<table>(table_view{key_views}, stream, mr);

    auto req_idx = size_type{0};
    std::vector<aggregation_result> results;

    for (auto num_aggs : _aggs_per_request) {
      aggregation_result agg_result;
      for (size_type agg_idx = 0; agg_idx < num_aggs; ++agg_idx) {
        auto const& plan = _plans[req_idx];

        switch (plan.finalization) {
          case finalize_kind::IDENTITY: {
            auto const state_col_idx = _agg_columns[plan.agg_column_refs[0]].column_index;
            auto const& src_col      = *_partial_columns[num_keys() + state_col_idx];
            if (req_idx < static_cast<size_type>(_finalize_output_types.size()) &&
                src_col.type() != _finalize_output_types[req_idx] &&
                cudf::is_fixed_width(src_col.type()) &&
                cudf::is_fixed_width(_finalize_output_types[req_idx])) {
              agg_result.results.push_back(
                cudf::cast(src_col.view(), _finalize_output_types[req_idx], stream, mr));
            } else {
              agg_result.results.push_back(std::make_unique<column>(src_col, stream, mr));
            }
            break;
          }
          case finalize_kind::MEAN_FROM_SUM_COUNT: {
            auto const sum_col_idx   = _agg_columns[plan.agg_column_refs[0]].column_index;
            auto const count_col_idx = _agg_columns[plan.agg_column_refs[1]].column_index;
            auto const sum_col       = _partial_columns[num_keys() + sum_col_idx]->view();
            auto const count_col     = _partial_columns[num_keys() + count_col_idx]->view();

            auto const count_as_double =
              cudf::cast(count_col, data_type{type_id::FLOAT64}, stream, mr);
            auto const sum_as_double = cudf::cast(sum_col, data_type{type_id::FLOAT64}, stream, mr);

            agg_result.results.push_back(cudf::binary_operation(sum_as_double->view(),
                                                                count_as_double->view(),
                                                                binary_operator::DIV,
                                                                data_type{type_id::FLOAT64},
                                                                stream,
                                                                mr));
            break;
          }
          case finalize_kind::VARIANCE_FROM_M2: {
            auto const m2_col_idx = _agg_columns[plan.agg_column_refs[0]].column_index;
            auto const m2_struct  = _partial_columns[num_keys() + m2_col_idx]->view();
            auto const count_col  = m2_struct.child(0);
            auto const m2_col     = m2_struct.child(2);

            auto ddof = size_type{1};
            if (auto const* var_agg =
                  dynamic_cast<cudf::detail::var_aggregation const*>(plan.requested_agg.get())) {
              ddof = var_agg->_ddof;
            }

            auto const count_as_double =
              cudf::cast(count_col, data_type{type_id::FLOAT64}, stream, mr);
            auto const ddof_scalar =
              cudf::numeric_scalar<double>{static_cast<double>(ddof), true, stream};
            auto const denominator  = cudf::binary_operation(count_as_double->view(),
                                                            ddof_scalar,
                                                            binary_operator::SUB,
                                                            data_type{type_id::FLOAT64},
                                                            stream,
                                                            mr);
            auto const zero_scalar  = cudf::numeric_scalar<double>{0.0, true, stream};
            auto const valid_mask   = cudf::binary_operation(denominator->view(),
                                                           zero_scalar,
                                                           binary_operator::GREATER,
                                                           data_type{type_id::BOOL8},
                                                           stream,
                                                           mr);
            auto const raw_variance = cudf::binary_operation(m2_col,
                                                             denominator->view(),
                                                             binary_operator::DIV,
                                                             data_type{type_id::FLOAT64},
                                                             stream,
                                                             mr);
            auto const null_scalar  = cudf::numeric_scalar<double>{0.0, false, stream};
            agg_result.results.push_back(cudf::copy_if_else(
              raw_variance->view(), null_scalar, valid_mask->view(), stream, mr));
            break;
          }
          case finalize_kind::STD_FROM_M2: {
            auto const m2_col_idx = _agg_columns[plan.agg_column_refs[0]].column_index;
            auto const m2_struct  = _partial_columns[num_keys() + m2_col_idx]->view();
            auto const count_col  = m2_struct.child(0);
            auto const m2_col     = m2_struct.child(2);

            auto ddof = size_type{1};
            if (auto const* std_agg =
                  dynamic_cast<cudf::detail::std_aggregation const*>(plan.requested_agg.get())) {
              ddof = std_agg->_ddof;
            }

            auto const count_as_double =
              cudf::cast(count_col, data_type{type_id::FLOAT64}, stream, mr);
            auto const ddof_scalar =
              cudf::numeric_scalar<double>{static_cast<double>(ddof), true, stream};
            auto const denominator = cudf::binary_operation(count_as_double->view(),
                                                            ddof_scalar,
                                                            binary_operator::SUB,
                                                            data_type{type_id::FLOAT64},
                                                            stream,
                                                            mr);
            auto const zero_scalar = cudf::numeric_scalar<double>{0.0, true, stream};
            auto const valid_mask  = cudf::binary_operation(denominator->view(),
                                                           zero_scalar,
                                                           binary_operator::GREATER,
                                                           data_type{type_id::BOOL8},
                                                           stream,
                                                           mr);
            auto const variance    = cudf::binary_operation(m2_col,
                                                         denominator->view(),
                                                         binary_operator::DIV,
                                                         data_type{type_id::FLOAT64},
                                                         stream,
                                                         mr);
            auto const raw_std =
              cudf::unary_operation(variance->view(), unary_operator::SQRT, stream, mr);
            auto const null_scalar = cudf::numeric_scalar<double>{0.0, false, stream};
            agg_result.results.push_back(
              cudf::copy_if_else(raw_std->view(), null_scalar, valid_mask->view(), stream, mr));
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
    if (!has_state()) { return {}; }
    auto cols = std::move(_partial_columns);
    _partial_column_types.clear();
    return cols;
  }
};

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
    auto const expected_cols = static_cast<size_type>(_impl->num_keys() + _impl->num_agg_columns());
    CUDF_EXPECTS(static_cast<size_type>(partial_state.size()) == expected_cols,
                 "partial_state column count (" + std::to_string(partial_state.size()) +
                   ") does not match expected (" + std::to_string(expected_cols) +
                   "): " + std::to_string(_impl->num_keys()) + " key columns + " +
                   std::to_string(_impl->num_agg_columns()) + " partial columns.",
                 std::invalid_argument);

    _impl->_partial_columns = std::move(partial_state);
    _impl->_partial_column_types.reserve(_impl->num_agg_columns());
    for (auto i = _impl->num_keys(); i < _impl->num_keys() + _impl->num_agg_columns(); ++i) {
      _impl->_partial_column_types.push_back(_impl->_partial_columns[i]->type());
    }
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
