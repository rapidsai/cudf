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

#include <memory>
#include <numeric>
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
           merge_kind == other.merge_kind && original_agg->is_equal(*other.original_agg);
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
 * @brief Register a unique_agg_column, reusing an existing one if a fully matching
 * entry (value_column_index, batch_kind, merge_kind, and aggregation parameters)
 * was already registered.
 *
 * @return The index of the (possibly existing) unique_agg_column.
 */
size_type register_agg_column(size_type value_column,
                              aggregation::Kind batch_kind,
                              aggregation::Kind merge_kind,
                              aggregation const& original_agg,
                              std::vector<unique_agg_column>& columns)
{
  for (size_type i = 0; i < static_cast<size_type>(columns.size()); ++i) {
    auto const& existing = columns[i];
    if (existing.value_column_index == value_column && existing.batch_kind == batch_kind &&
        existing.merge_kind == merge_kind && existing.original_agg->is_equal(original_agg)) {
      return i;
    }
  }

  auto const index = static_cast<size_type>(columns.size());
  columns.push_back({value_column,
                     batch_kind,
                     merge_kind,
                     std::shared_ptr<aggregation>(original_agg.clone()),
                     index});
  return index;
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

  for (auto const& request : requests) {
    for (auto const& agg : request.aggregations) {
      auto const column_idx = request.column_index;

      request_plan plan;
      plan.value_column_index = column_idx;
      plan.requested_kind     = agg->kind;
      plan.requested_agg      = agg->clone();

      switch (agg->kind) {
        case aggregation::SUM:
        case aggregation::PRODUCT:
        case aggregation::MIN:
        case aggregation::MAX:
          plan.agg_column_refs = {
            register_agg_column(column_idx, agg->kind, agg->kind, *agg, agg_columns)};
          plan.finalization = finalize_kind::IDENTITY;
          break;
        case aggregation::COUNT_VALID:
        case aggregation::COUNT_ALL:
          plan.agg_column_refs = {
            register_agg_column(column_idx, agg->kind, aggregation::SUM, *agg, agg_columns)};
          plan.finalization = finalize_kind::IDENTITY;
          break;
        case aggregation::MEAN:
          plan.agg_column_refs = {
            register_agg_column(column_idx, aggregation::SUM, aggregation::SUM, *agg, agg_columns),
            register_agg_column(
              column_idx, aggregation::COUNT_VALID, aggregation::SUM, *agg, agg_columns)};
          plan.finalization = finalize_kind::MEAN_FROM_SUM_COUNT;
          break;
        case aggregation::SUM_OF_SQUARES:
          plan.agg_column_refs = {register_agg_column(
            column_idx, aggregation::SUM_OF_SQUARES, aggregation::SUM, *agg, agg_columns)};
          plan.finalization    = finalize_kind::IDENTITY;
          break;
        case aggregation::M2:
        case aggregation::VARIANCE:
        case aggregation::STD:
          plan.agg_column_refs = {register_agg_column(
            column_idx, aggregation::M2, aggregation::MERGE_M2, *agg, agg_columns)};
          plan.finalization = (agg->kind == aggregation::VARIANCE) ? finalize_kind::VARIANCE_FROM_M2
                              : (agg->kind == aggregation::STD)    ? finalize_kind::STD_FROM_M2
                                                                   : finalize_kind::IDENTITY;
          break;
        case aggregation::TDIGEST:
          plan.agg_column_refs = {register_agg_column(
            column_idx, aggregation::TDIGEST, aggregation::MERGE_TDIGEST, *agg, agg_columns)};
          plan.finalization    = finalize_kind::IDENTITY;
          break;
        case aggregation::HISTOGRAM:
          plan.agg_column_refs = {register_agg_column(
            column_idx, aggregation::HISTOGRAM, aggregation::MERGE_HISTOGRAM, *agg, agg_columns)};
          plan.finalization    = finalize_kind::IDENTITY;
          break;
        case aggregation::COLLECT_LIST:
          plan.agg_column_refs = {register_agg_column(
            column_idx, aggregation::COLLECT_LIST, aggregation::MERGE_LISTS, *agg, agg_columns)};
          plan.finalization    = finalize_kind::IDENTITY;
          break;
        case aggregation::COLLECT_SET:
          plan.agg_column_refs = {register_agg_column(
            column_idx, aggregation::COLLECT_SET, aggregation::MERGE_SETS, *agg, agg_columns)};
          plan.finalization    = finalize_kind::IDENTITY;
          break;
        case aggregation::MERGE_TDIGEST:
        case aggregation::MERGE_HISTOGRAM:
        case aggregation::MERGE_LISTS:
        case aggregation::MERGE_SETS:
          plan.agg_column_refs = {
            register_agg_column(column_idx, agg->kind, agg->kind, *agg, agg_columns)};
          plan.finalization = finalize_kind::IDENTITY;
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
      auto const max_centroids =
        dynamic_cast<cudf::detail::tdigest_aggregation const&>(original).max_centroids;
      return make_tdigest_aggregation<groupby_aggregation>(max_centroids);
    }
    case aggregation::MERGE_TDIGEST: {
      auto const max_centroids =
        dynamic_cast<cudf::detail::merge_tdigest_aggregation const&>(original).max_centroids;
      return make_merge_tdigest_aggregation<groupby_aggregation>(max_centroids);
    }
    case aggregation::HISTOGRAM: return make_histogram_aggregation<groupby_aggregation>();
    case aggregation::MERGE_HISTOGRAM:
      return make_merge_histogram_aggregation<groupby_aggregation>();
    case aggregation::COLLECT_LIST: {
      auto const& source_agg =
        dynamic_cast<cudf::detail::collect_list_aggregation const&>(original);
      return make_collect_list_aggregation<groupby_aggregation>(source_agg._null_handling);
    }
    case aggregation::COLLECT_SET: {
      auto const& source_agg = dynamic_cast<cudf::detail::collect_set_aggregation const&>(original);
      return make_collect_set_aggregation<groupby_aggregation>(
        source_agg._null_handling, source_agg._nulls_equal, source_agg._nans_equal);
    }
    case aggregation::MERGE_LISTS: return make_merge_lists_aggregation<groupby_aggregation>();
    case aggregation::MERGE_SETS: {
      auto const& source_agg = dynamic_cast<cudf::detail::merge_sets_aggregation const&>(original);
      return make_merge_sets_aggregation<groupby_aggregation>(source_agg._nulls_equal,
                                                              source_agg._nans_equal);
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
      int max_centroids = 1000;
      if (auto const* tdigest_agg =
            dynamic_cast<cudf::detail::tdigest_aggregation const*>(&original)) {
        max_centroids = tdigest_agg->max_centroids;
      } else if (auto const* merge_tdigest_agg =
                   dynamic_cast<cudf::detail::merge_tdigest_aggregation const*>(&original)) {
        max_centroids = merge_tdigest_agg->max_centroids;
      }
      return make_merge_tdigest_aggregation<groupby_aggregation>(max_centroids);
    }
    case aggregation::MERGE_HISTOGRAM:
      return make_merge_histogram_aggregation<groupby_aggregation>();
    case aggregation::MERGE_LISTS: return make_merge_lists_aggregation<groupby_aggregation>();
    case aggregation::MERGE_SETS: {
      auto null_equality_val = null_equality::EQUAL;
      auto nan_equality_val  = nan_equality::ALL_EQUAL;
      if (auto const* collect_set_agg =
            dynamic_cast<cudf::detail::collect_set_aggregation const*>(&original)) {
        null_equality_val = collect_set_agg->_nulls_equal;
        nan_equality_val  = collect_set_agg->_nans_equal;
      } else if (auto const* merge_sets_agg =
                   dynamic_cast<cudf::detail::merge_sets_aggregation const*>(&original)) {
        null_equality_val = merge_sets_agg->_nulls_equal;
        nan_equality_val  = merge_sets_agg->_nans_equal;
      }
      return make_merge_sets_aggregation<groupby_aggregation>(null_equality_val, nan_equality_val);
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

table_view columns_as_table_view(std::vector<std::unique_ptr<column>> const& columns)
{
  std::vector<column_view> views;
  views.reserve(columns.size());
  for (auto const& column_ptr : columns) {
    views.push_back(column_ptr->view());
  }
  return table_view{views};
}

/**
 * @brief Merge new partial results into the accumulated state.
 *
 * Concatenates @p accumulated and @p new_partials vertically, then runs a groupby
 * with the appropriate merge aggregations (one per unique_agg_column) to collapse
 * duplicate keys back to one row per group.
 */
void merge_partial_states(std::vector<std::unique_ptr<column>>& accumulated,
                          table_view new_partials,
                          size_type num_keys,
                          std::vector<unique_agg_column> const& agg_columns,
                          null_policy null_handling,
                          rmm::cuda_stream_view stream,
                          rmm::device_async_resource_ref mr)
{
  if (new_partials.num_rows() == 0) { return; }

  auto const accumulated_view = columns_as_table_view(accumulated);
  if (accumulated_view.num_rows() == 0) {
    auto copied = std::make_unique<table>(new_partials, stream, mr);
    accumulated = copied->release();
    return;
  }

  std::vector<table_view> const to_concat{accumulated_view, new_partials};
  auto const concatenated = cudf::concatenate(to_concat, stream, mr);

  std::vector<size_type> key_col_indices(num_keys);
  std::iota(key_col_indices.begin(), key_col_indices.end(), 0);
  auto const concat_keys = concatenated->view().select(key_col_indices);

  cudf::groupby::groupby merge_groupby{concat_keys, null_handling};

  std::vector<aggregation_request> merge_requests;
  for (auto const& agg_column : agg_columns) {
    aggregation_request request;
    request.values = concatenated->view().column(num_keys + agg_column.column_index);
    request.aggregations.push_back(make_merge_agg(agg_column.merge_kind, *agg_column.original_agg));
    merge_requests.push_back(std::move(request));
  }

  auto [merged_keys, merged_results] = merge_groupby.aggregate(merge_requests, stream, mr);

  accumulated = merged_keys->release();
  accumulated.reserve(accumulated.size() + agg_columns.size());
  for (auto& result : merged_results) {
    for (auto& result_column : result.results) {
      accumulated.push_back(std::move(result_column));
    }
  }
}

/**
 * @brief Compute variance or standard deviation from an M2 struct column.
 */
std::unique_ptr<column> finalize_variance_or_std(column_view const& m2_struct,
                                                 size_type ddof,
                                                 bool take_sqrt,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  auto const count_col = m2_struct.child(0);
  auto const m2_col    = m2_struct.child(2);

  auto const count_as_double = cudf::cast(count_col, data_type{type_id::FLOAT64}, stream, mr);
  auto const ddof_scalar = cudf::numeric_scalar<double>{static_cast<double>(ddof), true, stream};
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
  auto const raw_result  = cudf::binary_operation(
    m2_col, denominator->view(), binary_operator::DIV, data_type{type_id::FLOAT64}, stream, mr);
  auto const null_scalar = cudf::numeric_scalar<double>{0.0, false, stream};

  if (take_sqrt) {
    auto const sqrt_result =
      cudf::unary_operation(raw_result->view(), unary_operator::SQRT, stream, mr);
    return cudf::copy_if_else(sqrt_result->view(), null_scalar, valid_mask->view(), stream, mr);
  }
  return cudf::copy_if_else(raw_result->view(), null_scalar, valid_mask->view(), stream, mr);
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

  impl(host_span<size_type const> key_indices,
       host_span<streaming_aggregation_request const> requests,
       null_policy null_handling,
       sorted keys_are_sorted,
       std::vector<order> const& column_order,
       std::vector<null_order> const& null_precedence)
    : _key_indices{},
      _null_handling{null_handling},
      _keys_are_sorted{keys_are_sorted},
      _column_order{column_order},
      _null_precedence{null_precedence}
  {
    if (!key_indices.empty()) { _key_indices.assign(key_indices.begin(), key_indices.end()); }
    auto [plans, agg_columns] = build_plans_and_columns(requests);
    _plans                    = std::move(plans);
    _agg_columns              = std::move(agg_columns);
    for (auto const& request : requests) {
      _aggs_per_request.push_back(static_cast<size_type>(request.aggregations.size()));
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
      auto const& agg_column = _agg_columns[i];
      auto const batch_type  = batch_partial_cols[i]->type();
      auto const state_type =
        compute_partial_state_type(batch_type, agg_column.batch_kind, agg_column.merge_kind);
      _partial_column_types.push_back(state_type);
    }

    for (auto const& plan : _plans) {
      auto const& state_col = batch_partial_cols[plan.agg_column_refs[0]];
      if (!cudf::is_fixed_width(state_col->type())) {
        _finalize_output_types.push_back(state_col->type());
      } else {
        _finalize_output_types.push_back(cudf::detail::target_type(
          data.column(plan.value_column_index).type(), plan.requested_kind));
      }
    }
  }

  /**
   * @brief Cast batch result columns to the established partial-state types where they differ.
   */
  std::vector<std::unique_ptr<column>> cast_to_state_types(
    std::vector<std::unique_ptr<column>>& columns,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr)
  {
    std::vector<std::unique_ptr<column>> result;
    for (size_t i = 0; i < columns.size(); ++i) {
      if (columns[i]->type() != _partial_column_types[i] &&
          cudf::is_fixed_width(columns[i]->type()) &&
          cudf::is_fixed_width(_partial_column_types[i])) {
        result.push_back(cudf::cast(columns[i]->view(), _partial_column_types[i], stream, mr));
      } else {
        result.push_back(std::move(columns[i]));
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
      size_type count_result_index;
      size_type mean_result_index;
      size_type m2_result_index;
    };
    std::vector<m2_assembly_info> m2_assemblies;

    std::vector<aggregation_request> batch_requests;

    for (size_type agg_column_index = 0; agg_column_index < num_agg_columns(); ++agg_column_index) {
      auto const& agg_column = _agg_columns[agg_column_index];

      if (agg_column.batch_kind == aggregation::M2) {
        {
          aggregation_request request;
          request.values = data.column(agg_column.value_column_index);
          request.aggregations.push_back(
            make_count_aggregation<groupby_aggregation>(null_policy::EXCLUDE));
          batch_requests.push_back(std::move(request));
        }
        auto const count_result_index = static_cast<size_type>(batch_requests.size() - 1);
        {
          aggregation_request request;
          request.values = data.column(agg_column.value_column_index);
          request.aggregations.push_back(make_mean_aggregation<groupby_aggregation>());
          batch_requests.push_back(std::move(request));
        }
        auto const mean_result_index = static_cast<size_type>(batch_requests.size() - 1);
        {
          aggregation_request request;
          request.values = data.column(agg_column.value_column_index);
          request.aggregations.push_back(make_m2_aggregation<groupby_aggregation>());
          batch_requests.push_back(std::move(request));
        }
        auto const m2_result_index = static_cast<size_type>(batch_requests.size() - 1);

        m2_assemblies.push_back({count_result_index, mean_result_index, m2_result_index});
      } else {
        aggregation_request request;
        request.values = data.column(agg_column.value_column_index);
        request.aggregations.push_back(
          make_batch_agg(agg_column.batch_kind, *agg_column.original_agg));
        batch_requests.push_back(std::move(request));
      }
    }

    auto [batch_keys, batch_results] = batch_groupby.aggregate(batch_requests, stream, mr);
    out_keys                         = std::move(batch_keys);

    std::vector<std::unique_ptr<column>> partial_cols;
    size_type result_cursor      = 0;
    size_type m2_assembly_cursor = 0;
    for (size_type agg_column_index = 0; agg_column_index < num_agg_columns(); ++agg_column_index) {
      if (_agg_columns[agg_column_index].batch_kind == aggregation::M2) {
        auto const& info    = m2_assemblies[m2_assembly_cursor];
        auto count_column   = std::move(batch_results[info.count_result_index].results[0]);
        auto mean_column    = std::move(batch_results[info.mean_result_index].results[0]);
        auto m2_column      = std::move(batch_results[info.m2_result_index].results[0]);
        auto const num_rows = count_column->size();

        auto count_as_float64 =
          cudf::cast(count_column->view(), data_type{type_id::FLOAT64}, stream, mr);

        std::vector<std::unique_ptr<column>> children;
        children.push_back(std::move(count_as_float64));
        children.push_back(std::move(mean_column));
        children.push_back(std::move(m2_column));
        partial_cols.push_back(
          make_structs_column(num_rows, std::move(children), 0, rmm::device_buffer{}, stream, mr));
        result_cursor += 3;
        ++m2_assembly_cursor;
      } else {
        partial_cols.push_back(std::move(batch_results[result_cursor].results[0]));
        ++result_cursor;
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

    if (batch_keys->num_rows() == 0 && has_state()) { return; }

    if (!types_initialized()) { initialize_types(batch_partials, data); }

    auto cast_partials = cast_to_state_types(batch_partials, stream, mr);

    auto batch_cols = batch_keys->release();
    for (auto& column_ptr : cast_partials) {
      batch_cols.push_back(std::move(column_ptr));
    }

    if (!has_state()) {
      _partial_columns = std::move(batch_cols);
      return;
    }

    auto const batch_view = columns_as_table_view(batch_cols);
    merge_partial_states(
      _partial_columns, batch_view, num_keys(), _agg_columns, _null_handling, stream, mr);
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
      auto const other_view = columns_as_table_view(other._partial_columns);
      auto copied           = std::make_unique<table>(other_view, stream, mr);
      _partial_columns      = copied->release();
      if (!types_initialized() && other.types_initialized()) {
        _partial_column_types  = other._partial_column_types;
        _finalize_output_types = other._finalize_output_types;
      }
      return;
    }

    auto const other_view = columns_as_table_view(other._partial_columns);
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

    auto request_index = size_type{0};
    std::vector<aggregation_result> results;

    for (auto num_aggregations : _aggs_per_request) {
      aggregation_result agg_result;
      for (size_type aggregation_index = 0; aggregation_index < num_aggregations;
           ++aggregation_index) {
        auto const& plan = _plans[request_index];

        switch (plan.finalization) {
          case finalize_kind::IDENTITY: {
            auto const state_col_idx  = _agg_columns[plan.agg_column_refs[0]].column_index;
            auto const& source_column = *_partial_columns[num_keys() + state_col_idx];
            if (request_index < static_cast<size_type>(_finalize_output_types.size()) &&
                source_column.type() != _finalize_output_types[request_index] &&
                cudf::is_fixed_width(source_column.type()) &&
                cudf::is_fixed_width(_finalize_output_types[request_index])) {
              agg_result.results.push_back(cudf::cast(
                source_column.view(), _finalize_output_types[request_index], stream, mr));
            } else {
              agg_result.results.push_back(std::make_unique<column>(source_column, stream, mr));
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
          case finalize_kind::VARIANCE_FROM_M2:
          case finalize_kind::STD_FROM_M2: {
            auto const m2_col_idx = _agg_columns[plan.agg_column_refs[0]].column_index;
            auto const m2_struct  = _partial_columns[num_keys() + m2_col_idx]->view();

            auto ddof = size_type{1};
            if (auto const* var_agg =
                  dynamic_cast<cudf::detail::var_aggregation const*>(plan.requested_agg.get())) {
              ddof = var_agg->_ddof;
            } else if (auto const* std_agg = dynamic_cast<cudf::detail::std_aggregation const*>(
                         plan.requested_agg.get())) {
              ddof = std_agg->_ddof;
            }

            agg_result.results.push_back(finalize_variance_or_std(
              m2_struct, ddof, plan.finalization == finalize_kind::STD_FROM_M2, stream, mr));
            break;
          }
        }
        ++request_index;
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
    auto columns = std::move(_partial_columns);
    _partial_column_types.clear();
    return columns;
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
