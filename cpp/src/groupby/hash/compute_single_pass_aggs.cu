/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "compute_single_pass_aggs.hpp"
#include "flatten_single_pass_aggs.hpp"
#include "helpers.cuh"
#include "single_pass_functors.cuh"
#include "var_hash_functor.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/groupby.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <algorithm>
#include <memory>
#include <vector>

namespace cudf::groupby::detail::hash {
// make table that will hold sparse results
auto create_sparse_results_table(table_view const& flattened_values,
                                 std::vector<aggregation::Kind> aggs,
                                 rmm::cuda_stream_view stream)
{
  // TODO single allocation - room for performance improvement
  std::vector<std::unique_ptr<column>> sparse_columns;
  sparse_columns.reserve(flatten_values.num_columns());
  std::transform(
    flattened_values.begin(),
    flattened_values.end(),
    aggs.begin(),
    std::back_inserter(sparse_columns),
    [stream](auto const& col, auto const& agg) {
      bool nullable =
        (agg == aggregation::COUNT_VALID or agg == aggregation::COUNT_ALL)
          ? false
          : (col.has_nulls() or agg == aggregation::VARIANCE or agg == aggregation::STD);
      auto mask_flag = (nullable) ? mask_state::ALL_NULL : mask_state::UNALLOCATED;

      auto col_type = cudf::is_dictionary(col.type())
                        ? cudf::dictionary_column_view(col).keys().type()
                        : col.type();

      return make_fixed_width_column(
        cudf::detail::target_type(col_type, agg), col.size(), mask_flag, stream);
    });

  table sparse_table(std::move(sparse_columns));
  mutable_table_view table_view = sparse_table.mutable_view();
  cudf::detail::initialize_with_identity(table_view, aggs, stream);
  return sparse_table;
}

/**
 * @brief Computes all aggregations from `requests` that require a single pass
 * over the data and stores the results in `sparse_results`
 */
template <typename SetType>
void compute_single_pass_aggs(int64_t num_keys,
                              bool skip_rows_with_nulls,
                              bitmask_type const* row_bitmask,
                              SetType set,
                              host_span<aggregation_request const> requests,
                              cudf::detail::result_cache* sparse_results,
                              rmm::cuda_stream_view stream)
{
  // flatten the aggs to a table that can be operated on by aggregate_row
  auto const [flattened_values, agg_kinds, aggs] = flatten_single_pass_aggs(requests);

  // make table that will hold sparse results
  table sparse_table = create_sparse_results_table(flattened_values, agg_kinds, stream);
  // prepare to launch kernel to do the actual aggregation
  auto d_sparse_table = mutable_table_device_view::create(sparse_table, stream);
  auto d_values       = table_device_view::create(flattened_values, stream);
  auto const d_aggs   = cudf::detail::make_device_uvector_async(
    agg_kinds, stream, cudf::get_current_device_resource_ref());

  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator(0),
    num_keys,
    hash::compute_single_pass_aggs_fn{
      set, *d_values, *d_sparse_table, d_aggs.data(), row_bitmask, skip_rows_with_nulls});
  // Add results back to sparse_results cache
  auto sparse_result_cols = sparse_table.release();
  for (size_t i = 0; i < aggs.size(); i++) {
    // Note that the cache will make a copy of this temporary aggregation
    sparse_results->add_result(
      flattened_values.column(i), *aggs[i], std::move(sparse_result_cols[i]));
  }
}

template void compute_single_pass_aggs<hash_set_ref_t<cuco::insert_and_find_tag>>(
  int64_t num_keys,
  bool skip_rows_with_nulls,
  bitmask_type const* row_bitmask,
  hash_set_ref_t<cuco::insert_and_find_tag> set,
  host_span<aggregation_request const> requests,
  cudf::detail::result_cache* sparse_results,
  rmm::cuda_stream_view stream);

template void compute_single_pass_aggs<nullable_hash_set_ref_t<cuco::insert_and_find_tag>>(
  int64_t num_keys,
  bool skip_rows_with_nulls,
  bitmask_type const* row_bitmask,
  nullable_hash_set_ref_t<cuco::insert_and_find_tag> set,
  host_span<aggregation_request const> requests,
  cudf::detail::result_cache* sparse_results,
  rmm::cuda_stream_view stream);
}  // namespace cudf::groupby::detail::hash
