/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include "join_common_utils.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/join.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/gather.h>
#include <thrust/iterator/tabulate_output_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace cudf {

namespace {
struct row_comparator {
  row_comparator(table_device_view const lhs,
                                table_device_view const rhs)
    : _lhs{lhs}, _rhs{rhs} {}

  __device__ bool operator()(size_type lhs_index,
                             size_type rhs_index) const noexcept
  {
    table_device_view const* ptr_left_dview = &_lhs;
    table_device_view const* ptr_right_dview = &_rhs;
    cudf::experimental::row::lexicographic::device_row_comparator<false, bool> comparator(
          true, *ptr_left_dview, *ptr_right_dview);

    return comparator(lhs_index, rhs_index) == weak_ordering::LESS;
  }

 private:
  table_device_view const _lhs;
  table_device_view const _rhs;
};
} // anonymous namespace

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
sort_merge_inner_join(table_view const& left,
           table_view const& right,
           rmm::cuda_stream_view stream,
           rmm::device_async_resource_ref mr)
{
 CUDF_FUNC_RANGE();

  // Sanity checks
  CUDF_EXPECTS(left.num_columns() == right.num_columns(), "Number of columns must match for a join");
  CUDF_EXPECTS(!cudf::has_nested_columns(left) && !cudf::has_nested_columns(right), "Don't have sorting logic for nested columns yet");

  std::vector<cudf::order> column_order(left.num_columns(), cudf::order::ASCENDING);
  std::vector<cudf::null_order> null_precedence(left.num_columns(), cudf::null_order::BEFORE);
  auto sorted_left = cudf::sort(left, column_order, null_precedence, stream, mr); 
  auto sorted_right = cudf::sort(right, column_order, null_precedence, stream, mr); 

  auto &smaller = left.num_rows() < right.num_rows() ? *sorted_left : *sorted_right;
  auto &larger = left.num_rows() >= right.num_rows() ? *sorted_left : *sorted_right;

  auto smaller_dv_ptr = cudf::table_device_view::create(smaller, stream);
  auto larger_dv_ptr = cudf::table_device_view::create(larger, stream);

  // naive: iterate through larger table and binary search on smaller table
  auto const larger_numrows = larger.num_rows();
  auto const smaller_numrows = smaller.num_rows();
  row_comparator comparator(*larger_dv_ptr, *smaller_dv_ptr);
  auto larger_it = cudf::experimental::row::lhs_iterator(0);
  auto smaller_it = cudf::experimental::row::rhs_iterator(0);

  rmm::device_uvector<size_type> match_counts(larger_numrows, stream, mr);
  thrust::upper_bound(rmm::exec_policy(stream), smaller_it, smaller_it + smaller_numrows, larger_it, larger_it + larger_numrows, match_counts.begin(), comparator);
  auto match_counts_update_it =
    thrust::make_transform_output_iterator(match_counts.begin(), [] __device__(auto i) {
      return -i;
    });
  thrust::lower_bound(rmm::exec_policy(stream), smaller_it, smaller_it + smaller_numrows, larger_it, larger_it + larger_numrows, match_counts_update_it, comparator);

  auto count_matches_it = 
    thrust::make_transform_iterator(match_counts.begin(), [] __device__(auto c) {
      return c ? 1 : 0;
    });
  auto total_matches_it = match_counts.begin();
  auto zip_matches_it = thrust::make_zip_iterator(count_matches_it, total_matches_it);
  auto const matches_tuple = thrust::reduce(
    rmm::exec_policy(stream), 
    zip_matches_it,
    zip_matches_it + larger_numrows);
  auto const count_matches = thrust::get<0>(matches_tuple);
  auto const total_matches = thrust::get<1>(matches_tuple);

  auto left_indices = cudf::detail::make_zeroed_device_uvector_async<size_type>(total_matches, stream, mr);
  rmm::device_uvector<size_type> right_indices(total_matches, stream, mr);
  rmm::device_uvector<size_type> nonzero_matches(count_matches, stream, mr);
  thrust::gather_if(
    rmm::exec_policy(stream), 
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + count_matches,
    match_counts.begin(),
    thrust::make_counting_iterator(0),
    nonzero_matches.begin());

  // populate left indices
  thrust::exclusive_scan(rmm::exec_policy(stream), match_counts.begin(), match_counts.end(), match_counts.begin());
  thrust::scatter(
    rmm::exec_policy(stream), 
    nonzero_matches.begin(),
    nonzero_matches.end(),
    thrust::make_permutation_iterator(match_counts.begin(), nonzero_matches.begin()),
    left_indices.begin());
  thrust::inclusive_scan(rmm::exec_policy(stream), left_indices.begin(), left_indices.end(), left_indices.begin(), thrust::maximum<size_type>{});

  //populate right indices
  thrust::fill(rmm::exec_policy(stream), right_indices.begin(), right_indices.end(), 1);
  auto right_tabulate_it = thrust::make_tabulate_output_iterator(
    [nonzero_matches = nonzero_matches.begin(), match_counts = match_counts.begin(), right_indices = right_indices.begin()] __device__(auto idx, auto lb) {
      right_indices[match_counts[nonzero_matches[idx]]] = lb;
    });
  thrust::lower_bound(rmm::exec_policy(stream), smaller_it, smaller_it + smaller_numrows, nonzero_matches.begin(), nonzero_matches.end(), right_tabulate_it, comparator);
  thrust::inclusive_scan_by_key(rmm::exec_policy(stream), left_indices.begin(), left_indices.end(), right_indices.begin(), right_indices.begin());

  return {std::make_unique<rmm::device_uvector<size_type>>(std::move(left_indices)), std::make_unique<rmm::device_uvector<size_type>>(std::move(right_indices))};
}

} //namespace cudf
