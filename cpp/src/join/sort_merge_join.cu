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
#include <thrust/tuple.h>

namespace cudf {

#define SORT_MERGE_JOIN_DEBUG 0

namespace {

#if SORT_MERGE_JOIN_DEBUG
template <typename T>
void debug_print(std::string str, host_span<const T> span) {
  std::cout << str << " : ";
  for (const auto& element : span) {
      std::cout << element << " ";
  }
  std::cout << std::endl;
}
#endif

enum class bound_type {
  UPPER,
  LOWER
};

struct row_comparator {
  row_comparator(table_device_view const lhs, table_device_view const rhs, column_device_view const rhs_order, bound_type *d_ptr) : 
    _lhs{lhs}, _rhs{rhs}, _rhs_order{rhs_order}, _d_ptr{d_ptr} {}
    

  __device__ bool operator()(size_type lhs_index, size_type rhs_index) const noexcept
  {
    if (*_d_ptr == bound_type::UPPER) {
      cudf::experimental::row::lexicographic::device_row_comparator<false, bool> comparator(
        true, _lhs, _rhs);
      return comparator(lhs_index, _rhs_order.data<size_type>()[rhs_index]) ==
             weak_ordering::LESS;
    }
    cudf::experimental::row::lexicographic::device_row_comparator<false, bool> comparator(
      true, _rhs, _lhs);
    return comparator(_rhs_order.data<size_type>()[lhs_index], rhs_index) ==
           weak_ordering::LESS;
  }

 private:
  table_device_view _lhs;
  table_device_view _rhs;
  column_device_view _rhs_order;
  bound_type *_d_ptr;
};

}  // anonymous namespace

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
sort_merge_inner_join(table_view const& left,
                      table_view const& right,
                      null_equality compare_nulls,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  // Sanity checks
  CUDF_EXPECTS(left.num_columns() == right.num_columns(),
               "Number of columns must match for a join");
  CUDF_EXPECTS(!cudf::has_nested_columns(left) && !cudf::has_nested_columns(right),
               "Don't have sorting logic for nested columns yet");

  std::vector<cudf::order> column_order(left.num_columns(), cudf::order::ASCENDING);
  std::vector<cudf::null_order> null_precedence(left.num_columns(), cudf::null_order::BEFORE);
  auto sorted_left_order_col = cudf::sorted_order(left, column_order, null_precedence, stream, mr);
  auto sorted_right_order_col = cudf::sorted_order(right, column_order, null_precedence, stream, mr);

  auto& smaller = left.num_rows() < right.num_rows() ? left : right;
  auto& sorted_smaller_order_col = left.num_rows() < right.num_rows() ? sorted_left_order_col : sorted_right_order_col;
  auto& larger = left.num_rows() >= right.num_rows() ? left : right;
  auto& sorted_larger_order_col = left.num_rows() >= right.num_rows() ? sorted_left_order_col : sorted_right_order_col;

#if SORT_MERGE_JOIN_DEBUG
  std::vector<size_type> host_data(sorted_larger_order_col->size());
  CUDF_CUDA_TRY(cudaMemcpyAsync(host_data.data(), sorted_larger_order_col->view().head<size_type>(), 
             sorted_larger_order_col->size() * sizeof(size_type), cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();
  debug_print<size_type>("h_sorted_larger_order_col", host_data);
  CUDF_CUDA_TRY(cudaMemcpyAsync(host_data.data(), sorted_smaller_order_col->view().head<size_type>(), 
             sorted_smaller_order_col->size() * sizeof(size_type), cudaMemcpyDeviceToHost, stream.value()));
  stream.synchronize();
  debug_print<size_type>("h_sorted_smaller_order_col", host_data);
#endif

  auto smaller_dv_ptr = cudf::table_device_view::create(smaller, stream);
  auto larger_dv_ptr  = cudf::table_device_view::create(larger, stream);
  auto sorted_smaller_order_dv_ptr = cudf::column_device_view::create(*sorted_smaller_order_col, stream);
  auto sorted_larger_order_dv_ptr = cudf::column_device_view::create(*sorted_larger_order_col, stream);

  // naive: iterate through larger table and binary search on smaller table
  auto const larger_numrows  = larger.num_rows();
  auto const smaller_numrows = smaller.num_rows();
  rmm::device_scalar<bound_type> d_lb_type(bound_type::LOWER, stream, mr);
  rmm::device_scalar<bound_type> d_ub_type(bound_type::UPPER, stream, mr);

  auto match_counts = cudf::detail::make_zeroed_device_uvector_async<size_type>(larger_numrows + 1, stream, mr);

#if SORT_MERGE_JOIN_DEBUG
  rmm::device_uvector<size_type> lb1(larger_numrows, stream, mr);
  row_comparator comp_lb_1(*larger_dv_ptr, *smaller_dv_ptr, *sorted_smaller_order_dv_ptr, d_lb_type.data());
  thrust::lower_bound(rmm::exec_policy(stream),
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(0) + smaller_numrows,
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(0) + larger_numrows,
                      lb1.begin(),
                      comp_lb_1);
  debug_print<size_type>("h_lb1", cudf::detail::make_host_vector_sync(lb1, stream));

  rmm::device_uvector<size_type> ub1(larger_numrows, stream, mr);
  row_comparator comp_ub_1(*larger_dv_ptr, *smaller_dv_ptr, *sorted_smaller_order_dv_ptr, d_ub_type.data());
  thrust::upper_bound(rmm::exec_policy(stream),
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(0) + smaller_numrows,
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(0) + larger_numrows,
                      ub1.begin(),
                      comp_ub_1);
  debug_print<size_type>("h_ub1", cudf::detail::make_host_vector_sync(ub1, stream));
#endif

  row_comparator comp_ub(*larger_dv_ptr, *smaller_dv_ptr, *sorted_smaller_order_dv_ptr, d_ub_type.data());
  auto match_counts_it = match_counts.begin();
  thrust::upper_bound(rmm::exec_policy(stream),
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(0) + smaller_numrows,
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(0) + larger_numrows,
                      match_counts_it,
                      comp_ub);

#if SORT_MERGE_JOIN_DEBUG
  debug_print<size_type>("h_match_counts", cudf::detail::make_host_vector_sync(match_counts, stream));
#endif

  row_comparator comp_lb(*larger_dv_ptr, *smaller_dv_ptr, *sorted_smaller_order_dv_ptr, d_lb_type.data());
  auto match_counts_update_it = thrust::make_tabulate_output_iterator(
    [match_counts = match_counts.begin()] __device__(size_type idx, size_type val) { 
      match_counts[idx] -= val;
    });

  thrust::lower_bound(rmm::exec_policy(stream),
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(0) + smaller_numrows,
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(0) + larger_numrows,
                      match_counts_update_it,
                      comp_lb);

#if SORT_MERGE_JOIN_DEBUG
  debug_print<size_type>("h_match_counts", cudf::detail::make_host_vector_sync(match_counts, stream));
#endif

  auto count_matches_it = thrust::make_transform_iterator(
    match_counts.begin(),
    cuda::proclaim_return_type<size_type>([] __device__(auto c) { return c ? 1 : 0; }));
  auto zip_matches_it_begin =
    thrust::make_zip_iterator(thrust::make_tuple(count_matches_it, match_counts.begin()));
  auto zip_matches_it_end = thrust::make_zip_iterator(
    thrust::make_tuple(count_matches_it + larger_numrows, match_counts.begin() + larger_numrows));
  auto const matches_tuple =
    thrust::reduce(rmm::exec_policy(stream),
                   zip_matches_it_begin,
                   zip_matches_it_end,
                   thrust::make_tuple(0, 0),
                   cuda::proclaim_return_type<thrust::tuple<size_type, size_type>>(
                     [] __device__(auto const& a, auto const& b) {
                       return thrust::make_tuple(thrust::get<0>(a) + thrust::get<0>(b),
                                                 thrust::get<1>(a) + thrust::get<1>(b));
                     }));
  auto const count_matches = thrust::get<0>(matches_tuple);
  auto const total_matches = thrust::get<1>(matches_tuple);

#if SORT_MERGE_JOIN_DEBUG
  debug_print<size_type>("h_match_counts", cudf::detail::make_host_vector_sync(match_counts, stream));
  std::printf("count_matches = %d, total_matches = %d\n", count_matches, total_matches);
#endif

  auto left_indices =
    cudf::detail::make_zeroed_device_uvector_async<size_type>(total_matches, stream, mr);
  rmm::device_uvector<size_type> right_indices(total_matches, stream, mr);
  rmm::device_uvector<size_type> nonzero_matches(count_matches, stream, mr);
  thrust::copy_if(rmm::exec_policy(stream),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(0) + larger_numrows,
    nonzero_matches.begin(),
    [match_counts = match_counts.begin()] __device__(auto idx) {
      return match_counts[idx];
    });

#if SORT_MERGE_JOIN_DEBUG
  debug_print<size_type>("h_nonzero_matches", cudf::detail::make_host_vector_sync(nonzero_matches, stream));
#endif

  // populate left indices
  thrust::exclusive_scan(
    rmm::exec_policy(stream), match_counts.begin(), match_counts.end(), match_counts.begin());
  thrust::scatter(rmm::exec_policy(stream),
                  nonzero_matches.begin(),
                  nonzero_matches.end(),
                  thrust::make_permutation_iterator(match_counts.begin(), nonzero_matches.begin()),
                  left_indices.begin());
  thrust::inclusive_scan(rmm::exec_policy(stream),
                         left_indices.begin(),
                         left_indices.end(),
                         left_indices.begin(),
                         thrust::maximum<size_type>{});

#if SORT_MERGE_JOIN_DEBUG
  debug_print<size_type>("h_match_counts", cudf::detail::make_host_vector_sync(match_counts, stream));
  rmm::device_uvector<size_type> dlb(nonzero_matches.size(), stream, mr);
  thrust::lower_bound(rmm::exec_policy(stream),
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(0) + smaller_numrows,
                      nonzero_matches.begin(),
                      nonzero_matches.end(),
                      dlb.begin(),
                      comp_lb);
  debug_print<size_type>("h_dlb", cudf::detail::make_host_vector_sync(dlb, stream));
#endif

  // populate right indices
  thrust::fill(rmm::exec_policy(stream), right_indices.begin(), right_indices.end(), -1);
  auto right_tabulate_it = thrust::make_tabulate_output_iterator(
    [nonzero_matches = nonzero_matches.begin(),
     match_counts    = match_counts.begin(),
     right_indices   = right_indices.begin(),
     sorted_larger_order = sorted_larger_order_col->view().begin<size_type>(),
     sorted_smaller_order = sorted_smaller_order_col->view().begin<size_type>()] __device__(auto idx, auto lb) {
      auto lhsidx = nonzero_matches[idx];
      auto i = match_counts[lhsidx];
      auto j = match_counts[lhsidx + 1];
      // iterate between i and j and update everything
      for(auto a = i; a < j; a++, lb++) {
        auto rhsidx = sorted_smaller_order[lb];
        right_indices[a] = rhsidx;
      }
    });
  thrust::lower_bound(rmm::exec_policy(stream),
                      thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(0) + smaller_numrows,
                      nonzero_matches.begin(),
                      nonzero_matches.end(),
                      right_tabulate_it,
                      comp_lb);

#if SORT_MERGE_JOIN_DEBUG
  debug_print<size_type>("h_left_indices", cudf::detail::make_host_vector_sync(left_indices, stream));
  debug_print<size_type>("h_right_indices", cudf::detail::make_host_vector_sync(right_indices, stream));
#endif

  return {std::make_unique<rmm::device_uvector<size_type>>(std::move(left_indices)),
          std::make_unique<rmm::device_uvector<size_type>>(std::move(right_indices))};
}

}  // namespace cudf
