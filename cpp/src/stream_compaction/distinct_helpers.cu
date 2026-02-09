/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "distinct_helpers.hpp"

#include <cudf/detail/utilities/algorithm.cuh>

#include <cuda/functional>
#include <cuda/std/atomic>
#include <cuda/std/iterator>

namespace cudf::detail {

template <typename RowEqual>
rmm::device_uvector<size_type> reduce_by_row(distinct_set_t<RowEqual>& set,
                                             size_type num_rows,
                                             duplicate_keep_option keep,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  auto output_indices = rmm::device_uvector<size_type>(num_rows, stream, mr);

  // If we don't care about order, just gather indices of distinct keys taken from set.
  if (keep == duplicate_keep_option::KEEP_ANY) {
    auto const iter = thrust::counting_iterator<cudf::size_type>{0};
    set.insert_async(iter, iter + num_rows, stream.value());
    auto const output_end = set.retrieve_all(output_indices.begin(), stream.value());
    output_indices.resize(cuda::std::distance(output_indices.begin(), output_end), stream);
    return output_indices;
  }

  auto reduction_results = rmm::device_uvector<size_type>(num_rows, stream, mr);
  thrust::uninitialized_fill(rmm::exec_policy_nosync(stream),
                             reduction_results.begin(),
                             reduction_results.end(),
                             reduction_init_value(keep));

  auto set_ref = set.ref(cuco::op::insert_and_find);

  thrust::for_each(rmm::exec_policy_nosync(stream),
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(num_rows),
                   [set_ref, keep, reduction_results = reduction_results.begin()] __device__(
                     size_type const idx) mutable {
                     auto const [inserted_idx_ptr, _] = set_ref.insert_and_find(idx);

                     auto ref = cuda::atomic_ref<size_type, cuda::thread_scope_device>{
                       reduction_results[*inserted_idx_ptr]};
                     if (keep == duplicate_keep_option::KEEP_FIRST) {
                       // Store the smallest index of all rows that are equal.
                       ref.fetch_min(idx, cuda::memory_order_relaxed);
                     } else if (keep == duplicate_keep_option::KEEP_LAST) {
                       // Store the greatest index of all rows that are equal.
                       ref.fetch_max(idx, cuda::memory_order_relaxed);
                     } else {
                       // Count the number of rows in each group of rows that are compared equal.
                       ref.fetch_add(size_type{1}, cuda::memory_order_relaxed);
                     }
                   });

  auto const map_end = [&] {
    if (keep == duplicate_keep_option::KEEP_NONE) {
      // Reduction results with `KEEP_NONE` are either group sizes of equal rows, or `0`.
      // Thus, we only output index of the rows in the groups having group size of `1`.
      return cudf::detail::copy_if(
        thrust::counting_iterator<size_type>(0),
        thrust::counting_iterator<size_type>(num_rows),
        output_indices.begin(),
        cuda::proclaim_return_type<bool>(
          [reduction_results = reduction_results.begin()] __device__(auto const idx) {
            return reduction_results[idx] == size_type{1};
          }),
        stream);
    }

    // Reduction results with `KEEP_FIRST` and `KEEP_LAST` are row indices of the first/last row in
    // each group of equal rows (which are the desired output indices), or the value given by
    // `reduction_init_value()`.
    return cudf::detail::copy_if(
      reduction_results.begin(),
      reduction_results.end(),
      output_indices.begin(),
      cuda::proclaim_return_type<bool>([init_value = reduction_init_value(keep)] __device__(
                                         auto const idx) { return idx != init_value; }),
      stream);
  }();

  output_indices.resize(cuda::std::distance(output_indices.begin(), map_end), stream);
  return output_indices;
}

template rmm::device_uvector<size_type> reduce_by_row(
  distinct_set_t<cudf::detail::row::equality::device_row_comparator<
    false,
    cudf::nullate::DYNAMIC,
    cudf::detail::row::equality::nan_equal_physical_equality_comparator>>& set,
  size_type num_rows,
  duplicate_keep_option keep,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

template rmm::device_uvector<size_type> reduce_by_row(
  distinct_set_t<cudf::detail::row::equality::device_row_comparator<
    true,
    cudf::nullate::DYNAMIC,
    cudf::detail::row::equality::nan_equal_physical_equality_comparator>>& set,
  size_type num_rows,
  duplicate_keep_option keep,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

template rmm::device_uvector<size_type> reduce_by_row(
  distinct_set_t<cudf::detail::row::equality::device_row_comparator<
    false,
    cudf::nullate::DYNAMIC,
    cudf::detail::row::equality::physical_equality_comparator>>& set,
  size_type num_rows,
  duplicate_keep_option keep,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

template rmm::device_uvector<size_type> reduce_by_row(
  distinct_set_t<cudf::detail::row::equality::device_row_comparator<
    true,
    cudf::nullate::DYNAMIC,
    cudf::detail::row::equality::physical_equality_comparator>>& set,
  size_type num_rows,
  duplicate_keep_option keep,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace cudf::detail
