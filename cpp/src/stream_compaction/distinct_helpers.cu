/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include "distinct_helpers.hpp"

#include <cuda/functional>
#include <cuda/std/atomic>

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
    output_indices.resize(thrust::distance(output_indices.begin(), output_end), stream);
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
      return thrust::copy_if(
        rmm::exec_policy(stream),
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(num_rows),
        output_indices.begin(),
        cuda::proclaim_return_type<bool>(
          [reduction_results = reduction_results.begin()] __device__(auto const idx) {
            return reduction_results[idx] == size_type{1};
          }));
    }

    // Reduction results with `KEEP_FIRST` and `KEEP_LAST` are row indices of the first/last row in
    // each group of equal rows (which are the desired output indices), or the value given by
    // `reduction_init_value()`.
    return thrust::copy_if(
      rmm::exec_policy(stream),
      reduction_results.begin(),
      reduction_results.end(),
      output_indices.begin(),
      cuda::proclaim_return_type<bool>([init_value = reduction_init_value(keep)] __device__(
                                         auto const idx) { return idx != init_value; }));
  }();

  output_indices.resize(thrust::distance(output_indices.begin(), map_end), stream);
  return output_indices;
}

template rmm::device_uvector<size_type> reduce_by_row(
  distinct_set_t<cudf::experimental::row::equality::device_row_comparator<
    false,
    cudf::nullate::DYNAMIC,
    cudf::experimental::row::equality::nan_equal_physical_equality_comparator>>& set,
  size_type num_rows,
  duplicate_keep_option keep,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

template rmm::device_uvector<size_type> reduce_by_row(
  distinct_set_t<cudf::experimental::row::equality::device_row_comparator<
    true,
    cudf::nullate::DYNAMIC,
    cudf::experimental::row::equality::nan_equal_physical_equality_comparator>>& set,
  size_type num_rows,
  duplicate_keep_option keep,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

template rmm::device_uvector<size_type> reduce_by_row(
  distinct_set_t<cudf::experimental::row::equality::device_row_comparator<
    false,
    cudf::nullate::DYNAMIC,
    cudf::experimental::row::equality::physical_equality_comparator>>& set,
  size_type num_rows,
  duplicate_keep_option keep,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

template rmm::device_uvector<size_type> reduce_by_row(
  distinct_set_t<cudf::experimental::row::equality::device_row_comparator<
    true,
    cudf::nullate::DYNAMIC,
    cudf::experimental::row::equality::physical_equality_comparator>>& set,
  size_type num_rows,
  duplicate_keep_option keep,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace cudf::detail
