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

template <typename RowHasher>
rmm::device_uvector<size_type> reduce_by_row(hash_map_type<RowHasher>& map,
                                             size_type num_rows,
                                             duplicate_keep_option keep,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  if ((keep == duplicate_keep_option::KEEP_FIRST) or (keep == duplicate_keep_option::KEEP_LAST)) {
    auto output_indices = rmm::device_uvector<size_type>(num_rows, stream, mr);

    auto pairs =
      thrust::make_transform_iterator(thrust::counting_iterator<size_type>(0),
                                      cuda::proclaim_return_type<cuco::pair<size_type, size_type>>(
                                        [] __device__(size_type const i) {
                                          return cuco::pair<size_type, size_type>{i, i};
                                        }));

    if (keep == duplicate_keep_option::KEEP_FIRST) {
      map.insert_or_apply_async(pairs, pairs + num_rows, min_op{}, stream.value());
    } else {
      map.insert_or_apply_async(pairs, pairs + num_rows, max_op{}, stream.value());
    }

    auto const [_, output_end] =
      map.retrieve_all(thrust::make_discard_iterator(), output_indices.begin(), stream.value());
    output_indices.resize(thrust::distance(output_indices.begin(), output_end), stream);
    return output_indices;
  }

  auto keys   = rmm::device_uvector<size_type>(num_rows, stream);
  auto values = rmm::device_uvector<size_type>(num_rows, stream);

  auto pairs = thrust::make_transform_iterator(
    thrust::counting_iterator<size_type>(0),
    cuda::proclaim_return_type<cuco::pair<size_type, size_type>>([] __device__(size_type const i) {
      return cuco::pair<size_type, size_type>{i, 1};
    }));

  map.insert_or_apply_async(pairs, pairs + num_rows, plus_op{}, stream.value());
  auto const [keys_end, _] = map.retrieve_all(keys.begin(), values.begin(), stream.value());

  auto num_distinct_keys = thrust::distance(keys.begin(), keys_end);
  keys.resize(num_distinct_keys, stream);
  values.resize(num_distinct_keys, stream);

  auto output_indices = rmm::device_uvector<size_type>(num_distinct_keys, stream, mr);

  auto const output_iter = cudf::detail::make_counting_transform_iterator(
    size_type(0),
    cuda::proclaim_return_type<size_type>(
      [keys = keys.begin(), values = values.begin()] __device__(auto const idx) {
        return values[idx] == size_type{1} ? keys[idx] : -1;
      }));

  auto const map_end = thrust::copy_if(
    rmm::exec_policy_nosync(stream),
    output_iter,
    output_iter + num_distinct_keys,
    output_indices.begin(),
    cuda::proclaim_return_type<bool>([] __device__(auto const idx) { return idx != -1; }));

  output_indices.resize(thrust::distance(output_indices.begin(), map_end), stream);
  return output_indices;
}

template rmm::device_uvector<size_type> reduce_by_row(
  hash_map_type<cudf::experimental::row::equality::device_row_comparator<
    false,
    cudf::nullate::DYNAMIC,
    cudf::experimental::row::equality::nan_equal_physical_equality_comparator>>& map,
  size_type num_rows,
  duplicate_keep_option keep,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

template rmm::device_uvector<size_type> reduce_by_row(
  hash_map_type<cudf::experimental::row::equality::device_row_comparator<
    true,
    cudf::nullate::DYNAMIC,
    cudf::experimental::row::equality::nan_equal_physical_equality_comparator>>& map,
  size_type num_rows,
  duplicate_keep_option keep,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

template rmm::device_uvector<size_type> reduce_by_row(
  hash_map_type<cudf::experimental::row::equality::device_row_comparator<
    false,
    cudf::nullate::DYNAMIC,
    cudf::experimental::row::equality::physical_equality_comparator>>& map,
  size_type num_rows,
  duplicate_keep_option keep,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

template rmm::device_uvector<size_type> reduce_by_row(
  hash_map_type<cudf::experimental::row::equality::device_row_comparator<
    true,
    cudf::nullate::DYNAMIC,
    cudf::experimental::row::equality::physical_equality_comparator>>& map,
  size_type num_rows,
  duplicate_keep_option keep,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace cudf::detail
