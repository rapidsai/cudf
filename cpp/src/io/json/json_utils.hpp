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

#pragma once

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/io/detail/tokenize_json.hpp>
#include <cudf/io/types.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <thrust/sequence.h>

namespace cudf::io::json::detail {
/**
 * @brief Returns stable sorted keys and its sorted order
 *
 * Uses cub stable radix sort. The order is internally generated, hence it saves a copy and memory.
 * Since the key and order is returned, using double buffer helps to avoid extra copy to user
 * provided output iterator.
 *
 * @tparam IndexType sorted order type
 * @tparam KeyType key type
 * @param keys keys to sort
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return Sorted keys and indices producing that sorted order
 */
template <typename IndexType = size_t, typename KeyType>
std::pair<rmm::device_uvector<KeyType>, rmm::device_uvector<IndexType>> stable_sorted_key_order(
  cudf::device_span<KeyType const> keys, rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  // Determine temporary device storage requirements
  rmm::device_uvector<KeyType> keys_buffer1(keys.size(), stream);
  rmm::device_uvector<KeyType> keys_buffer2(keys.size(), stream);
  rmm::device_uvector<IndexType> order_buffer1(keys.size(), stream);
  rmm::device_uvector<IndexType> order_buffer2(keys.size(), stream);
  cub::DoubleBuffer<IndexType> order_buffer(order_buffer1.data(), order_buffer2.data());
  cub::DoubleBuffer<KeyType> keys_buffer(keys_buffer1.data(), keys_buffer2.data());
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(
    nullptr, temp_storage_bytes, keys_buffer, order_buffer, keys.size());
  rmm::device_buffer d_temp_storage(temp_storage_bytes, stream);

  thrust::copy(rmm::exec_policy_nosync(stream), keys.begin(), keys.end(), keys_buffer1.begin());
  thrust::sequence(rmm::exec_policy_nosync(stream), order_buffer1.begin(), order_buffer1.end());

  cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                  temp_storage_bytes,
                                  keys_buffer,
                                  order_buffer,
                                  keys.size(),
                                  0,
                                  sizeof(KeyType) * 8,
                                  stream.value());

  return std::pair{keys_buffer.Current() == keys_buffer1.data() ? std::move(keys_buffer1)
                                                                : std::move(keys_buffer2),
                   order_buffer.Current() == order_buffer1.data() ? std::move(order_buffer1)
                                                                  : std::move(order_buffer2)};
}

template <typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2>
void max_row_offsets_col_categories(InputIterator1 keys_first,
                                    InputIterator1 keys_last,
                                    InputIterator2 values_first,
                                    OutputIterator1 keys_output,
                                    OutputIterator2 values_output,
                                    rmm::cuda_stream_view stream)
{
  using row_offset_t = size_type;
  thrust::reduce_by_key(rmm::exec_policy_nosync(stream),
                        keys_first,
                        keys_last,
                        values_first,
                        keys_output,
                        values_output,
                        thrust::equal_to<NodeIndexT>(),
                        [] __device__(auto a, auto b) {
                          auto row_offset_a = thrust::get<0>(a);
                          auto row_offset_b = thrust::get<0>(b);
                          auto type_a       = thrust::get<1>(a);
                          auto type_b       = thrust::get<1>(b);

                          NodeT ctg;
                          auto is_a_leaf = (type_a == NC_VAL || type_a == NC_STR);
                          auto is_b_leaf = (type_b == NC_VAL || type_b == NC_STR);
                          // (v+v=v, *+*=*,  *+v=*, *+#=E, NESTED+VAL=NESTED)
                          // *+*=*, v+v=v
                          if (type_a == type_b) {
                            ctg = type_a;
                          } else if (is_a_leaf) {
                            // *+v=*, N+V=N
                            // STRUCT/LIST + STR/VAL = STRUCT/LIST, STR/VAL + FN = ERR, STR/VAL +
                            // STR = STR
                            ctg = (type_b == NC_FN ? NC_ERR : (is_b_leaf ? NC_STR : type_b));
                          } else if (is_b_leaf) {
                            ctg = (type_a == NC_FN ? NC_ERR : (is_a_leaf ? NC_STR : type_a));
                          } else {
                            ctg = NC_ERR;
                          }

                          return thrust::make_pair(
                            thrust::maximum<row_offset_t>{}(row_offset_a, row_offset_b), ctg);
                        });
}


}  // namespace cudf::io::json::detail
