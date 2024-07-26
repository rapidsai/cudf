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

#include "decimal_conversion_utilities.cuh"

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/linked_column.hpp>
#include <cudf/fixed_point/fixed_point.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>

#include <type_traits>

namespace cudf {
namespace detail {

template <typename DecimalType>
std::unique_ptr<rmm::device_buffer> convert_decimals_to_decimal128(
  cudf::column_view const& column, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  static_assert(std::is_same_v<DecimalType, int32_t> or std::is_same_v<DecimalType, int64_t>,
                "Only int32 and int64 decimal types can be converted to decimal128.");

  constexpr size_type BIT_WIDTH_RATIO = sizeof(__int128_t) / sizeof(DecimalType);
  auto buf = std::make_unique<rmm::device_buffer>(column.size() * sizeof(__int128_t), stream, mr);

  thrust::for_each(rmm::exec_policy_nosync(stream, mr),
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(column.size()),
                   [in  = column.begin<DecimalType>(),
                    out = reinterpret_cast<DecimalType*>(buf->data()),
                    BIT_WIDTH_RATIO] __device__(auto in_idx) {
                     auto const out_idx = in_idx * BIT_WIDTH_RATIO;
                     // the lowest order bits are the value, the remainder
                     // simply matches the sign bit to satisfy the two's
                     // complement integer representation of negative numbers.
                     out[out_idx] = in[in_idx];
#pragma unroll BIT_WIDTH_RATIO - 1
                     for (auto i = 1; i < BIT_WIDTH_RATIO; ++i) {
                       out[out_idx + i] = in[in_idx] < 0 ? -1 : 0;
                     }
                   });

  return buf;
}

// Instantiate templates for int32_t and int64_t decimal types
template std::unique_ptr<rmm::device_buffer> convert_decimals_to_decimal128<int32_t>(
  cudf::column_view const& column, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);

template std::unique_ptr<rmm::device_buffer> convert_decimals_to_decimal128<int64_t>(
  cudf::column_view const& column, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace cudf
