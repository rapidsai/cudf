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

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/linked_column.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>

#include <type_traits>

namespace cudf::detail {

namespace {

/**
 * @copydoc cudf::detail::convert_decimal_data_to_decimal128
 */
template <typename DecimalType>
rmm::device_uvector<__int128_t> gpu_convert_decimal_data_to_decimal128(
  cudf::column_view const& column, rmm::cuda_stream_view stream)
{
  cudf::size_type constexpr BIT_WIDTH_RATIO = sizeof(__int128_t) / sizeof(DecimalType);

  rmm::device_uvector<__int128_t> d128_buffer(column.size(), stream);

  thrust::for_each(rmm::exec_policy_nosync(stream),
                   thrust::make_counting_iterator(0),
                   thrust::make_counting_iterator(column.size()),
                   [in  = column.begin<DecimalType>(),
                    out = reinterpret_cast<DecimalType*>(d128_buffer.data()),
                    BIT_WIDTH_RATIO] __device__(auto in_idx) {
                     auto const out_idx = in_idx * BIT_WIDTH_RATIO;
                     // The lowest order bits are the value, the remainder
                     // simply matches the sign bit to satisfy the two's
                     // complement integer representation of negative numbers.
                     out[out_idx] = in[in_idx];
#pragma unroll BIT_WIDTH_RATIO - 1
                     for (auto i = 1; i < BIT_WIDTH_RATIO; ++i) {
                       out[out_idx + i] = in[in_idx] < 0 ? -1 : 0;
                     }
                   });

  return d128_buffer;
}

}  // namespace

/**
 * @brief Convert decimal32 and decimal64 numeric data to decimal128 and return the device vector
 *
 * @tparam DecimalType to convert from
 *
 * @param column A view of the input columns
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return A device vector containing the converted decimal128 data
 */
template <typename DecimalType,
          std::enable_if_t<std::is_same_v<DecimalType, int32_t> or
                           std::is_same_v<DecimalType, int64_t>>* = nullptr>
rmm::device_uvector<__int128_t> convert_decimal_data_to_decimal128(cudf::column_view const& column,
                                                                   rmm::cuda_stream_view stream)
{
  return gpu_convert_decimal_data_to_decimal128<DecimalType>(column, stream);
}

}  // namespace cudf::detail
