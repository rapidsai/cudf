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
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <type_traits>

namespace cudf::detail {

/**
 * @brief Convert decimal32 and decimal64 numeric data to decimal128 and return the device vector
 *
 * @tparam DecimalType to convert from
 *
 * @param column A view of the input columns
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource to use for device memory allocation
 *
 * @return A device vector containing the converted decimal128 data
 */
template <typename DecimalType>
std::unique_ptr<rmm::device_buffer> convert_decimals_to_decimal128(
  cudf::column_view const& input, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);

}  // namespace cudf::detail
