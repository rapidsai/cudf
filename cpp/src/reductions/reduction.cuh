/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/cudf.h>
#include <rmm/device_buffer.hpp>

#include <cub/device/device_reduce.cuh>

namespace cudf {
namespace experimental {
namespace reduction {
namespace detail {

/** --------------------------------------------------------------------------*
 * @brief compute reduction by the operator
 *
 * @param[out] dev_result  output device memory
 * @param[in] d_in      the begin iterator
 * @param[in] num_items the number of items
 * @param[in] init      the initial value of reduction
 * @param[in] op        the device binary operator
 * @param[in] stream    cuda stream
 *
 * @tparam Op               the device binary operator
 * @tparam InputIterator    the input column iterator
 * @tparam OutputType       the output type of reduction
 * ----------------------------------------------------------------------------**/
 template <typename Op, typename InputIterator, typename OutputType>
void reduce(OutputType* dev_result, InputIterator d_in, cudf::size_type num_items,
    OutputType init, Op op, 
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream)
{
    rmm::device_buffer  d_temp_storage;
    size_t  temp_storage_bytes = 0;

    cub::DeviceReduce::Reduce(d_temp_storage.data(), temp_storage_bytes, d_in, dev_result, num_items,
        op, init, stream);
    // Allocate temporary storage
    d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream, mr};

    // Run reduction
    cub::DeviceReduce::Reduce(d_temp_storage.data(), temp_storage_bytes, d_in, dev_result, num_items,
        op, init, stream);
}

} // namespace detail
} // namespace reduction
} // namespace experimental
} // namespace cudf

