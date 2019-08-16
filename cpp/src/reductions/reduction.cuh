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

#ifndef CUDF_REDUCTION_CUH
#define CUDF_REDUCTION_CUH

#include <cudf/cudf.h>
#include <cudf/reduction.hpp>
#include "reduction_operators.cuh"

#include <rmm/rmm.h>
#include <utilities/cudf_utils.h>
#include <utilities/error_utils.hpp>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <utilities/column_utils.hpp>

#include <cub/device/device_reduce.cuh>

namespace cudf {
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
void reduce(OutputType* dev_result, InputIterator d_in, gdf_size_type num_items,
    OutputType init, Op op, cudaStream_t stream)
{
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;

    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, dev_result, num_items,
        op, init, stream);
    // Allocate temporary storage
    RMM_TRY(RMM_ALLOC(&d_temp_storage, temp_storage_bytes, stream));

    // Run reduction
    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, dev_result, num_items,
        op, init, stream);

    // Free temporary storage
    RMM_TRY(RMM_FREE(d_temp_storage, stream));
}

} // namespace detail
} // namespace reduction
} // namespace cudf
#endif

