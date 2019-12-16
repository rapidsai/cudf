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

#include <cudf/dlpack.hpp>

namespace cudf {
namespace detail {

/**
 * @brief Convert a DLPack DLTensor into a cudf table
 *
 * @note The managed tensor is not deleted by this function.
 *
 * The `device_type` of the DLTensor must be `kDLGPU`, `kDLCPU`, or
 * `kDLCPUPinned`, and `device_id` must match the current device. The `ndim`
 * must be set to 1 or 2. The `dtype` must have 1 lane and the bitsize must
 * match a supported `cudf::data_type`.
 *
 * @throw cudf::logic_error if the any of the DLTensor fields are unsupported
 *
 * @param managed_tensor a 1D or 2D column-major (Fortran order) tensor
 * @param mr Optional resource to use for device memory allocation
 * @param stream Optional stream on which to execute
 *
 * @return Table with a copy of the tensor data
 */
std::unique_ptr<experimental::table> from_dlpack(
    DLManagedTensor const* managed_tensor,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);

/**
 * @brief Convert a cudf table into a DLPack DLTensor
 * 
 * All columns must have the same data type and this type must be numeric. The
 * columns may be nullable, but the null count must be zero. If the input table
 * is empty or has zero rows, the result will be nullptr.
 * 
 * @note The `deleter` method of the returned `DLManagedTensor` must be used to
 * free the memory allocated for the tensor.
 * 
 * @throw cudf::logic_error if the data types are not equal or not numeric,
 * or if any of columns have non-zero null count
 * 
 * @param input Table to convert to DLPack
 * @param mr Optional resource to use for device memory allocation
 * @param stream Optional stream on which to execute
 * 
 * @return 1D or 2D DLPack tensor with a copy of the table data, or nullptr
 */
DLManagedTensor* to_dlpack(table_view const& input,
    rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
    cudaStream_t stream = 0);

}  // namespace detail
}  // namespace cudf
