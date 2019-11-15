/*
 * Copyright (c) 2018-2019, NVIDIA CORPORATION.
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
#include <cudf/types.hpp>
#include <cudf/copying.hpp>

namespace cudf {
namespace experimental {
namespace detail {
/*
 * Initializes and returns an empty column of the same type as the `input`.
 *
 * @param[in] input Immutable view of input column to emulate
 * @param[in] stream Optional CUDA stream on which to execute kernels
 * @return std::unique_ptr<column> An empty column of same type as `input`
 */
std::unique_ptr<column> empty_like(column_view input, cudaStream_t stream = 0);

/**
 * @brief Creates an uninitialized new column of the specified size and same type as the `input`.
 * Supports only fixed-width types.
 *
 * @param[in] input Immutable view of input column to emulate
 * @param[in] size The desired number of elements that the new column should have capacity for
 * @param[in] mask_alloc Optional, Policy for allocating null mask. Defaults to RETAIN.
 * @param[in] mr Optional, The resource to use for all allocations
 * @param[in] stream Optional CUDA stream on which to execute kernels
 * @return std::unique_ptr<column> A column with sufficient uninitialized capacity to hold the specified number of elements as `input` of the same type as `input.type()`
 */
std::unique_ptr<column> allocate_like(column_view input, size_type size,
                                      mask_allocation_policy mask_alloc = 
                                          mask_allocation_policy::RETAIN,
                                      rmm::mr::device_memory_resource *mr =
                                          rmm::mr::get_default_resource(),
                                      cudaStream_t stream = 0);

/**
 * @brief Creates a table of empty columns with the same types as the `input_table`
 *
 * Creates the `cudf::column` objects, but does not allocate any underlying device
 * memory for the column's data or bitmask.
 *
 * @param[in] input_table Immutable view of input table to emulate
 * @param[in] stream Optional CUDA stream on which to execute kernels
 * @return std::unique_ptr<table> A table of empty columns with the same types as the columns in `input_table`
 */
std::unique_ptr<table> empty_like(table_view input_table, cudaStream_t stream = 0);

/**
 * @brief   Returns a new column, where each element is selected from either @p lhs or 
 *          @p rhs based on the value of the corresponding element in @p boolean_mask
 *
 * Selects each element i in the output column from either @p rhs or @p lhs using the following rule:
 *          output[i] = (boolean_mask[i]) ? lhs[i] : rhs[i]
 *         
 * @throws cudf::logic_error if lhs and rhs are not of the same type
 * @throws cudf::logic_error if lhs and rhs are not of the same length
 * @throws cudf::logic_error if boolean mask is not of type bool8
 * @throws cudf::logic_error if boolean mask is not of the same length as lhs and rhs  
 * @param[in] left-hand column_view
 * @param[in] right-hand column_view
 * @param[in] column_view representing "left (true) / right (false)" boolean for each element
 * @param[in] mr resource for allocating device memory
 * @param[in] stream Optional CUDA stream on which to execute kernels
 *
 * @returns new column with the selected elements
 */
std::unique_ptr<column> copy_if_else( column_view const& lhs, column_view const& rhs, column_view const& boolean_mask,
                                    rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource(),
                                    cudaStream_t stream = 0);

}  // namespace detail
}  // namespace experimental
}  // namespace cudf
