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

#ifndef COPYING_HPP
#define COPYING_HPP

#include "cudf.h"
#include "types.hpp"
#include <cudf/legacy/copying.hpp>
#include <vector>

// Forward declaration
typedef struct CUstream_st* cudaStream_t;

namespace cudf {
namespace exp {

/*
 * Initializes and returns column of the same type as the input.
 * 
 * @param input The input column to emulate
 * @return column An unallocated column of same type as input
 */
std::unique_ptr<column> empty_like(column_view input);

#if 0
/**
 * @brief Allocates a new column of the same size and type as the input.
 *
 * @param input The input column to emulate
 * @param state Controls allocation/initialization of the mask
 * @param stream Optional The stream on which to execute all allocations and copies
 * @param mr Optional The resource to use for all allocations
 * @return column An allocated column of same size and type of input
 */
column allocate_like(column_view input, mask_state state, cudaStream_t stream,
                     rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

/**
 * @brief Allocates a new column of the specified size and same type as the input.
 *
 * @param input The input column to emulate
 * @param size The size of the column to allocate in rows
 * @param state Controls allocation/initialization of the mask
 * @param stream Optional The stream on which to execute all allocations and copies
 * @param mr Optional The resource to use for all allocations
 * @return column An allocated column of same size and type of input
 */
column allocate_like(column_view input, gdf_size_type size,
                     mask_state state, cudaStream_t stream = 0,
                     rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

/**
 * @brief Creates a table of empty columns with the same types as the inputs
 *
 * Creates the `cudf::column` objects, but does not allocate any underlying device
 * memory for the column's data or bitmask.
 *
 * @param t The input table to emulate
 * @return table A table of empty columns of same type as input
 */
table empty_like(table_view t);

/**
 * @brief Creates a table of columns with the same type and allocation size as
 * the input.
 *
 * Creates the `cudf::column` objects, and allocates underlying device memory for
 * each column matching the input columns
 *
 * @param t The table to emulate
 * @param state Controls allocation/initialization of the mask
 * @param stream Optional stream in which to perform allocations
 * @param mr Optional The resource to use for all allocations
 * @return table A table of columns with same type and allocation size as input
 */
table allocate_like(table_view t,
                    mask_state state,
                    cudaStream_t stream = 0,
                    rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

/**
 * @brief Creates a table of columns with the specified size and same type as
 * the input.
 *
 * Creates the `cudf::column` objects, and allocates underlying device memory for
 * each column matching the input columns
 *
 * @param t The table to emulate
 * @param size The size of the columns to allocate
 * @param state Controls allocation/initialization of the mask
 * @param stream Optional stream in which to perform allocations
 * @param mr Optional The resource to use for all allocations
 * @return table A table of columns with same type as @p t and specified @p size
 */
table allocate_like(table_view t,
                    gdf_size_type size,
                    mask_state state,
                    cudaStream_t stream = 0,
                    rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource());

#endif
#if 0
/**
 * @brief Creates a table of columns and deep copies the data from an input
 * table.
 *
 * @param t The table to copy
 * @return table A table that is an exact copy of @p t
 */
table copy(table_view t); 
#endif



}  // namespace ext
}  // namespace cudf

#endif  // COPYING_H
