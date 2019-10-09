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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either experimentalress or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once 

#include "cudf.h"
#include "types.hpp"
#include <vector>

// Forward declaration
typedef struct CUstream_st* cudaStream_t;

namespace cudf {
namespace experimental {

/** ---------------------------------------------------------------------------*
* @brief Indicates when to allocate a mask, based on an existing mask.
* ---------------------------------------------------------------------------**/
enum mask_allocation_policy {
    NEVER, ///< Do not allocate a null mask, regardless of input
    RETAIN, ///< Allocate a null mask if the input contains one
    ALWAYS ///< Allocate a null mask, regardless of input
};

/*
 * Initializes and returns column of the same type as the `input`.
 *
 * @param input Immutable view of input column to emulate
 * @return std::unique_ptr<column> An unallocated column of same type as `input`
 */
std::unique_ptr<column> empty_like(column_view input);

/**
 * @brief Allocates a new column of the same size and type as the `input`.
 *
 * @param input Immutable view of input column to emulate
 * @param mask_alloc Optional, Policy for allocating null mask. Defaults to RETAIN.
 * @param mr Optional, The resource to use for all allocations
 * @return std::unique_ptr<column> An allocated column of same size and type of `input`
 */
std::unique_ptr<column> allocate_like(column_view input,
		                      mask_allocation_policy mask_alloc = RETAIN,
                                      rmm::mr::device_memory_resource *mr =
				          rmm::mr::get_default_resource());

/**
 * @brief Allocates a new column of the specified size and same type as the `input`.
 *
 * @param input Immutable view of input column to emulate
 * @param size The size of the column to allocate in rows
 * @param mask_alloc Optional, Policy for allocating null mask. Defaults to RETAIN.
 * @param mr Optional, The resource to use for all allocations
 * @return std::unique_ptr<column> An allocated column of same size and type of `input`
 */
std::unique_ptr<column> allocate_like(column_view input, gdf_size_type size,
                                      mask_allocation_policy mask_alloc = RETAIN,
                                      rmm::mr::device_memory_resource *mr =
				          rmm::mr::get_default_resource());

/**
 * @brief Creates a table of empty columns with the same types as the `input_table`
 *
 * Creates the `cudf::column` objects, but does not allocate any underlying device
 * memory for the column's data or bitmask.
 *
 * @param input_table Immutable view of input table to emulate
 * @return std::unique_ptr<table> A table of empty columns of same type as `input_table`
 */
std::unique_ptr<table> empty_like(table_view input_table);

/**
 * @brief Creates a table of columns with the same type and allocation size as
 * the `input_table`.
 *
 * Creates the `cudf::column` objects, and allocates underlying device memory for
 * each column matching the input columns
 *
 * @param input_table Immutable view of input table to emulate
 * @param mask_alloc Optional, Policy for allocating null mask. Defaults to RETAIN.
 * @param mr Optional, The resource to use for all allocations
 * @return std::unique_ptr<table> A table of columns with same type and allocation size as `input_table`
 */
std::unique_ptr<table> allocate_like(table_view input_table,
                                     mask_allocation_policy mask_alloc = RETAIN,
                                     rmm::mr::device_memory_resource *mr =
				         rmm::mr::get_default_resource());

/**
 * @brief Creates a table of columns with the specified size and same type as
 * the `input_table`.
 *
 * Creates the `cudf::column` objects, and allocates underlying device memory for
 * each column matching the input columns
 *
 * @param input_table Immutable view of input table to emulate
 * @param size The size of the columns to allocate
 * @param mask_alloc Optional, Policy for allocating null mask. Defaults to RETAIN.
 * @param mr Optional, The resource to use for all allocations
 * @return std::unique_ptr<table> A table of columns with same type as `input_table` and of specified size
 */
std::unique_ptr<table> allocate_like(table_view input_table,
                                     gdf_size_type size,
                                     mask_allocation_policy mask_alloc = RETAIN,
                                     rmm::mr::device_memory_resource *mr =
				         rmm::mr::get_default_resource());

}  // namespace experimental
}  // namespace cudf
