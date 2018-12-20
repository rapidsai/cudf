/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#include "cudf.h"


/**
 * @brief Operations for copying from one column to another
 * @file copying_ops.cu
 */

/** 
 * @brief Scatters the values of a source column into a destination column.
 * 
 * Scatters the elements of the source column into the destination column according
 * to a scatter map such that source[i] will be scattered to destination[ scatter_map[i] ]
 * 
 * The number of elements in the scatter_map must equal the number of elements in the source
 * column. 
 * 
 * If any index in scatter_map is outside the range of [0, destination->size), it 
 * will be ignored.
 * 
 * If the same index appears more than once in scatter_map, the result is undefined.
 * 
 * @Param[in] source The input column whose rows will be scattered
 * @Param[in] scatter_map An array that maps rows in the input column
 * to rows in the output column. The size of the the scatter_map must equal
 * the size of the source column. 
 * @Param[out] destination A preallocated column equal in size to the source column that 
 * will contain the rearrangement of the source column based on the mapping determined 
 * by the scatter_map 
 * 
 * @Returns GDF_SUCCESS upon successful completion
 */
gdf_error gdf_scatter(gdf_column const * source, gdf_column const * scatter_map, gdf_column * destination)
{
    return GDF_SUCCESS;
}

/**
 * @brief Gathers the values from the source column into the destination column.
 * 
 * Gathers the elements of the source column into the destination column according 
 * to a gather map such that destination[i] will contain source[ gather_map[i] ]
 * 
 * The number of elements in the gather_map must equal the number of elements in the 
 * destination column. 
 * 
 * If gather_map[i] is outside the range [0, source->size),
 * it will be ignored and the value of destination[i] will not be modified.
 * 
 * If the same index appears more than once in gather_map, the result is undefined.
 * 
 * @param[in] source The input column whose rows will be gathered
 * @param[in] gather_map An array of indices that maps the rows in the source column
 * to rows in the destination column. 
 * @param[out] destination The preallocated column equal in size 
 */
gdf_error gdf_gather(gdf_column const * source, gdf_column const * gather_map, gdf_column * destination)
{
    return GDF_SUCCESS;
}
