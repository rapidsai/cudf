/*
 * Copyright 2018-2019 BlazingDB, Inc.
 *     Copyright 2018 Jean Pierre Huaroto <jeanpierre@blazingdb.com>
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

#include <type_traits>
#include <algorithm>

#include <cudf/cudf.h>
#include <utilities/cudf_utils.h>
#include <utilities/error_utils.hpp>

#include <table/device_table.cuh>
#include <table/device_table_row_operators.cuh>

#include <rmm/thrust_rmm_allocator.h>

/* --------------------------------------------------------------------------*/
/** 
 * @brief Sorts an array of gdf_column.
 * 
 * @param[in] cols Array of gdf_columns
 * @param[in] asc_desc Device array of sort order types for each column
 * (0 is ascending order and 1 is descending). If NULL is provided defaults
 * to ascending order for evey column.
 * @param[in] ncols # columns
 * @param[in] flag_nulls_are_smallest Flag to indicate if nulls are to be considered
 * smaller than non-nulls or viceversa
 * @param[out] output_indices Pre-allocated gdf_column to be filled
 * with sorted indices
 * 
 * @returns GDF_SUCCESS upon successful completion
 */
/* ----------------------------------------------------------------------------*/
gdf_error gdf_order_by(gdf_column const* const* cols,
                       int8_t* asc_desc,
                       size_t num_inputs,
                       gdf_column* output_indices,
                       gdf_context * context)                       
{
  GDF_REQUIRE(cols != nullptr && output_indices != nullptr, GDF_DATASET_EMPTY);
  GDF_REQUIRE(cols[0]->size == output_indices->size, GDF_COLUMN_SIZE_MISMATCH);
  /* NOTE: providing support for indexes to be multiple different types explodes compilation time, such that it become infeasible */
  GDF_REQUIRE(output_indices->dtype == GDF_INT32, GDF_UNSUPPORTED_DTYPE);
    
  bool nulls_are_smallest = false;
  if (context->flag_null_sort_behavior == GDF_NULL_AS_SMALLEST) {
  /* When sorting NULLS will be treated as the smallest number */
    nulls_are_smallest = true;
  } 
  
  cudaStream_t stream = 0;
  gdf_index_type* d_indx = static_cast<gdf_index_type*>(output_indices->data);
  gdf_size_type nrows = cols[0]->size;

  thrust::sequence(rmm::exec_policy(stream)->on(stream), d_indx, d_indx+nrows, 0);
  auto table = device_table::create(num_inputs, cols, stream);
  bool nullable = table.get()->has_nulls();
 
  if (nullable){
    auto ineq_op = row_inequality_comparator<true>(*table, nulls_are_smallest, asc_desc); 
    thrust::sort(rmm::exec_policy(stream)->on(stream),
                  d_indx, d_indx+nrows,
                  ineq_op);				        
  } else {
    auto ineq_op = row_inequality_comparator<false>(*table, nulls_are_smallest, asc_desc); 
    thrust::sort(rmm::exec_policy(stream)->on(stream),
                  d_indx, d_indx+nrows,
                  ineq_op);				        
  }
  
  return GDF_SUCCESS;
}
