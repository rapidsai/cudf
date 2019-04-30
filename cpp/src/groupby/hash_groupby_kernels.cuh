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

#ifndef GROUPBY_KERNELS_H
#define GROUPBY_KERNELS_H

#include "hash/concurrent_unordered_map.cuh"
#include <table/device_table.cuh>

#include "aggregation_operations.hpp"

/* --------------------------------------------------------------------------*/
/** 
 * @brief Takes in two columns of equal length. One column to groupby and the
 * other to aggregate with a provided aggregation operation. The pair
 * (groupby[i], aggregation[i]) are inserted as a key-value pair into a hash table.
 * When inserting values for the same key multiple times, the aggregation operation 
 * is performed between the existing value and the new value.
 *            
 * 
 * @param the_map The hash table to use for building the aggregation table
 * @param groupby_column The column used as keys into the hash table
 * @param aggregation_column The column used as the values of the hash table
 * @param column_size The size of both columns
 * @param op The aggregation operation to perform between new and existing hash table values
 * 
 * @returns   
 */
/* ----------------------------------------------------------------------------*/
template<typename map_type, 
         typename aggregation_operation,
         typename aggregation_type,
         typename row_comparator>
__global__ void build_aggregation_table(map_type * const __restrict__ the_map,
                                        device_table groupby_input_table,
                                        const aggregation_type * const __restrict__ aggregation_column,
                                        gdf_size_type column_size,
                                        aggregation_operation op,
                                        row_comparator the_comparator)
{
  gdf_size_type i = threadIdx.x + blockIdx.x * blockDim.x;

  while( i < column_size ){

    // Hash the current row of the input table
    const auto row_hash = hash_row(groupby_input_table, i);

    // Attempt to insert the current row's index.  
    // The hash value of the row will determine the write location.
    // The rows at the current row index and the existing row index 
    // will be compared for equality. If they are equal, the aggregation
    // operation is performed.
    the_map->insert(thrust::make_pair(i, aggregation_column[i]), 
                    op,
                    the_comparator,
                    true,
                    row_hash);

    i += blockDim.x * gridDim.x;
  }
}

// Specialization for COUNT operation that ignores the values of the input aggregation column
template<typename map_type,
         typename aggregation_type,
         typename row_comparator>
__global__ void build_aggregation_table(map_type * const __restrict__ the_map,
                                        device_table groupby_input_table,
                                        const aggregation_type * const __restrict__ aggregation_column,
                                        gdf_size_type column_size,
                                        count_op<typename map_type::mapped_type> op,
                                        row_comparator the_comparator)
{
  gdf_size_type i = threadIdx.x + blockIdx.x * blockDim.x;

  // Hash the current row of the input table
  const auto row_hash = hash_row(groupby_input_table,i);

  while( i < column_size ){

    // When the aggregator is COUNT, ignore the aggregation column and just insert '0'
    // Attempt to insert the current row's index.  
    // The hash value of the row will determine the write location.
    // The rows at the current row index and the existing row index 
    // will be compared for equality. If they are equal, the aggregation
    // operation is performed.
    the_map->insert(thrust::make_pair(i, static_cast<typename map_type::mapped_type>(0)), 
                    op,
                    the_comparator,
                    true,
                    row_hash);
    i += blockDim.x * gridDim.x;
  }
}

/* --------------------------------------------------------------------------*/
/** 
 * @brief Extracts the keys and their respective values from the hash table
 * and returns them as contiguous arrays.
 * 
 * @param the_map The hash table to extract from 
 * @param map_size The total capacity of the hash table
 * @param groupby_out_column The output array for the hash table keys
 * @param aggregation_out_column The output array for the hash table values
 * @param global_write_index A variable in device global memory used to coordinate
 * where threads write their output
 * 
 * @returns   
 */
/* ----------------------------------------------------------------------------*/
template<typename map_type,
         typename aggregation_type>
__global__ void extract_groupby_result(const map_type * const __restrict__ the_map,
                                       const size_t map_size,
                                       device_table groupby_output_table,
                                       device_table groupby_input_table,
                                       aggregation_type * const __restrict__ aggregation_out_column,
                                       gdf_size_type * const global_write_index)
{
  size_t i = threadIdx.x + blockIdx.x * blockDim.x;

  constexpr typename map_type::key_type unused_key{map_type::get_unused_key()};

  const typename map_type::value_type * const __restrict__ hashtabl_values = the_map->data();

  // TODO: Use _shared_ thread block cache for writing temporary ouputs and then
  // write to the global output
  while(i < map_size){

    const typename map_type::key_type current_key = hashtabl_values[i].first;

    if( current_key != unused_key){
      const gdf_size_type thread_write_index = atomicAdd(global_write_index, 1);

      // Copy the row at current_key from the input table to the row at
      // thread_write_index in the output table
      copy_row(groupby_output_table, thread_write_index, groupby_input_table,
               current_key);

      aggregation_out_column[thread_write_index] = hashtabl_values[i].second;
    }
    i += gridDim.x * blockDim.x;
  }
}
#endif
