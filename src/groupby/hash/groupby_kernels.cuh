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

#include <gdf/utils.h>
#include "../../hashmap/concurrent_unordered_map.cuh"
#include "aggregation_operations.cuh"
#include "../../gdf_table.cuh"


/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  Indicates the state of a hash bucket in the hash table.
 */
/* ----------------------------------------------------------------------------*/
enum bucket_state : int
{
  EMPTY = 0,        /** Indicates that the hash bucket is empty */
  NULL_VALUE,       /** Indicates that the bucket's payload contains a NULL value */
  VALID_VALUE       /** Indicates that the bucket's payload contains a valid value */
};
using state_t = std::underlying_type<bucket_state>::type;

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis Takes in two columns of equal length. One column to groupby and the
 * other to aggregate with a provided aggregation operation. The pair
 * (groupby[i], aggregation[i]) are inserted as a key-value pair into a hash table.
 * When inserting values for the same key multiple times, the aggregation operation 
 * is performed between the existing value and the new value.
 *            
 * 
 * @Param the_map The hash table to use for building the aggregation table
 * @Param groupby_column The column used as keys into the hash table
 * @Param aggregation_column The column used as the values of the hash table
 * @Param column_size The size of both columns
 * @Param op The aggregation operation to perform between new and existing hash table values
 * 
 * @Returns   
 */
/* ----------------------------------------------------------------------------*/
template<typename map_type, 
         typename aggregation_operation,
         typename aggregation_type,
         typename size_type,
         typename row_comparator>
__global__ void build_aggregation_table(map_type * const __restrict__ the_map,
                                        gdf_table<size_type> const & groupby_input_table,
                                        const aggregation_type * const __restrict__ aggregation_column,
                                        gdf_valid_type const * const __restrict__ aggregation_validitity_mask,
                                        bucket_state * const __restrict__ hash_bucket_states,
                                        size_type column_size,
                                        aggregation_operation op,
                                        row_comparator the_comparator)
{
  size_type i = threadIdx.x + blockIdx.x * blockDim.x;

  const auto map_start = the_map->begin();

  while( i < column_size ){

    // Only insert into the hash table if the row is valid
    if( true == groupby_input_table.is_row_valid(i) )
    {

      // Hash the current row of the input table
      const auto row_hash = groupby_input_table.hash_row(i);

      if(false == gdf_is_valid(aggregation_validitity_mask,i))
      {
        // If the value in the aggregation column is NULL, only insert
        // the key without a payload 
        const size_type insert_location = the_map->insert_key(i, the_comparator, true, row_hash);

        // If the aggregation value is NULL, and the hash bucket is empty,
        // then set the state of the bucket to show that there is a NULL value for this key
        // The casts are required to cast the enum type to a type supported by 
        // atomicCAS
        // TODO Use a bitmask instead of a 32 bit flag for every bucket
        atomicCAS(reinterpret_cast<state_t*>(&hash_bucket_states[insert_location]), 
                  static_cast<state_t>(bucket_state::EMPTY), 
                  static_cast<state_t>(bucket_state::NULL_VALUE));
      }
      else
      {

        // Attempt to insert the current row's index.  
        // The hash value of the row will determine the write location.
        // The rows at the current row index and the existing row index 
        // will be compared for equality. If they are equal, the aggregation
        // operation is performed.
        const size_type insert_location = the_map->insert(thrust::make_pair(i, aggregation_column[i]), 
                                                           op,
                                                           the_comparator,
                                                           true,
                                                           row_hash);

        // Indicate that the payload for this hash bucket is valid
        atomicExch(reinterpret_cast<state_t*>(&hash_bucket_states[insert_location]),
                                              static_cast<state_t>(bucket_state::VALID_VALUE));
      }
    }

    i += blockDim.x * gridDim.x;
  }
}

// Specialization for COUNT operation that ignores the values of the input aggregation column
template<typename map_type,
         typename aggregation_type,
         typename size_type,
         typename row_comparator>
__global__ void build_aggregation_table(map_type * const __restrict__ the_map,
                                        gdf_table<size_type> const & groupby_input_table,
                                        const aggregation_type * const __restrict__ aggregation_column,
                                        gdf_valid_type const * const __restrict__ aggregation_validitity_mask,
                                        bucket_state * const __restrict__ hash_bucket_states,
                                        size_type column_size,
                                        count_op<typename map_type::mapped_type> op,
                                        row_comparator the_comparator)
{

  auto map_start = the_map->begin();
  size_type i = threadIdx.x + blockIdx.x * blockDim.x;

  while( i < column_size ){

    // Only insert into the hash table if the the row is valid
    if(groupby_input_table.is_row_valid(i) )
    {
      // Hash the current row of the input table
      const auto row_hash = groupby_input_table.hash_row(i);

      size_type insert_location{0};
      if(false == gdf_is_valid(aggregation_validitity_mask,i))
      {
        // For COUNT, the aggregation result value can never be NULL, i.e., counting an
        // aggregation column of all NULL should return 0. Therefore, insert the key 
        // only and set the state to VALID. Since the payload is initialized with 0,
        // it will return 0 for a column of all nulls as expected
       insert_location = the_map->insert_key(i, the_comparator, true, row_hash);
      }
      else
      {
        // When the aggregator is COUNT, ignore the aggregation column and just insert '0'
        // Attempt to insert the current row's index.  
        // The hash value of the row will determine the write location.
        // The rows at the current row index and the existing row index 
        // will be compared for equality. If they are equal, the aggregation
        // operation is performed.
        insert_location = the_map->insert(thrust::make_pair(i, static_cast<typename map_type::mapped_type>(0)), 
                                                           op,
                                                           the_comparator,
                                                           true,
                                                           row_hash);
      }

      // Indicate that the payload for this hash bucket is valid
      atomicExch(reinterpret_cast<state_t*>(&hash_bucket_states[insert_location]),
                                            static_cast<state_t>(bucket_state::VALID_VALUE));
    }
    i += blockDim.x * gridDim.x;
  }
}

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis Extracts the keys and their respective values from the hash table
 * and returns them as contiguous arrays.
 * 
 * @Param the_map The hash table to extract from 
 * @Param map_size The total capacity of the hash table
 * @Param groupby_out_column The output array for the hash table keys
 * @Param aggregation_out_column The output array for the hash table values
 * @Param global_write_index A variable in device global memory used to coordinate
 * where threads write their output
 * 
 */
/* ----------------------------------------------------------------------------*/
template<typename map_type,
         typename size_type,
         typename aggregation_type>
__global__ void extract_groupby_result(const map_type * const __restrict__ the_map,
                                       const size_type map_size,
                                       const bucket_state * const __restrict__ hash_bucket_states,
                                       gdf_table<size_type> & groupby_output_table,
                                       gdf_table<size_type> const & groupby_input_table,
                                       aggregation_type * const __restrict__ aggregation_out_column,
                                       gdf_valid_type * aggregation_out_valid_mask,
                                       size_type * const global_write_index)
{
  size_type i = threadIdx.x + blockIdx.x * blockDim.x;

  const typename map_type::value_type * const __restrict__ hashtabl_values = the_map->data();

  // TODO: Use _shared_ thread block cache for writing temporary ouputs and then
  // write to the global output
  while(i < map_size){

    const bucket_state current_state = hash_bucket_states[i];

    // If the hash bucket isn't empty, then we need to add it to the output
    if(bucket_state::EMPTY != current_state)
    {
      const typename map_type::key_type output_row = hashtabl_values[i].first;
      const size_type thread_write_index = atomicAdd(global_write_index, 1);

      // Copy the row from the input table to the row at
      // thread_write_index in the output table
      groupby_output_table.copy_row(groupby_input_table, 
                                    thread_write_index,
                                    output_row);

      // If this bucket holds a valid aggregation value, copy it to the
      // aggregation output and set it's validity bit
      if( bucket_state::NULL_VALUE != current_state )
      {
        aggregation_out_column[thread_write_index] = hashtabl_values[i].second;
        
        // Set the valid bit for this row. Need to cast the valid mask type
        // to a 32 bit type where atomics are supported
        if(nullptr != aggregation_out_valid_mask)
        {
          // FIXME Replace with a standard `set_bit` function
          uint32_t * valid_mask32 = reinterpret_cast<uint32_t*>(aggregation_out_valid_mask);
          const uint32_t output_bit32 = (uint32_t(1) << (thread_write_index % uint32_t(32)));
          uint32_t * output_mask32 = &(valid_mask32[(thread_write_index / uint32_t(32))]);
          atomicOr(output_mask32, output_bit32);
        }
      }
    }

    i += gridDim.x * blockDim.x;
  }
}
#endif
