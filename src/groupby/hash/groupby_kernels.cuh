#ifndef GROUPBY_KERNELS_H
#define GROUPBY_KERNELS_H

#include "../../hashmap/concurrent_unordered_map.cuh"

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
template<typename map_type, typename aggregation_type>
__global__ void build_aggregation_table(map_type * const __restrict__ the_map,
                                        const typename map_type::key_type * const __restrict__ groupby_column,
                                        const typename map_type::mapped_type * const __restrict__ aggregation_column,
                                        const typename map_type::size_type column_size,
                                        aggregation_type op)
{
  using size_type = typename map_type::size_type;

  size_type i = threadIdx.x + blockIdx.x * blockDim.x;

  while( i < column_size ){
    the_map->insert(thrust::make_pair(groupby_column[i], aggregation_column[i]), op);
    i += blockDim.x * gridDim.x;
  }

}

template<typename map_type>

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
 * @Returns   
 */
/* ----------------------------------------------------------------------------*/
__global__ void extract_groupby_result(const map_type * const __restrict__ the_map,
                                       const typename map_type::size_type map_size,
                                       typename map_type::key_type * const __restrict__ groupby_out_column,
                                       typename map_type::mapped_type * const __restrict__ aggregation_out_column,
                                       unsigned int * const global_write_index)
{
  using size_type = typename map_type::size_type;
  using key_type = typename map_type::key_type;
  
  //const size_type map_size = the_map->get_size();

  size_type i = threadIdx.x + blockIdx.x * blockDim.x;

  constexpr key_type unused_key{map_type::get_unused_key()};

  const typename map_type::value_type * const __restrict__ hashtabl_values = the_map->data();

  // TODO: Use _shared_ thread block cache for writing temporary ouputs and then
  // write to the global output
  while(i < map_size){
    const key_type current_key = hashtabl_values[i].first;
    if( current_key != unused_key){
      const size_type thread_write_index = atomicAdd(global_write_index, 1);
      groupby_out_column[thread_write_index] = current_key;
      aggregation_out_column[thread_write_index] = hashtabl_values[i].second;
    }
    i += gridDim.x * blockDim.x;
  }
}
#endif
