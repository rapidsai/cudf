
#include "../../hashmap/concurrent_unordered_map.cuh"

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

  constexpr key_type unused_key{the_map->get_unused_key()};

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
