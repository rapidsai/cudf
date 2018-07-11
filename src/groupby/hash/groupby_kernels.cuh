
#include "../../hashmap/concurrent_unordered_map.cuh"

template<typename map_type, typename aggregation_type>
__global__ void build_aggregation_table(map_type * const the_map,
                                        const typename map_type::key_type * const groupby_column,
                                        const typename map_type::mapped_type * const aggregation_column,
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
