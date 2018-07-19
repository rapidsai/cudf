#ifndef GROUPBY_COMPUTE_API_H
#define GROUPBY_COMPUTE_API_H

#include <cuda_runtime.h>
#include <limits>
#include <memory>
#include "groupby_kernels.cuh"

constexpr unsigned int DEFAULT_HASH_TABLE_OCCUPANCY{50};
constexpr unsigned int THREAD_BLOCK_SIZE{256};

template<typename groupby_type,
         typename aggregation_type,
         typename size_type,
         typename aggregation_operation>
cudaError_t GroupbyHash(const groupby_type * const in_groupby_column,
                        const aggregation_type * const in_aggregation_column,
                        const size_type in_column_size,
                        groupby_type * const out_groupby_column,
                        aggregation_type * const out_aggregation_column,
                        size_type * out_size,
                        aggregation_operation aggregation_op)
{

  using map_type = concurrent_unordered_map<groupby_type, aggregation_type, std::numeric_limits<groupby_type>::max()>;

  cudaError_t error{cudaSuccess};

  // Inputs cannot be null
  if(in_groupby_column == nullptr || in_aggregation_column == nullptr)
    return cudaErrorNotPermitted;

  // Input size cannot be 0 or negative
  if(in_column_size <= 0)
    return cudaErrorNotPermitted;

  // Output buffers must already be allocated
  if(out_groupby_column == nullptr || out_aggregation_column == nullptr)
    return cudaErrorNotPermitted;

  std::unique_ptr<map_type> the_map;

  const size_type hash_table_size = (in_column_size * 100 / DEFAULT_HASH_TABLE_OCCUPANCY);

  const dim3 build_grid_size ((in_column_size + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE, 1, 1);
  const dim3 block_size (THREAD_BLOCK_SIZE, 1, 1);

  the_map.reset(new map_type(hash_table_size, aggregation_operation::IDENTITY));

  error = cudaDeviceSynchronize();
  if(error != cudaSuccess)
    return error;

  build_aggregation_table<<<build_grid_size, block_size>>>(the_map.get(), 
                                                           in_groupby_column, 
                                                           in_aggregation_column,
                                                           in_column_size,
                                                           aggregation_op);
  error = cudaDeviceSynchronize();
  if(error != cudaSuccess)
    return error;

  unsigned int * global_write_index{nullptr};
  cudaMallocManaged(&global_write_index, sizeof(unsigned int));
  *global_write_index = 0;

  error = cudaDeviceSynchronize();
  if(error != cudaSuccess)
    return error;

  const dim3 extract_grid_size ((the_map->size() + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE, 1, 1);

  extract_groupby_result<<<extract_grid_size, block_size>>>(the_map.get(),
                                                            the_map->size(),
                                                            out_groupby_column,
                                                            out_aggregation_column,
                                                            global_write_index);
  error = cudaDeviceSynchronize();
  if(error != cudaSuccess)
    return error;

  *out_size = *global_write_index;

  return error;
}
#endif




