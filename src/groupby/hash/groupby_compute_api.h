#ifndef GROUPBY_COMPUTE_API_H
#define GROUPBY_COMPUTE_API_H

#include <cuda_runtime.h>
#include <moderngpu/context.hxx>
#include "groupby_kernels.cuh"
#include "aggregation_operations.h"

constexpr unsigned int DEFAULT_HASH_TABLE_OCCUPANCY{50};
constexpr unsigned int THREAD_BLOCK_SIZE{256};

template<typename groupby_type,
         typename aggregation_type,
         typename size_type>
cudaError_t GroupbyHash(mgpu::context_t &compute_ctx, 
                        const groupby_type * const groupby_column_in,
                        const aggregation_type * const aggregation_column_in,
                        const size_type column_size,
                        groupby_type * const groupby_column_out,
                        aggregation_type * const aggregation_column_out,
                        aggregation_operation aggregation_op)
{

  using map_type = concurrent_unordered_map<groupby_type, aggregation_type, std::numeric_limits<groupby_type>::max()>;

  cudaError_t error{cudaSuccess};

  // Inputs cannot be null
  if(groupby_column_in == nullptr || aggregation_column_in == nullptr)
    return cudaErrorNotPermitted;

  // Input size cannot be 0 or negative
  if(column_size <= 0)
    return cudaErrorNotPermitted;

  // Output buffers must already be allocated
  if(groupby_column_out == nullptr || aggregation_column_out == nullptr)
    return cudaErrorNotPermitted;

  std::unique_ptr<map_type> the_map;

  const size_type hash_table_size = (column_size * 100 / DEFAULT_HASH_TABLE_OCCUPANCY);

  const dim3 build_grid_size ((column_size + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE, 1, 1);
  const dim3 block_size (THREAD_BLOCK_SIZE, 1, 1);

  switch(aggregation_op)
  {
    case aggregation_operation::max_op:
      the_map.reset(new map_type(hash_table_size, max_op<aggregation_type>::IDENTITY));
      error = cudaDeviceSynchronize();
      if(error != cudaSuccess)
        return error;
      build_aggregation_table<<<build_grid_size, block_size>>>(the_map.get(), 
                                                         groupby_column_in, 
                                                         groupby_column_out,
                                                         column_size,
                                                         max_op<aggregation_type>());
      error = cudaDeviceSynchronize();
      if(error != cudaSuccess)
        return error;
      break;

    case aggregation_operation::min_op:
      the_map.reset(new map_type(hash_table_size, min_op<aggregation_type>::IDENTITY));
      error = cudaDeviceSynchronize();
      if(error != cudaSuccess)
        return error;
      build_aggregation_table<<<build_grid_size, block_size>>>(the_map.get(), 
                                                         groupby_column_in, 
                                                         groupby_column_out,
                                                         column_size,
                                                         min_op<aggregation_type>());
      error = cudaDeviceSynchronize();
      if(error != cudaSuccess)
        return error;
      break;

    default:
      return cudaErrorNotSupported;
      break;
  }

  unsigned int * global_write_index{nullptr};
  cudaMallocManaged(&global_write_index, sizeof(unsigned int));
  *global_write_index = 0;

  error = cudaDeviceSynchronize();
  if(error != cudaSuccess)
    return error;

  const dim3 extract_grid_size ((the_map->size() + THREAD_BLOCK_SIZE - 1) / THREAD_BLOCK_SIZE, 1, 1);

  extract_groupby_result<<<extract_grid_size, block_size>>>(the_map.get(),
                                                            the_map->size(),
                                                            groupby_column_out,
                                                            aggregation_column_out,
                                                            global_write_index);
  error = cudaDeviceSynchronize();
  if(error != cudaSuccess)
    return error;

  return error;
}
#endif




