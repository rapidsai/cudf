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

#include <cuda_runtime.h>
#include <future>
#include <gdf/errorutils.h>

#include "join_kernels.cuh"
#include "../../gdf_table.cuh"

// TODO for Arrow integration:
//   1) replace mgpu::context_t with a new CudaComputeContext class (see the design doc)
//   2) replace cudaError_t with arrow::Status
//   3) replace input iterators & input counts with arrow::Datum
//   3) replace output iterators & output counts with arrow::ArrayData

#include <moderngpu/context.hxx>

#include <moderngpu/kernel_scan.hxx>

constexpr int64_t DEFAULT_HASH_TABLE_OCCUPANCY = 50;
constexpr int DEFAULT_CUDA_BLOCK_SIZE = 128;
constexpr int DEFAULT_CUDA_CACHE_SIZE = 128;

/* --------------------------------------------------------------------------*/
/** 
 * @Synopsis  Gives an estimate of the size of the join output produced when
 * joining two tables together. If the two tables are of relatively equal size,
 * then the returned output size will be the exact output size. However, if the
 * probe table is significantly larger than the build table, then we attempt
 * to estimate the output size by using only a subset of the rows in the probe table.
 * 
 * @Param build_table The right hand table
 * @Param probe_table The left hand table
 * @Param hash_table A hash table built on the build table that maps the index
 * of every row to the hash value of that row.
 * 
 * @Returns An estimate of the size of the output of the join operation
 */
/* ----------------------------------------------------------------------------*/
template <JoinType join_type,
          typename multimap_type,
          typename size_type>
gdf_error estimate_join_output_size(gdf_table<size_type> const & build_table,
                                    gdf_table<size_type> const & probe_table,
                                    multimap_type const & hash_table,
                                    size_type * join_output_size_estimate)
{
  const size_type build_table_num_rows{build_table.get_column_length()};
  const size_type probe_table_num_rows{probe_table.get_column_length()};
  
  // If the probe table is significantly larger (5x) than the build table, 
  // then we attempt to only use a subset of the probe table rows to compute an
  // estimate of the join output size.
  size_type probe_to_build_ratio{0};
  if(build_table_num_rows > 0) {
    probe_to_build_ratio = static_cast<size_type>(std::ceil(static_cast<float>(probe_table_num_rows)/build_table_num_rows));
  }
  else {
    // If the build table is empty, we know exactly how large the output
    // will be for the different types of joins and can return immediately
    switch(join_type)
    {
      case JoinType::INNER_JOIN:
        {
          // Inner join with an empty table will have no output
          *join_output_size_estimate = 0;
          break;
        }
      case JoinType::LEFT_JOIN:
        {
          // Left join with an empty table will have an output of NULL rows
          // equal to the number of rows in the probe table
          *join_output_size_estimate = probe_table_num_rows;
          break;
        }
      default:
        return GDF_UNSUPPORTED_JOIN_TYPE;
    }
    return GDF_SUCCESS;
  }

  size_type sample_probe_num_rows{probe_table_num_rows};
  constexpr size_type MAX_RATIO{5};
  if(probe_to_build_ratio > MAX_RATIO)
  {
    sample_probe_num_rows = build_table_num_rows;
  }

  // Allocate storage for the counter used to get the size of the join output
  size_type * d_size_estimate{nullptr};
  size_type h_size_estimate{0};

  CUDA_TRY(cudaMallocHost(&d_size_estimate, sizeof(size_type)));
  *d_size_estimate = 0;

  CUDA_TRY( cudaGetLastError() );

  // Continue probing with a subset of the probe table until either:
  // a non-zero output size estimate is found OR
  // all of the rows in the probe table have been sampled
  do{

    sample_probe_num_rows = std::min(sample_probe_num_rows, probe_table_num_rows);

    *d_size_estimate = 0;

    constexpr int block_size{DEFAULT_CUDA_BLOCK_SIZE};
    const size_type probe_grid_size{(sample_probe_num_rows + block_size -1)/block_size};
    
    // Probe the hash table without actually building the output to simply
    // find what the size of the output will be.
    compute_join_output_size<join_type,
                             multimap_type,
                             size_type,
                             block_size,
                             DEFAULT_CUDA_CACHE_SIZE>
    <<<probe_grid_size, block_size>>>(&hash_table,
                                      build_table,
                                      probe_table,
                                      sample_probe_num_rows,
                                      d_size_estimate);

    // Device sync is required to ensure d_size_estimate is updated
    CUDA_TRY( cudaDeviceSynchronize() );
    
    	
    // Increase the estimated output size by a factor of the ratio between the
    // probe and build tables
    h_size_estimate = *d_size_estimate * probe_to_build_ratio;

    // If the size estimate is non-zero, then we have a valid estimate and can break
    // If sample_probe_num_rows >= probe_table_num_rows, then we've sampled the entire
    // probe table, in which case the estimate is exact and we can break 
    if((h_size_estimate > 0) 
       || (sample_probe_num_rows >= probe_table_num_rows))
    {
      break;
    }

    // If the size estimate is zero, then double the number of sampled rows in the probe
    // table. Reduce the ratio of the number of probe rows sampled to the 
    // number of rows in the build table by the same factor
    if(0 == h_size_estimate)
    {
      constexpr size_type GROW_RATIO{2};
      sample_probe_num_rows *= GROW_RATIO;
      probe_to_build_ratio = static_cast<size_type>(std::ceil(static_cast<float>(probe_to_build_ratio)/GROW_RATIO));
    }

  } while(true);

  CUDA_TRY( cudaFreeHost(d_size_estimate) );

  *join_output_size_estimate = h_size_estimate;

  return GDF_SUCCESS;
}

/* --------------------------------------------------------------------------*/
/**
* @Synopsis  Performs a hash-based join between two sets of gdf_tables.
*
* @Param compute_ctx The Modern GPU context
* @Param joined_output The output of the join operation
* @Param left_table The left table to join
* @Param right_table The right table to join
* @Param flip_results Flag that indicates whether the left and right tables have been
* switched, indicating that the output indices should also be flipped
* @tparam join_type The type of join to be performed
* @tparam hash_value_type The data type to be used for the Keys in the hash table
* @tparam output_index_type The data type to be used for the output indices
*
* @Returns  cudaSuccess upon successful completion of the join. Otherwise returns
* the appropriate CUDA error code
*/
/* ----------------------------------------------------------------------------*/
template<JoinType join_type,
         typename output_index_type,
         typename size_type>
gdf_error compute_hash_join(mgpu::context_t & compute_ctx,
                            gdf_column * const output_l, 
                            gdf_column * const output_r,
                            gdf_table<size_type> const & left_table,
                            gdf_table<size_type> const & right_table,
                            bool flip_results = false)
{
  gdf_error gdf_error_code{GDF_SUCCESS};

  gdf_column_view(output_l, nullptr, nullptr, 0, N_GDF_TYPES);
  gdf_column_view(output_r, nullptr, nullptr, 0, N_GDF_TYPES);

  // The LEGACY allocator allocates the hash table array with normal cudaMalloc,
  // the non-legacy allocator uses managed memory
#ifdef HT_LEGACY_ALLOCATOR
  using multimap_type = concurrent_unordered_multimap<hash_value_type,
                                                      output_index_type,
                                                      size_type,
                                                      std::numeric_limits<hash_value_type>::max(),
                                                      std::numeric_limits<output_index_type>::max(),
                                                      default_hash<hash_value_type>,
                                                      equal_to<hash_value_type>,
                                                      legacy_allocator< thrust::pair<hash_value_type, output_index_type> > >;
#else
  using multimap_type = concurrent_unordered_multimap<hash_value_type,
                                                      output_index_type,
                                                      size_type,
                                                      std::numeric_limits<hash_value_type>::max(),
                                                      std::numeric_limits<size_type>::max()>;
#endif

  // Hash table will be built on the right table
  gdf_table<size_type> const & build_table{right_table};
  const size_type build_table_num_rows{build_table.get_column_length()};
  
  // Probe with the left table
  gdf_table<size_type> const & probe_table{left_table};
  const size_type probe_table_num_rows{probe_table.get_column_length()};

  // Calculate size of hash map based on the desired occupancy
  size_type hash_table_size{(build_table_num_rows * 100) / DEFAULT_HASH_TABLE_OCCUPANCY};

  // It's possible that the hash table size will be zero, in which case
  // we still need to allocate something.
  hash_table_size = std::max(hash_table_size, size_type(1));
 
  std::unique_ptr<multimap_type> hash_table(new multimap_type(hash_table_size));

  // FIXME: use GPU device id from the context?
  // but moderngpu only provides cudaDeviceProp
  // (although should be possible once we move to Arrow)
  hash_table->prefetch(0);

  CUDA_TRY( cudaDeviceSynchronize() );

  // Allocate a gdf_error for the device to hold error code returned from
  // the build kernel and intialize with GDF_SUCCESS
  // Use Page Locked memory to avoid overhead of memcpys
  gdf_error * d_gdf_error_code{nullptr};
  CUDA_TRY( cudaMallocHost(&d_gdf_error_code, sizeof(gdf_error)) );
  *d_gdf_error_code = GDF_SUCCESS;

  constexpr int block_size{DEFAULT_CUDA_BLOCK_SIZE};

  // build the hash table
  if(build_table_num_rows > 0)
  {
    const size_type build_grid_size{(build_table_num_rows + block_size - 1)/block_size};
    build_hash_table<<<build_grid_size, block_size>>>(hash_table.get(),
                                                      build_table,
                                                      build_table_num_rows,
                                                      d_gdf_error_code);
    
    // Device synch is required to ensure d_gdf_error_code 
    // has been written
    CUDA_TRY( cudaDeviceSynchronize() );
  }

  // Check error code from the kernel
  gdf_error_code = *d_gdf_error_code;
  if(GDF_SUCCESS != gdf_error_code){
    return gdf_error_code;
  }


  size_type estimated_join_output_size{0};
  gdf_error_code = estimate_join_output_size<join_type, multimap_type>(build_table, probe_table, *hash_table, &estimated_join_output_size);

  if(GDF_SUCCESS != gdf_error_code){
    return gdf_error_code;
  }

  // If the estimated output size is zero, return immediately
  if(0 == estimated_join_output_size){
    return GDF_SUCCESS;
  }

  // Because we are approximating the number of joined elements, our approximation 
  // might be incorrect and we might have underestimated the number of joined elements. 
  // As such we will need to de-allocate memory and re-allocate memory to ensure 
  // that the final output is correct.
  size_type h_actual_found{0};
  output_index_type *output_l_ptr{nullptr};
  output_index_type *output_r_ptr{nullptr};
  bool cont = true;

  // Allocate device global counter used by threads to determine output write location
  size_type *d_global_write_index{nullptr};
  CUDA_TRY( cudaMalloc(&d_global_write_index, sizeof(size_type)) );
 
  // Because we only have an estimate of the output size, we may need to probe the
  // hash table multiple times until we've found an output buffer size that is large enough
  // to hold the output
  while(cont)
  {
    output_l_ptr = nullptr;
    output_r_ptr = nullptr;

    // Allocate temporary device buffer for join output
    CUDA_TRY( cudaMalloc(&output_l_ptr, estimated_join_output_size*sizeof(output_index_type)) );
    CUDA_TRY( cudaMalloc(&output_r_ptr, estimated_join_output_size*sizeof(output_index_type)) );
    CUDA_TRY( cudaMemsetAsync(d_global_write_index, 0, sizeof(size_type), 0) );

    const size_type probe_grid_size{(probe_table_num_rows + block_size -1)/block_size};
    
    // Do the probe of the hash table with the probe table and generate the output for the join
    probe_hash_table<join_type,
                     multimap_type,
                     hash_value_type,
                     size_type,
                     output_index_type,
                     block_size,
                     DEFAULT_CUDA_CACHE_SIZE>
    <<<probe_grid_size, block_size>>> (hash_table.get(),
                                       build_table,
                                       probe_table,
                                       probe_table.get_column_length(),
                                       output_l_ptr,
                                       output_r_ptr,
                                       d_global_write_index,
                                       estimated_join_output_size,
                                       flip_results);

    CUDA_TRY( cudaGetLastError() );

    CUDA_TRY( cudaMemcpy(&h_actual_found, d_global_write_index, sizeof(size_type), cudaMemcpyDeviceToHost));

    // The estimate was too small. Double the estimate and try again
    if(estimated_join_output_size < h_actual_found){
      cont = true;
      estimated_join_output_size *= 2;
    }
    else
    {
      cont = false;
    }
  }

  // free memory used for the counters
  CUDA_TRY( cudaFree(d_global_write_index) );

  // If the estimated join output size was larger than the actual output size,
  // then the buffers are larger than necessary. Allocate buffers of the actual 
  // output size and copy the results to the buffers of the correct size
  // FIXME Is this really necessary? It's probably okay to have the buffers be oversized
  // and avoid the extra allocation/memcopy
  if (estimated_join_output_size > h_actual_found) {
      output_index_type *copy_output_l_ptr{nullptr};
      output_index_type *copy_output_r_ptr{nullptr};
      CUDA_TRY( cudaMalloc(&copy_output_l_ptr, h_actual_found*sizeof(output_index_type)) );
      CUDA_TRY( cudaMalloc(&copy_output_r_ptr, h_actual_found*sizeof(output_index_type)) );
      CUDA_TRY( cudaMemcpy(copy_output_l_ptr, output_l_ptr, h_actual_found*sizeof(output_index_type), cudaMemcpyDeviceToDevice) );
      CUDA_TRY( cudaMemcpy(copy_output_r_ptr, output_r_ptr, h_actual_found*sizeof(output_index_type), cudaMemcpyDeviceToDevice) );
      CUDA_TRY( cudaFree(output_l_ptr) );
      CUDA_TRY( cudaFree(output_r_ptr) );
      output_l_ptr = copy_output_l_ptr;
      output_r_ptr = copy_output_r_ptr;
  }
  
  // Free the device error code 
  CUDA_TRY( cudaFreeHost(d_gdf_error_code) );
  
  // Deduce the type of the output gdf_columns
  gdf_dtype dtype;
  switch(sizeof(output_index_type)) 
  {
    case 1 : dtype = GDF_INT8;  break;
    case 2 : dtype = GDF_INT16; break;
    case 4 : dtype = GDF_INT32; break;
    case 8 : dtype = GDF_INT64; break;
  }

  gdf_column_view(output_l, output_l_ptr, nullptr, h_actual_found, dtype);
  gdf_column_view(output_r, output_r_ptr, nullptr, h_actual_found, dtype);

  return gdf_error_code;
}
