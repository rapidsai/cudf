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

template<typename size_type>
struct join_pair 
{ 
  size_type first; 
  size_type second; 
};

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
  gdf_error error{GDF_SUCCESS};

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

  // Calculate size of hash map based on the desired occupancy
  const size_type hash_table_size{(build_table_num_rows * 100) / DEFAULT_HASH_TABLE_OCCUPANCY};
 
  std::unique_ptr<multimap_type> hash_table(new multimap_type(hash_table_size));

  // FIXME: use GPU device id from the context?
  // but moderngpu only provides cudaDeviceProp
  // (although should be possible once we move to Arrow)
  hash_table->prefetch(0);

  CUDA_TRY( cudaDeviceSynchronize() );

  // build the hash table
  constexpr int block_size{DEFAULT_CUDA_BLOCK_SIZE};
  const size_type build_grid_size{(build_table_num_rows + block_size - 1)/block_size};
  build_hash_table<<<build_grid_size, block_size>>>(hash_table.get(),
                                                    build_table,
                                                    build_table_num_rows);

  CUDA_TRY( cudaGetLastError() );

  // To avoid a situation where the entire probing column, left_Table, is probed into the build table (right_table) we use the following approximation technique.
  // First of all we check the ratios of the sizes between A (left) and B(right). Only if A is much bigger than B does this optimization make sense.
  // We define much bigger to be 5 times bigger as for smaller ratios, the following optimization might lose its benefit.
  // When the ratio is big enough, we will take a subset of A equal in length to B and probe (without writing outputs). We will then approximate
  // the number of joined elements as the number of found elements times the ratio.
  size_type leftSize  = left_table.get_column_length();
  size_type rightSize = right_table.get_column_length();

  size_type leftSampleSize=leftSize;
  size_type size_ratio = 1;
  if (leftSize > 5*rightSize){
  	leftSampleSize	= rightSize;
  	size_ratio		= leftSize/rightSize + 1;
  }

  // Allocate storage for the counter used to get the size of the join output
  size_type * d_join_output_size;
  size_type h_join_output_size{0};

  CUDA_TRY(cudaMalloc(&d_join_output_size, sizeof(size_type)));
  CUDA_TRY(cudaMemset(d_join_output_size, 0, sizeof(size_type)));

  // Probe with the left table
  gdf_table<size_type> const & probe_table{left_table};
  //const size_type probe_grid_size{(probe_column_length + block_size -1)/block_size};

  CUDA_TRY( cudaGetLastError() );

  // A situation can arise such that the number of elements found in the probing phase is equal to zero. This would lead us to approximating
  // the number of joined elements to be zero. As such we need to increase the subset and continue probing to get a bettter approximation value.
  do{
    if(leftSampleSize>leftSize)
      leftSampleSize=leftSize;
    // step 3ab: scan table A (left), probe the HT without outputting the joined indices. Only get number of outputted elements.
    CUDA_TRY(cudaMemset(d_join_output_size, 0, sizeof(size_type)));

    const size_type probe_grid_size{(leftSampleSize + block_size -1)/block_size};
    // Probe the hash table without actually building the output to simply
    // find what the size of the output will be.
    compute_join_output_size<join_type,
                             multimap_type,
                             size_type,
                             block_size,
                             DEFAULT_CUDA_CACHE_SIZE>
    <<<probe_grid_size, block_size>>>(hash_table.get(),
                                      build_table,
                                      probe_table,
                                      leftSampleSize,
                                      d_join_output_size);

    CUDA_TRY( cudaGetLastError() );

    CUDA_TRY( cudaMemcpy(&h_join_output_size, d_join_output_size, sizeof(size_type), cudaMemcpyDeviceToHost));

    h_join_output_size = h_join_output_size * size_ratio;

    if(h_join_output_size>0 || leftSampleSize >= leftSize)
      break;
    if(h_join_output_size==0){
      leftSampleSize  *= 2;
      size_ratio	  /= 2;
      if(size_ratio==0)
        size_ratio=1;
    }
  } while(true);

  CUDA_TRY( cudaFree(d_join_output_size) );

  // If the output size is zero, return immediately
  if(0 == h_join_output_size){
    return error;
  }

  // As we are now approximating the number of joined elements, our approximation might be incorrect and we might have underestimated the
  // number of joined elements. As such we will need to de-allocate memory and re-allocate memory to ensure that the final output is correct.
  size_type h_actual_found;
  output_index_type *output_l_ptr{nullptr};
  output_index_type *output_r_ptr{nullptr};
  bool cont = true;

  // Allocate device global counter used by threads to determine output write location
  size_type *d_global_write_index{nullptr};
  CUDA_TRY( cudaMalloc(&d_global_write_index, sizeof(size_type)) );
  int dev_ordinal{0};
  CUDA_TRY( cudaGetDevice(&dev_ordinal));
 
  while(cont){
    output_l_ptr = nullptr;
    output_r_ptr = nullptr;
  	CUDA_TRY( cudaGetDevice(&dev_ordinal));

    // Allocate temporary device buffer for join output
    CUDA_TRY( cudaMalloc(&output_l_ptr, h_join_output_size*sizeof(output_index_type)) );
    CUDA_TRY( cudaMalloc(&output_r_ptr, h_join_output_size*sizeof(output_index_type)) );
    CUDA_TRY( cudaMemsetAsync(d_global_write_index, 0, sizeof(size_type), 0) );

	const size_type probe_grid_size{(leftSize + block_size -1)/block_size};
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
                                       h_join_output_size,
                                       flip_results);

    CUDA_TRY(cudaDeviceSynchronize());

  	CUDA_TRY( cudaMemcpy(&h_actual_found, d_global_write_index, sizeof(size_type), cudaMemcpyDeviceToHost));
  	cont=false;
  	if(h_join_output_size < h_actual_found){
  	  // Not enough memory. Double memory footprint and try again
	  cont				  = true;
  	  h_join_output_size  = h_join_output_size*2;
  	}
  }

  // free memory used for the counters
  CUDA_TRY( cudaFree(d_global_write_index) );

  gdf_dtype dtype;
  switch(sizeof(output_index_type)) 
  {
    case 1 : dtype = GDF_INT8;  break;
    case 2 : dtype = GDF_INT16; break;
    case 4 : dtype = GDF_INT32; break;
    case 8 : dtype = GDF_INT64; break;
  }

  if (h_join_output_size > h_actual_found) {
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

  gdf_column_view(output_l, output_l_ptr, nullptr, h_actual_found, dtype);
  gdf_column_view(output_r, output_r_ptr, nullptr, h_actual_found, dtype);

  return error;
}
