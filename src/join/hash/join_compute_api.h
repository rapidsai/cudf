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

#include "join_kernels.cuh"
#include "../../gdf_table.cuh"

// TODO for Arrow integration:
//   1) replace mgpu::context_t with a new CudaComputeContext class (see the design doc)
//   2) replace cudaError_t with arrow::Status
//   3) replace input iterators & input counts with arrow::Datum
//   3) replace output iterators & output counts with arrow::ArrayData

#include <moderngpu/context.hxx>

#include <moderngpu/kernel_scan.hxx>

constexpr int DEFAULT_HASH_TABLE_OCCUPANCY = 50;
constexpr int DEFAULT_CUDA_BLOCK_SIZE = 128;
constexpr int DEFAULT_CUDA_CACHE_SIZE = 128;

template<typename size_type>
struct join_pair
{
  size_type first;
  size_type second;
};

/// \brief Transforms the data from an array of structurs to two column.
///
/// \param[out] out An array with the indices of the common values. Stored in a 1D array with the indices of A appearing before those of B.
/// \param[in] Number of common values found)
/// \param[in] Common indices stored an in array of structure.
///
/// \param[in] compute_ctx The CudaComputeContext to shedule this to.
/// \param[in] Flag signifying if the order of the indices for A and B need to be swapped. This flag is used when the order of A and B are swapped to build the hash table for the smalle column.
template<typename size_type, typename join_output_pair>
void pairs_to_decoupled(mgpu::mem_t<size_type> &output, const size_type output_npairs, join_output_pair *joined, mgpu::context_t &context, bool flip_indices)
{
  if (output_npairs > 0) {
    size_type* output_data = output.data();
    auto k = [=] MGPU_DEVICE(size_type index) {
      output_data[index] = flip_indices ? joined[index].second : joined[index].first;
      output_data[index + output_npairs] = flip_indices ? joined[index].first : joined[index].second;
    };
    mgpu::transform(k, output_npairs, context);
  }
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
* @tparam key_type The data type to be used for the Keys in the hash table
* @tparam index_type The data type to be used for the output indices
*
* @Returns  cudaSuccess upon successful completion of the join. Otherwise returns
* the appropriate CUDA error code
*/
/* ----------------------------------------------------------------------------*/
template<JoinType join_type,
         typename key_type,
         typename index_type,
         typename size_type>
cudaError_t compute_hash_join(mgpu::context_t & compute_ctx,
                              mgpu::mem_t<index_type> & joined_output,
                              gdf_table<size_type> const & left_table,
                              gdf_table<size_type> const & right_table,
                              bool flip_results = false)
{
  cudaError_t error(cudaSuccess);

  // Data type used for join output results. Stored as a pair of indices
  // (left index, right index) where left_table[left index] == right_table[right index]
  using join_output_pair = join_pair<index_type>;

  // The LEGACY allocator allocates the hash table array with normal cudaMalloc,
  // the non-legacy allocator uses managed memory
#ifdef HT_LEGACY_ALLOCATOR
  using multimap_type = concurrent_unordered_multimap<key_type,
                                                      index_type,
                                                      size_type,
                                                      std::numeric_limits<key_type>::max(),
                                                      std::numeric_limits<index_type>::max(),
                                                      default_hash<key_type>,
                                                      equal_to<key_type>,
                                                      legacy_allocator< thrust::pair<key_type, index_type> > >;
#else
  using multimap_type = concurrent_unordered_multimap<key_type,
                                                      index_type,
                                                      size_type,
                                                      std::numeric_limits<key_type>::max(),
                                                      std::numeric_limits<size_type>::max()>;
#endif

  // Hash table will be built on the right table
  gdf_table<size_type> const & build_table{right_table};
  const size_type build_column_length{build_table.get_column_length()};
  const key_type * const build_column{static_cast<key_type*>(build_table.get_build_column_data())};

  // Allocate the hash table
  const size_type hash_table_size = (static_cast<size_type>(build_column_length) * 100 / DEFAULT_HASH_TABLE_OCCUPANCY);
  std::unique_ptr<multimap_type> hash_table(new multimap_type(hash_table_size));

  // FIXME: use GPU device id from the context?
  // but moderngpu only provides cudaDeviceProp
  // (although should be possible once we move to Arrow)
  hash_table->prefetch(0);

  CUDA_RT_CALL( cudaDeviceSynchronize() );

  // build the hash table
  constexpr int block_size = DEFAULT_CUDA_BLOCK_SIZE;
  const size_type build_grid_size{(build_column_length + block_size - 1)/block_size};
  build_hash_table<<<build_grid_size, block_size>>>(hash_table.get(),
                                                    build_column,
                                                    build_column_length);

  CUDA_RT_CALL( cudaGetLastError() );

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

  cudaMalloc(&d_join_output_size, sizeof(size_type));
  cudaMemset(d_join_output_size, 0, sizeof(size_type));

  // Probe with the left table
  gdf_table<size_type> const & probe_table{left_table};
  const key_type * const probe_column{static_cast<key_type*>(probe_table.get_probe_column_data())};
  //const size_type probe_grid_size{(probe_column_length + block_size -1)/block_size};

  CUDA_RT_CALL( cudaGetLastError() );

  // A situation can arise such that the number of elements found in the probing phase is equal to zero. This would lead us to approximating
  // the number of joined elements to be zero. As such we need to increase the subset and continue probing to get a bettter approximation value.
  do{
  	if(leftSampleSize>leftSize)
  	  leftSampleSize=leftSize;
  	// step 3ab: scan table A (left), probe the HT without outputting the joined indices. Only get number of outputted elements.
    cudaMemset(d_join_output_size, 0, sizeof(size_type));

	const size_type probe_grid_size{(leftSampleSize + block_size -1)/block_size};
    // Probe the hash table without actually building the output to simply
    // find what the size of the output will be.
    compute_join_output_size<join_type,
                             multimap_type,
                             key_type,
                             size_type,
                             block_size,
                             DEFAULT_CUDA_CACHE_SIZE>
  	<<<probe_grid_size, block_size>>>(hash_table.get(),
                                      build_table,
                                      probe_table,
                                      probe_column,
                                      leftSampleSize,
                                      d_join_output_size);

  	if (error != cudaSuccess)
  	  return error;

    CUDA_RT_CALL( cudaMemcpy(&h_join_output_size, d_join_output_size, sizeof(size_type), cudaMemcpyDeviceToHost));
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

  CUDA_RT_CALL( cudaFree(d_join_output_size) );

  // If the output size is zero, return immediately
  if(0 == h_join_output_size){
    return error;
  }

  // As we are now approximating the number of joined elements, our approximation might be incorrect and we might have underestimated the
  // number of joined elements. As such we will need to de-allocate memory and re-allocate memory to ensure that the final output is correct.
  size_type h_actual_found;
  join_output_pair* tempOut=NULL;
  bool cont = true;

  // Allocate device global counter used by threads to determine output write location
  size_type *d_global_write_index{nullptr};
  CUDA_RT_CALL( cudaMalloc(&d_global_write_index, sizeof(size_type)) );
  int dev_ordinal{0};
  CUDA_RT_CALL( cudaGetDevice(&dev_ordinal));
 
  while(cont){
  	tempOut=NULL;
  	CUDA_RT_CALL( cudaGetDevice(&dev_ordinal));

    // Allocate temporary device buffer for join output
    CUDA_RT_CALL( cudaMallocManaged   ( &tempOut, sizeof(join_output_pair)*h_join_output_size));
    CUDA_RT_CALL( cudaMemPrefetchAsync( tempOut , sizeof(join_output_pair)*h_join_output_size, dev_ordinal));
    CUDA_RT_CALL( cudaMemsetAsync(d_global_write_index, 0, sizeof(size_type), 0) );

	const size_type probe_grid_size{(leftSize + block_size -1)/block_size};
    // Do the probe of the hash table with the probe table and generate the output for the join
	probe_hash_table<join_type,
					 multimap_type,
                     key_type,
                     size_type,
                     join_output_pair,
                     block_size,
                     DEFAULT_CUDA_CACHE_SIZE>
  	<<<probe_grid_size, block_size>>> (hash_table.get(),
                                       build_table,
                                       probe_table,
									   probe_column,
                                       probe_table.get_column_length(),
                                       static_cast<join_output_pair*>(tempOut),
                                       d_global_write_index,
                                       h_join_output_size);

    CUDA_RT_CALL(cudaDeviceSynchronize());

  	CUDA_RT_CALL( cudaMemcpy(&h_actual_found, d_global_write_index, sizeof(size_type), cudaMemcpyDeviceToHost));
  	cont=false;
  	if(h_join_output_size < h_actual_found){
  	  // Not enough memory. Double memory footprint and try again
	  cont				  = true;
  	  h_join_output_size  = h_join_output_size*2;
  	  CUDA_RT_CALL( cudaFree(tempOut) );
  	}
  }

  // Allocate modern GPU storage for the join output
  joined_output = mgpu::mem_t<size_type> (2 * (h_actual_found), compute_ctx);

  // Transform the join output from an array of pairs, to an array of indices where the first
  // n/2 elements are the left indices and the last n/2 elements are the right indices
  pairs_to_decoupled(joined_output, h_actual_found, tempOut, compute_ctx, flip_results);

  // free memory used for the counters
  CUDA_RT_CALL( cudaFree(d_global_write_index) );

  // Free temporary device buffer
  CUDA_RT_CALL( cudaFree(tempOut) );

  return error;
}


