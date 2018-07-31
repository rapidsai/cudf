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

// TODO for Arrow integration:
//   1) replace mgpu::context_t with a new CudaComputeContext class (see the design doc)
//   2) replace cudaError_t with arrow::Status
//   3) replace input iterators & input counts with arrow::Datum
//   3) replace output iterators & output counts with arrow::ArrayData

#include <moderngpu/context.hxx>

#include <moderngpu/kernel_scan.hxx>

constexpr int DEFAULT_HASH_TBL_OCCUPANCY = 50;
constexpr int DEFAULT_CUDA_BLOCK_SIZE = 128;
constexpr int DEFAULT_CUDA_CACHE_SIZE = 128;

template<typename size_type>
struct join_pair { size_type first, second; };

/// \brief Transforms the data from an array of structurs to two column.
///
/// \param[out] out An array with the indices of the common values. Stored in a 1D array with the indices of A appearing before those of B.
/// \param[in] Number of common values found)                                                                                      
/// \param[in] Common indices stored an in array of structure.
///
/// \param[in] compute_ctx The CudaComputeContext to shedule this to.
/// \param[in] Flag signifying if the order of the indices for A and B need to be swapped. This flag is used when the order of A and B are swapped to build the hash table for the smalle column.
template<typename size_type, typename joined_type>
void pairs_to_decoupled(mgpu::mem_t<size_type> &output, const size_type output_npairs, joined_type *joined, mgpu::context_t &context, bool flip_indices)
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


/// \brief Performs a generic hash based join of columns a and b. Works for both inner and left joins.
///
/// \param[in] compute_ctx The CudaComputeContext to shedule this to.
/// \param[out] out row references into a and b of matching rows
/// \param[in] a first column to join (left)
/// \param[in] Number of element in a column (left)
/// \param[in] b second column to join (right)
/// \param[in] Number of element in b column (right)
/// \param[in] additional columns to join (default == NULL)
/// \param[in] Flag used to reorder the left and right column indices found in the join (default = false)
template<JoinType join_type,
  typename input_it,
  typename input2_it,
  typename input3_it,
  typename size_type>
cudaError_t GenericJoinHash(mgpu::context_t &compute_ctx, mgpu::mem_t<size_type>& joined_output, 
	const input_it a, const size_type a_count, const input_it b, const size_type b_count,
	const input2_it a2 = (int*)NULL, const input2_it b2 = (int*)NULL,
	const input3_it a3 = (int*)NULL, const input3_it b3 = (int*)NULL,
	bool flip_results = false)
{
  cudaError_t error(cudaSuccess);

  typedef typename std::iterator_traits<input_it>::value_type key_type;
  typedef typename std::iterator_traits<input2_it>::value_type key_type2;
  typedef typename std::iterator_traits<input3_it>::value_type key_type3;
  typedef join_pair<size_type> joined_type;

  // allocate a counter and reset
  size_type *d_joined_idx;
  CUDA_RT_CALL( cudaMalloc(&d_joined_idx, sizeof(size_type)) );
  CUDA_RT_CALL( cudaMemsetAsync(d_joined_idx, 0, sizeof(size_type), 0) );

  // step 0: check if the output is provided or we need to allocate it

  // step 1: initialize a HT for table B (right)
#ifdef HT_LEGACY_ALLOCATOR
  typedef concurrent_unordered_multimap<key_type, size_type, std::numeric_limits<key_type>::max(), std::numeric_limits<size_type>::max(), default_hash<key_type>, equal_to<key_type>, legacy_allocator<thrust::pair<key_type, size_type> > > multimap_type;
#else
  typedef concurrent_unordered_multimap<key_type, size_type, std::numeric_limits<key_type>::max(), std::numeric_limits<size_type>::max()> multimap_type;
#endif
  size_type hash_tbl_size = (size_type)((size_t) b_count * 100 / DEFAULT_HASH_TBL_OCCUPANCY);
  std::unique_ptr<multimap_type> hash_tbl(new multimap_type(hash_tbl_size));
  hash_tbl->prefetch(0);  // FIXME: use GPU device id from the context? but moderngpu only provides cudaDeviceProp (although should be possible once we move to Arrow)
  error = cudaGetLastError();
  if (error != cudaSuccess)
	return error;

  // step 2: build the HT
  constexpr int block_size = DEFAULT_CUDA_BLOCK_SIZE;
  build_hash_tbl<<<(b_count + block_size-1) / block_size, block_size>>>(hash_tbl.get(), b, b_count);
  error = cudaGetLastError();
  if (error != cudaSuccess)
	return error;


  // step 3ab: scan table A (left), probe the HT without outputting the joined indices. Only get number of outputted elements.
  size_type* d_actualFound;
  cudaMalloc(&d_actualFound, sizeof(size_type));
  cudaMemset(d_actualFound, 0, sizeof(size_type));
  probe_hash_tbl_count_common<join_type, multimap_type, key_type, key_type2, key_type3, size_type, block_size, DEFAULT_CUDA_CACHE_SIZE>
	<<<(a_count + block_size-1) / block_size, block_size>>>
	(hash_tbl.get(), a, a_count, a2, b2, a3, b3,d_actualFound);
  if (error != cudaSuccess)
	return error;

  size_type scanSize=0;
  CUDA_RT_CALL( cudaMemcpy(&scanSize, d_actualFound, sizeof(size_type), cudaMemcpyDeviceToHost));

  int dev_ordinal;
  joined_type* tempOut=NULL;
  CUDA_RT_CALL( cudaGetDevice(&dev_ordinal));
  joined_output = mgpu::mem_t<size_type> (2 * (scanSize), compute_ctx);

  // Checking if any common elements exists. If not, then there is no point scanning again.
  if(scanSize==0){
	return error;
  }

  CUDA_RT_CALL( cudaMallocManaged   ( &tempOut, sizeof(joined_type)*scanSize));
  CUDA_RT_CALL( cudaMemPrefetchAsync( tempOut , sizeof(joined_type)*scanSize, dev_ordinal));

  CUDA_RT_CALL( cudaMemset(d_joined_idx, 0, sizeof(size_type)) );
  // step 3b: scan table A (left), probe the HT and output the joined indices - doing left join here
  probe_hash_tbl<join_type, multimap_type, key_type, key_type2, key_type3, size_type, joined_type, block_size, DEFAULT_CUDA_CACHE_SIZE>
	<<<(a_count + block_size-1) / block_size, block_size>>>
	(hash_tbl.get(), a, a_count, a2, b2, a3, b3,
	 static_cast<joined_type*>(tempOut), d_joined_idx, scanSize);
  error = cudaDeviceSynchronize();

  // free memory used for the counters
  CUDA_RT_CALL( cudaFree(d_joined_idx) );
  CUDA_RT_CALL( cudaFree(d_actualFound) ); 


  pairs_to_decoupled(joined_output, scanSize, tempOut, compute_ctx, flip_results);

  CUDA_RT_CALL( cudaFree(tempOut) );
  return error;
}


/// \brief Performs a hash based left join of columns a and b.
///
/// \param[in] compute_ctx The CudaComputeContext to shedule this to.
/// \param[out] out row references into a and b of matching rows
/// \param[in] a first column to join (left)
/// \param[in] Number of element in a column (left)
/// \param[in] b second column to join (right)
/// \param[in] Number of element in b column (right)
/// \param[in] additional columns to join (default == NULL)
template<typename input_it,
  typename input2_it,
  typename input3_it,
  typename size_type>
  cudaError_t LeftJoinHash(mgpu::context_t &compute_ctx, mgpu::mem_t<size_type>& joined_output, 
	  const input_it a, const size_type a_count, const input_it b, const size_type b_count,
	  const input2_it a2 = (int*)NULL, const input2_it b2 = (int*)NULL,
	  const input3_it a3 = (int*)NULL, const input3_it b3 = (int*)NULL){


	return GenericJoinHash<JoinType::LEFT_JOIN>(compute_ctx, joined_output, a, a_count, b, b_count, a2, b2, a3, b3);
  }


/// \brief Performs a hash based inner join of columns a and b.
///
/// \param[in] compute_ctx The CudaComputeContext to shedule this to.
/// \param[out] out row references into a and b of matching rows
/// \param[in] a first column to join (left)
/// \param[in] Number of element in a column (left)
/// \param[in] b second column to join (right)
/// \param[in] Number of element in b column (right)
/// \param[in] additional columns to join (default == NULL)
/// \param[in] Flag used to reorder the left and right column indices found in the join (default = false)
template<typename input_it,
  typename input2_it,
  typename input3_it,
  typename size_type>
  cudaError_t InnerJoinHash(mgpu::context_t &compute_ctx, mgpu::mem_t<size_type>& joined_output, 
	  const input_it a, const size_type a_count, const input_it b, const size_type b_count,
	  const input2_it a2 = (int*)NULL, const input2_it b2 = (int*)NULL,
	  const input3_it a3 = (int*)NULL, const input3_it b3 = (int*)NULL){

	if (b_count > a_count)
	  return GenericJoinHash<JoinType::INNER_JOIN>(compute_ctx, joined_output, b, b_count, a, a_count, b2, a2, b3, a3, true);
	else
	  return GenericJoinHash<JoinType::INNER_JOIN>(compute_ctx, joined_output, a, a_count, b, b_count, a2, b2, a3, b3);
  }
