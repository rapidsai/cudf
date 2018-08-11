/*
 * Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

 // transpose
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
 


/// \brief Performs a hash based inner join of columns a and b, stores only rows that have matching values in columns a2 and b2, etc.
/// 
/// \param[in] compute_ctx The CudaComputeContext to shedule this to.
/// \param[out] out row references into a and b of matching rows
/// \param[in] maximum size of the allocated output to avoid overflow
/// \param[in] a first column to join from table1 (larger table)
/// \param[in] b first column to join from table2 (smaller table)
/// \param[in] additional columns to join (default = NULL)
template<typename input_it,
	 typename input2_it,
	 typename input3_it,
	 typename size_type>
cudaError_t InnerJoinHash(mgpu::context_t &compute_ctx, void **out, size_type *out_count, 
			  const input_it a, const size_type a_count, const input_it b, const size_type b_count,
			  const input2_it a2 = (int*)NULL, const input2_it b2 = (int*)NULL,
			  const input3_it a3 = (int*)NULL, const input3_it b3 = (int*)NULL)
{
  cudaError_t error;

  typedef typename std::iterator_traits<input_it>::value_type key_type;
  typedef typename std::iterator_traits<input2_it>::value_type key_type2;
  typedef typename std::iterator_traits<input3_it>::value_type key_type3;
  typedef join_pair<size_type> joined_type;

  // step 0: check if the output is provided or we need to allocate it
  if (*out == NULL) return cudaErrorNotSupported;

  // step 1: initialize a HT for the smaller table b
  //TODO: swap a and b if b is larger than a
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
  
  // step 3: scan a, probe the HT, if matching also check columns a2 and b2, then output the results
  probe_hash_tbl<INNER_JOIN, multimap_type, key_type, key_type2, key_type3, size_type, joined_type, block_size, DEFAULT_CUDA_CACHE_SIZE>
                 <<<(a_count + block_size-1) / block_size, block_size>>>
                  (hash_tbl.get(), a, a_count, a2, b2, a3, b3,
		   static_cast<joined_type*>(*out), out_count, 0);
  printf("NEEED TO UPDATE 0 to scan size");
  error = cudaDeviceSynchronize();

  return error;
}

/// \brief Performs a hash based left join of columns a and b.
///
/// \param[in] compute_ctx The CudaComputeContext to shedule this to.
/// \param[out] out row references into a and b of matching rows
/// \param[in] maximum size of the allocated output to avoid overflow
/// \param[in] a first column to join (left)
/// \param[in] b second column to join (right)
/// \param[in] additional columns to join (default = NULL)
/*template<typename input_it,
	 typename input2_it,
	 typename input3_it,
	 typename size_type>
cudaError_t LeftJoinHash(mgpu::context_t &compute_ctx, void **out, size_type *out_count, 
                          const input_it a, const size_type a_count, const input_it b, const size_type b_count,
			  const input2_it a2 = (int*)NULL, const input2_it b2 = (int*)NULL,
			  const input3_it a3 = (int*)NULL, const input3_it b3 = (int*)NULL)
*/
template<JoinType join_type,
	 typename input_it,
	 typename input2_it,
	 typename input3_it,
	 typename size_type>
cudaError_t LeftJoinHash(mgpu::context_t &compute_ctx, mgpu::mem_t<size_type>& joined_output, 
                          const input_it a, const size_type a_count, const input_it b, const size_type b_count,
			  const input2_it a2 = (int*)NULL, const input2_it b2 = (int*)NULL,
			  const input3_it a3 = (int*)NULL, const input3_it b3 = (int*)NULL)
{
  cudaError_t error;

  typedef typename std::iterator_traits<input_it>::value_type key_type;
  typedef typename std::iterator_traits<input2_it>::value_type key_type2;
  typedef typename std::iterator_traits<input3_it>::value_type key_type3;
  typedef join_pair<size_type> joined_type;
  
  // allocate a counter and reset
  size_type *d_joined_idx;
  CUDA_RT_CALL( cudaMalloc(&d_joined_idx, sizeof(size_type)) );
  CUDA_RT_CALL( cudaMemsetAsync(d_joined_idx, 0, sizeof(size_type), 0) );
   
  // step 0: check if the output is provided or we need to allocate it
  //if (*out == NULL) return cudaErrorNotSupported;

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
  probe_hash_tbl_no_add<join_type, multimap_type, key_type, key_type2, key_type3, size_type, block_size, DEFAULT_CUDA_CACHE_SIZE>
                 <<<(a_count + block_size-1) / block_size, block_size>>>
                  (hash_tbl.get(), a, a_count, a2, b2, a3, b3,d_actualFound);
   if (error != cudaSuccess)
    return error;

  size_type scanSize=0;
  CUDA_RT_CALL( cudaMemcpy(&scanSize, d_actualFound, sizeof(size_type), cudaMemcpyDeviceToHost));
  printf("Post scan : %d %d\n", scanSize, join_type);
  
  int dev_ordinal;
  joined_type* tempOut=NULL;
  CUDA_RT_CALL( cudaGetDevice(&dev_ordinal));
  CUDA_RT_CALL( cudaMallocManaged   ( &tempOut, sizeof(joined_type)*scanSize));
  CUDA_RT_CALL( cudaMemPrefetchAsync( tempOut , sizeof(joined_type)*scanSize, dev_ordinal));
  
  // copy the counter to the cpu
  //CUDA_RT_CALL( cudaMemcpy(out_count, d_joined_idx, sizeof(size_type), cudaMemcpyDefault) );
                                                          
  CUDA_RT_CALL( cudaMemset(d_joined_idx, 0, sizeof(size_type)) );
  // step 3b: scan table A (left), probe the HT and output the joined indices - doing left join here
/*  probe_hash_tbl<join_type, multimap_type, key_type, key_type2, key_type3, size_type, joined_type, block_size, DEFAULT_CUDA_CACHE_SIZE>
                 <<<(a_count + block_size-1) / block_size, block_size>>>
                  (hash_tbl.get(), a, a_count, a2, b2, a3, b3,
		   static_cast<joined_type*>(tempOut), d_joined_idx, scanSize);
  error = cudaDeviceSynchronize();
*/
  //cudaFree(d_actualFound);
   // free memory used for the counters
  CUDA_RT_CALL( cudaFree(d_joined_idx) );
  CUDA_RT_CALL( cudaFree(d_actualFound) ); 

  joined_output = mgpu::mem_t<size_type> (2 * (scanSize), compute_ctx);

  printf("Flip indices");
  pairs_to_decoupled(joined_output, scanSize, tempOut, compute_ctx, false);
 
  CUDA_RT_CALL( cudaFree(tempOut) );
  printf ("\nSCAN: %d %d\n", joined_output.size(),scanSize);
  return error;
}


