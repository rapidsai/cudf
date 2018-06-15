/* Copyright 2018 NVIDIA Corporation.  All rights reserved. */
#include <cuda_runtime.h>
#include <future>

#include "join_kernels.cuh"

// TODO for Arrow integration:
//   1) replace mgpu::context_t with a new CudaComputeContext class (see the design doc)
//   2) replace cudaError_t with arrow::Status
//   3) replace input iterators & input counts with arrow::Datum
//   3) replace output iterators & output counts with arrow::ArrayData

#include <moderngpu/context.hxx>

using namespace mgpu;

#define DEFAULT_HASH_TBL_OCCUPANCY	50
#define DEFAULT_CUDA_BLOCK_SIZE		128
#define DEFAULT_CUDA_CACHE_SIZE		128

// TODO: thrust::pair<size_type, size_type> would be better here, but doesn't work correctly
//       since the constructor gets invoked for shared memory cache and creates race condition
//       thus I replaced thrust::pair with a bare pair struct that works just fine
template<typename size_type>
struct join_pair { size_type first, second; };

template <typename T>
class cuda_future : public std::shared_future<T> {
  public:
    cuda_future(cudaEvent_t event);
    cuda_future(cudaStream_t stream);
    
    /// \brief Returns the event associated with this cuda_future.
    cudaEvent_t getEvent();
    
    /// \brief Returns the stream associated with this cuda_future.
    cudaStream_t getStream();
    
    /// \brief Blocks until all GPU work in stream or until event is done. 
    //Problem: wait is not virutal in std::shared_future. How to handle this?
    virtual void wait() const;
};

struct HashTbl_t {

};

/// \brief Creates an empty hash table with the given capacity.
cudaError_t CreateHashTbl(context_t &compute_ctx, std::shared_ptr<HashTbl_t>* out, int64_t capacity);

/// \brief Fills the passed in HashTbl from the passed in Datum variant.
///
/// Asynchronous operation. If the capacity of the passed in HashTbl is sufficient no memory allocation is performed.
template<typename input_it>
cudaError_t BuildHashTblAsync(context_t &compute_ctx, std::shared_ptr<std::shared_future<void>>* done, std::shared_ptr<HashTbl_t> out, const input_it& in, std::shared_future<void>* dependency = 0)
{
    cudaStream_t stream = compute_ctx.stream();
    cuda_future<void>* cuda_dependency = ( 0 != dependency ) ? static_cast<cuda_future<void>*>(dependency) : 0;
    if ( 0 != cuda_dependency && cuda_dependency->valid() )
    {
        if ( 0 != cuda_dependency->getStream() )
        {
            stream = cuda_dependency->getStream();
        }
        else if ( 0 != cuda_dependency->getEvent() )
        {
            CUDA_RT_CALL( cudaStreamWaitEvent( stream, cuda_dependency->getEvent(), 0 ) );
        }
    }
    else if ( 0 != dependency && dependency->valid() )
    {
        dependency->wait();
    }
    
    //...
    
    cudaEvent_t event = compute_ctx.event();
    CUDA_RT_CALL( cudaEventRecord( event, stream ) );
    *done = std::make_shared<cuda_future<void> >(event);
    return cudaSuccess;
}

/// \brief Probes the passed in Column or Array vs. the hash table.
///
/// Optionally uses a multi pass approach to work around TLB issues as described in Karnagel, Tomas, et al.
/// "Big data causing big (TLB) problems: taming random memory accesses on the GPU."
/// Proceedings of the 13th International Workshop on Data Management on New Hardware. ACM, 2017.
/// \param[in] compute_ctx The CudaComputeContext to shedule this to.
/// \param[out] done A future that can be used to wait for this asynchronous operation to finish.
/// \param[out] out row references into the build and the probe table
/// \param[in] ht hash table to probe
/// \param[in] in values to probe vs. the hash table
/// \param[in] dependency optional dependency to wait for before this operation is started.
/// \return Status
/// Asynchronous operation. If the capacity of the passed in ArrayData is sufficient no memory allocation is performed.
template<typename input_it, typename output_it>
cudaError_t ProbeHashTblAsync(context_t &compute_ctx, std::shared_ptr<std::shared_future<void>>* done, std::shared_ptr<output_it>* out, const HashTbl_t& ht, const input_it& in, std::shared_future<void>* dependency = 0);

/// \brief Performs an async hash based inner join of columns a and b.
///
/// Memory is only allocated if the capacity of any passed in data structure is not sufficient. Asynchronous operation.
/// \param[in] compute_ctx The CudaComputeContext to shedule this to.
/// \param[out] done A future that can be used to wait for this asynchronous operation to finish.
/// \param[out] out row references into a and b of matching rows
/// \param[in,out] ht preallocated hash table to use
/// \param[in] a first column to join
/// \param[in] b second column to join
/// \param[in] dependency optional dependency to wait for before this operation is started.
template<typename input1_it, typename input2_it, typename output_it>
cudaError_t InnerJoinHashAsync(context_t &compute_ctx, std::shared_ptr<std::shared_future<void>>* done, std::shared_ptr<output_it>* out, std::shared_ptr<HashTbl_t> ht, const input1_it& a, const input2_it& b, std::shared_future<void>* dependency = 0);

/// \brief Performs a hash based inner join of columns a and b, stores only rows that have matching values in columns a2 and b2, etc.
/// 
/// It is possible to pass in preallocated ArrayData, in which case new memory via the MemoryPool associated
/// with handle is only allocated if the capacity of the passed in ArrayData container is not sufficient.
/// \param[in] compute_ctx The CudaComputeContext to shedule this to.
/// \param[out] out row references into a and b of matching rows
/// \param[in] maximum size of the allocated output to avoid overflow
/// \param[in] a first column to join from table1 (larger table)
/// \param[in] b first column to join from table2 (smaller table)
/// \param[in] dependency optional dependency to wait for before this operation is started.
/// \param[in] additional columns to join (default = NULL)
template<typename input_it,
	 typename input2_it,
	 typename input3_it,
	 typename size_type>
cudaError_t InnerJoinHash(context_t &compute_ctx, void **out, size_type *out_count, const size_type max_out_count,
			  const input_it a, const size_type a_count, const input_it b, const size_type b_count,
			  std::shared_future<void>* dependency = 0,
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
  size_type hash_tbl_size = (size_type)(b_count * 100 / DEFAULT_HASH_TBL_OCCUPANCY);
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
		   static_cast<joined_type*>(*out), out_count, max_out_count);
  error = cudaDeviceSynchronize();

  return error;
}

/// \brief Performs a hash based left join of columns a and b.
///
/// It is possible to pass in preallocated ArrayData, in which case new memory via the MemoryPool associated
/// with handle is only allocated if the capacity of the passed in ArrayData container is not sufficient.
/// \param[in] compute_ctx The CudaComputeContext to shedule this to.
/// \param[out] out row references into a and b of matching rows
/// \param[in] maximum size of the allocated output to avoid overflow
/// \param[in] a first column to join (left)
/// \param[in] b second column to join (right)
/// \param[in] dependency optional dependency to wait for before this operation is started.
/// \param[in] additional columns to join (default = NULL)
template<typename input_it,
	 typename input2_it,
	 typename input3_it,
	 typename size_type>
cudaError_t LeftJoinHash(context_t &compute_ctx, void **out, size_type *out_count, const size_type max_out_count,
                          const input_it a, const size_type a_count, const input_it b, const size_type b_count,
                          std::shared_future<void>* dependency = 0,
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

  // step 1: initialize a HT for table B (right)
#ifdef HT_LEGACY_ALLOCATOR
  typedef concurrent_unordered_multimap<key_type, size_type, std::numeric_limits<key_type>::max(), std::numeric_limits<size_type>::max(), default_hash<key_type>, equal_to<key_type>, legacy_allocator<thrust::pair<key_type, size_type> > > multimap_type;
#else
  typedef concurrent_unordered_multimap<key_type, size_type, std::numeric_limits<key_type>::max(), std::numeric_limits<size_type>::max()> multimap_type;
#endif
  size_type hash_tbl_size = (size_type)(b_count * 100 / DEFAULT_HASH_TBL_OCCUPANCY);
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

  // step 3: scan table A (left), probe the HT and output the joined indices - doing left join here
  probe_hash_tbl<LEFT_JOIN, multimap_type, key_type, key_type2, key_type3, size_type, joined_type, block_size, DEFAULT_CUDA_CACHE_SIZE>
                 <<<(a_count + block_size-1) / block_size, block_size>>>
                  (hash_tbl.get(), a, a_count, a2, b2, a3, b3,
		   static_cast<joined_type*>(*out), out_count, max_out_count);
  error = cudaDeviceSynchronize();

  return error;
}
