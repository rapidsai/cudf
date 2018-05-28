/* Copyright 2018 NVIDIA Corporation.  All rights reserved. */

/* Header-only join C++ API (high-level) */

#include <limits>
#include <memory>

#include "hash-join/join_compute_api.h"
#include "sort-join.cuh"

using namespace mgpu;

// transpose
template<typename size_type, typename joined_type>
void pairs_to_decoupled(mem_t<size_type> &output, const size_type output_npairs, joined_type *joined, context_t &context, bool flip_indices)
{
  if (output_npairs > 0) {
    size_type* output_data = output.data();
    auto k = [=] MGPU_DEVICE(size_type index) {
      output_data[index] = flip_indices ? joined[index].second : joined[index].first;
      output_data[index + output_npairs] = flip_indices ? joined[index].first : joined[index].second;
    };
    transform(k, output_npairs, context);
  }
}

// two-table single-column join
template<JoinType join_type,
	 typename size_type,
	 typename a_it, typename b_it,
	 typename comp_t>
mem_t<size_type> join_hash(a_it a, size_type a_count,
				 b_it b, size_type b_count,
				 comp_t comp, context_t& context,
				 size_type estimated_join_count = 0, bool flip_indices = false)
{
  // here follows the custom code for hash-joins
  typedef join_pair<size_type> joined_type;

  if (join_type == INNER_JOIN) {
    // swap buffers if we're doing inner join & b_count > a_count to use the smaller table for build
    if (b_count > a_count)
      return join_hash<join_type>(b, b_count, a, a_count, comp, context, estimated_join_count, true);
  }

  // allocate a counter
  size_type *joined_idx;
  CUDA_RT_CALL( cudaMallocManaged(&joined_idx, sizeof(size_type)) );

  // TODO: here we don't know the output size so we'll start with some estimate and increase as necessary
  size_type joined_size = (size_type)(a_count);
  if (estimated_join_count > 0)
    joined_size = estimated_join_count;

  joined_type* joined = NULL;
  size_type output_npairs = 0;

  cudaError_t error;
  bool cont = true;
  while (cont) {
    // allocate an output buffer to store pairs, prefetch the estimated output size
    CUDA_RT_CALL( cudaMallocManaged(&joined, sizeof(joined_type) * joined_size) );
    CUDA_RT_CALL( cudaMemPrefetchAsync(joined, sizeof(joined_type) * joined_size, 0) ); // FIXME: use GPU device id from the context?

    // reset the counter
    CUDA_RT_CALL( cudaMemsetAsync(joined_idx, 0, sizeof(size_type), 0) );

    // using the new low-level API for hash-joins
    switch (join_type) {
    case INNER_JOIN: error = InnerJoinHash(context, (void**)&joined, joined_idx, joined_size, a, a_count, b, b_count); break;
    case LEFT_JOIN: error = LeftJoinHash(context, (void**)&joined, joined_idx, joined_size, a, a_count, b, b_count); break;
    }

    output_npairs = *joined_idx;
    if (error != cudaSuccess || output_npairs > joined_size) {
      cudaGetLastError();			// clear any errors
      CUDA_RT_CALL( cudaFree(joined) );		// free allocated memory
      joined_size *= 2; 			// simple heuristic, just double the size
    }
    else {
      cont = false; // found the right output size!
    }
  }

  // TODO: can we avoid this transformation?
  mem_t<size_type> output(2 * output_npairs, context);
  pairs_to_decoupled(output, output_npairs, joined, context, flip_indices);

  return output;
}

// two-table two-column inner join
template<typename size_type,
         typename a1_it, typename b1_it,
         typename a2_it, typename b2_it,
	 typename comp_t>
mem_t<size_type> inner_join_hash(a1_it a1, a2_it a2, size_type a_count,
                                 b1_it b1, b2_it b2, size_type b_count,
				 comp_t comp, context_t& context,
				 size_type estimated_join_count = 0, bool flip_indices = false)
{
  // here follows the custom code for hash-joins
  typedef join_pair<size_type> joined_type;

  // swap buffers if b_count > a_count to use the smaller table for build
  if (b_count > a_count)
    return inner_join_hash(b1, b2, b_count, a1, a2, a_count, comp, context, estimated_join_count, true);

  // allocate a counter
  size_type *joined_idx;
  CUDA_RT_CALL( cudaMallocManaged(&joined_idx, sizeof(size_type)) );

  // TODO: here we don't know the output size so we'll start with some estimate and increase as necessary
  size_type joined_size = (size_type)(a_count);
  if (estimated_join_count > 0)
    joined_size = estimated_join_count;

  joined_type* joined = NULL;
  size_type output_npairs = 0;

  bool cont = true;
  while (cont) {
    // allocate an output buffer to store pairs, prefetch the estimated output size
    CUDA_RT_CALL( cudaMallocManaged(&joined, sizeof(joined_type) * joined_size) );
    CUDA_RT_CALL( cudaMemPrefetchAsync(joined, sizeof(joined_type) * joined_size, 0) ); // FIXME: use GPU device id from the context?

    // reset the counter
    CUDA_RT_CALL( cudaMemsetAsync(joined_idx, 0, sizeof(size_type), 0) );

    // using the new low-level API for hash-joins
    cudaError_t error = InnerJoinHash(context, (void**)&joined, joined_idx, joined_size, a1, a_count, b1, b_count, a2, b2);

    output_npairs = *joined_idx;
    if (error != cudaSuccess || output_npairs > joined_size) {
      cudaGetLastError();                       // clear any errors
      CUDA_RT_CALL( cudaFree(joined) );         // free allocated memory
      joined_size *= 2;                         // simple heuristic, just double the size
    }
    else {
      cont = false; // found the right output size!
    }
  }

  // TODO: can we avoid this transformation?
  mem_t<size_type> output(2 * output_npairs, context);
  pairs_to_decoupled(output, output_npairs, joined, context, flip_indices);

  return output;
}

struct join_result_base {
  virtual ~join_result_base() {}
  virtual void* data() = 0;
  virtual size_t size() = 0;
};

template <typename T>
struct join_result : public join_result_base {
  standard_context_t context;
  mem_t<T> result;

  join_result() : context(false) {}
  virtual void* data() {
    return result.data();
  }
  virtual size_t size() {
    return result.size();
  }
};
