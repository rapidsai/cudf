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

/* Header-only join C++ API (high-level) */

#include <limits>
#include <iostream>
#include <memory>

#include "hash-join/join_compute_api.h"
#include "sort-join.cuh"

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

// N-column join (N up to 3 currently)
template<JoinType join_type,
	 typename size_type,
	 typename col1_it,
	 typename col2_it,
	 typename col3_it,
	 typename comp_t>
mgpu::mem_t<size_type> join_hash(col1_it a, size_type a_count,
			         col1_it b, size_type b_count,
			         col2_it a2, col2_it b2,
			         col3_it a3, col3_it b3,
			         comp_t comp, mgpu::context_t& context,
			         size_type estimated_join_count = 0, bool flip_indices = false)
{
  // here follows the custom code for hash-joins
  typedef join_pair<size_type> joined_type;

  if (join_type == INNER_JOIN) {
    // swap buffers if we're doing inner join & b_count > a_count to use the smaller table for build
    if (b_count > a_count)
      return join_hash<join_type>(b, b_count, a, a_count, b2, a2, b3, a3, comp, context, estimated_join_count, true);
  }

  // get device id
  int dev_ordinal;
  CUDA_RT_CALL( cudaGetDevice(&dev_ordinal) );

  // allocate a counter
  size_type *d_joined_idx, *h_joined_idx;
  CUDA_RT_CALL( cudaMalloc(&d_joined_idx, sizeof(size_type)) );
  CUDA_RT_CALL( cudaMallocHost(&h_joined_idx, sizeof(size_type)) );

  // TODO: here we don't know the output size so we'll start with some estimate and increase as necessary
  size_type joined_size = (size_type)(a_count);
  if (estimated_join_count > 0)
    joined_size = estimated_join_count;

  // output buffer
  joined_type* joined = NULL;

  cudaError_t error;
  bool cont = true;
  while (cont) {
    // allocate an output buffer to store pairs, prefetch the estimated output size
    CUDA_RT_CALL( cudaMallocManaged(&joined, sizeof(joined_type) * joined_size) );
    CUDA_RT_CALL( cudaMemPrefetchAsync(joined, sizeof(joined_type) * joined_size, dev_ordinal) );

    // reset the counter
    CUDA_RT_CALL( cudaMemsetAsync(d_joined_idx, 0, sizeof(size_type), 0) );

    // using the new low-level API for hash-joins
    switch (join_type) {
    case INNER_JOIN: error = InnerJoinHash(context, (void**)&joined, d_joined_idx, joined_size, a, a_count, b, b_count, a2, b2, a3, b3); break;
    case LEFT_JOIN: error = LeftJoinHash(context, (void**)&joined, d_joined_idx, joined_size, a, a_count, b, b_count, a2, b2, a3, b3); break;
    }

    // copy the counter to the cpu
    CUDA_RT_CALL( cudaMemcpy(h_joined_idx, d_joined_idx, sizeof(size_type), cudaMemcpyDefault) );

    if (error != cudaSuccess || (*h_joined_idx) > joined_size) {
      cudaGetLastError();			// clear any errors
      CUDA_RT_CALL( cudaFree(joined) );		// free allocated memory
      joined_size *= 2; 			// simple heuristic, just double the size
    }
    else {
      cont = false; // found the right output size!
    }
  }

  // TODO: can we avoid this transformation?
  mgpu::mem_t<size_type> output(2 * (*h_joined_idx), context);
  pairs_to_decoupled(output, (*h_joined_idx), joined, context, flip_indices);

  // free memory used for the counters
  CUDA_RT_CALL( cudaFree(d_joined_idx) );
  CUDA_RT_CALL( cudaFreeHost(h_joined_idx) );

  // free memory used for the join output
  CUDA_RT_CALL( cudaFree(joined) );

  return output;
}

struct join_result_base {
  virtual ~join_result_base() {}
  virtual void* data() = 0;
  virtual size_t size() = 0;
};

template <typename T>
struct join_result : public join_result_base {
  mgpu::standard_context_t context;
  mgpu::mem_t<T> result;

  join_result() : context(false) {}
  virtual void* data() {
    return result.data();
  }
  virtual size_t size() {
    return result.size();
  }
};
