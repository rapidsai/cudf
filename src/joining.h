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
#include <memory>

#include "hash-join/join_compute_api.h"
#include "sort-join.cuh"

// transpose
/*
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
*/
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
	comp_t comp, mgpu::context_t& context)
{
  // here follows the custom code for hash-joins
  cudaError_t error = cudaSuccess;

  mgpu::mem_t<size_type> joined_output;
  // using the new low-level API for hash-join
  switch (join_type) {
	case INNER_JOIN: error = InnerJoinHash<INNER_JOIN>(context, joined_output, a, a_count, b, b_count, a2, b2, a3, b3); break;
	case LEFT_JOIN: error = LeftJoinHash<LEFT_JOIN>(context, joined_output, a, a_count, b, b_count, a2, b2, a3, b3); break;
  }

  return joined_output;
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
