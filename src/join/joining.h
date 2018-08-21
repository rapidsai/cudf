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

/* Header-only join C++ API (high-level) */

#include <limits>
#include <memory>

#include "hash/join_compute_api.h"
#include "sort/sort-join.cuh"

// N-column join (N up to 3 currently)
// \brief Performs a hash based join of columns a and b.
///
/// \param[in] a first column to join (left)
/// \param[in] Number of element in a column (left)
/// \param[in] b second column to join (right)
/// \param[in] Number of element in b column (right)
/// \param[in] additional columns to join (default == NULL)
/// \param[in] Flag used to reorder the left and right column indices found in the join (default = false)
/// \param[in] compute_ctx The CudaComputeContext to shedule this to.
/// \return	   Array of matching rows
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

  mgpu::mem_t<size_type> joined_output;
  // using the new low-level API for hash-join
  switch (join_type) {
      case JoinType::INNER_JOIN: InnerJoinHash(context, joined_output, a, a_count, b, b_count, a2, b2, a3, b3); break;
      case JoinType::LEFT_JOIN: LeftJoinHash(context, joined_output, a, a_count, b, b_count, a2, b2, a3, b3); break;
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
