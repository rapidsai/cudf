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
#include <utility>

#include "cudf/functions.h"
#include "cudf/types.h"
#include "dataframe/cudf_table.cuh"

#include "sort_join.cuh"
#include "join_compute_api.h"

class rmm_mgpu_context_t; // forward decl

 /* --------------------------------------------------------------------------*/
 /**
  * @brief  Computes the hash-based join between two sets of gdf_tables.
  *
  * @param left_table The left table to be joined
  * @param right_table The right table to be joined
  * @param flip_indices Flag that indicates whether the left and right tables have been
  * flipped, meaning the output indices should also be flipped.
  * @tparam join_type The type of join to be performed
  * @tparam output_index_type The datatype used for the output indices
  *
  * @returns
  */
 /* ----------------------------------------------------------------------------*/
template<JoinType join_type,
         typename output_index_type,
         typename size_type>
gdf_error join_hash(gdf_table<size_type> const & left_table,
                    gdf_table<size_type> const & right_table,
                    gdf_column * const output_l,
                    gdf_column * const output_r,
                    bool flip_indices = false)
{

  // Hash table is built on the right table.
  // For inner joins, doesn't matter which table is build/probe, so we want
  // to build the hash table on the smaller table.
  if((join_type == JoinType::INNER_JOIN) &&
     (right_table.get_column_length() > left_table.get_column_length()))
  {
    return join_hash<join_type, output_index_type>(right_table, 
                                                   left_table, 
                                                   output_l, 
                                                   output_r, 
                                                   true);
  }

  return compute_hash_join<join_type, output_index_type>(output_l,
                                                         output_r, 
                                                         left_table, 
                                                         right_table, 
                                                         flip_indices);
}

// Overload Modern GPU memory allocation and free to use RMM
class rmm_mgpu_context_t : public mgpu::standard_context_t
{
public:
  rmm_mgpu_context_t(bool print_prop = true, cudaStream_t stream_ = 0) :
    mgpu::standard_context_t(print_prop, stream_) {}
  ~rmm_mgpu_context_t() {}

  virtual void* alloc(size_t size, memory_space_t space) {
    void *p = nullptr;
    if(size) {
      if (memory_space_device == space) {
        if (RMM_SUCCESS != RMM_ALLOC(&p, size, stream()))
          throw cuda_exception_t(cudaPeekAtLastError());
      }
      else {
        cudaError_t result = cudaMallocHost(&p, size);
        if (cudaSuccess != result) throw cuda_exception_t(result);
      }
    }
    return p;
  }

  virtual void free(void* p, memory_space_t space) {
    if (p) {
      if (memory_space_device == space) {
        if (RMM_SUCCESS != RMM_FREE(p, stream()))
          throw cuda_exception_t(cudaPeekAtLastError());
      }
      else {
        cudaError_t result = cudaFreeHost(&p);
        if (cudaSuccess != result) throw cuda_exception_t(result);
      }
    }
  }
};
