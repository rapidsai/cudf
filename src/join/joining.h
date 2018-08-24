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
#include "../gdf_table.cuh"

 /* --------------------------------------------------------------------------*/
 /** 
  * @Synopsis  Computes the hash-based join between two sets of gdf_tables.
  * 
  * @Param left_table The left table to be joined
  * @Param right_table The right table to be joined
  * @Param context Modern GPU context
  * @Param flip_indices Flag that indicates whether the left and right tables have been
  * flipped, meaning the output indices should also be flipped.
  * @tparam join_type The type of join to be performed
  * @tparam output_type The datatype used for the output indices
  * 
  * @Returns   
  */
 /* ----------------------------------------------------------------------------*/
template<JoinType join_type, 
         typename output_type,
         typename size_type>
mgpu::mem_t<output_type> join_hash(gdf_table<size_type> const & left_table, 
                                   gdf_table<size_type> const & right_table, 
                                   mgpu::context_t & context,
                                   bool flip_indices = false) 
{

  // Hash table is built on the right table. 
  // For inner joins, doesn't matter which table is build/probe, so we want 
  // to build the hash table on the smaller table.
  if((join_type == JoinType::INNER_JOIN) && 
     (right_table.get_column_length() > left_table.get_column_length()))
  {
    return join_hash<join_type, output_type>(right_table, left_table, context, true);
  }


  mgpu::mem_t<output_type> joined_output;

  const gdf_dtype key_type = left_table.get_build_column_type();

  switch(key_type)
  {
    case GDF_INT8:    
      {
        compute_hash_join<join_type, int8_t, output_type>(context, joined_output, left_table, right_table, flip_indices); 
        break;
      }
    case GDF_INT16:   
      {
        compute_hash_join<join_type, int16_t, output_type>(context, joined_output, left_table, right_table, flip_indices); 
        break;
      }
    case GDF_INT32:   
      {
        compute_hash_join<join_type, int32_t, output_type>(context, joined_output, left_table, right_table, flip_indices); 
        break;
      }
    case GDF_INT64:   
      {
        compute_hash_join<join_type, int64_t, output_type>(context, joined_output, left_table, right_table, flip_indices);                    
        break;
      }
    // For floating point types build column, treat as an integral type
    case GDF_FLOAT32: 
      {
        compute_hash_join<join_type, int32_t, output_type>(context, joined_output, left_table, right_table, flip_indices);
        break;
      }
    case GDF_FLOAT64: 
      {
        compute_hash_join<join_type, int64_t, output_type>(context, joined_output, left_table, right_table, flip_indices);
        break;
      }
    case GDF_DATE32:   
      {
        compute_hash_join<join_type, int32_t, output_type>(context, joined_output, left_table, right_table, flip_indices); 
        break;
      }
    case GDF_DATE64:   
      {
        compute_hash_join<join_type, int64_t, output_type>(context, joined_output, left_table, right_table, flip_indices);                    
        break;
      }
    case GDF_TIMESTAMP:   
      {
        compute_hash_join<join_type, int64_t, output_type>(context, joined_output, left_table, right_table, flip_indices);                    
        break;
      }
    default:
      assert(false && "Invalid build column datatype.");
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
