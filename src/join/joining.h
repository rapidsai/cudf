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

#include <gdf/cffi/functions.h>
#include <gdf/cffi/types.h>

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
  * @tparam output_index_type The datatype used for the output indices
  *
  * @Returns
  */
 /* ----------------------------------------------------------------------------*/
template<JoinType join_type,
         typename output_index_type,
         typename size_type>
gdf_error join_hash(gdf_table<size_type> const & left_table,
                    gdf_table<size_type> const & right_table,
                    mgpu::context_t & context,
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
                                                   context, 
                                                   output_l, 
                                                   output_r, 
                                                   true);
  }

  return compute_hash_join<join_type, output_index_type>(context, 
                                                         output_l, 
                                                         output_r, 
                                                         left_table, 
                                                         right_table, 
                                                         flip_indices);
}

