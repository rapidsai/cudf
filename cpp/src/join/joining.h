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
#include <table/device_table.cuh>

#include "sort_join.cuh"
#include "join_compute_api.h"

 /* --------------------------------------------------------------------------*/
 /**
  * @brief  Computes the hash-based join between two sets of device_tables.
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
         typename output_index_type>
gdf_error join_hash(cudf::table const & left_table,
                    cudf::table const & right_table,
                    gdf_column * const output_l,
                    gdf_column * const output_r,
                    bool flip_indices = false)
{

  // Hash table is built on the right table.
  // For inner joins, doesn't matter which table is build/probe, so we want
  // to build the hash table on the smaller table.
  if((join_type == JoinType::INNER_JOIN) &&
     (right_table.num_rows() > left_table.num_rows()))
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

 /* --------------------------------------------------------------------------*/
 /**
  * @Synopsis  Computes the sort-based join between two columns.
  *
  * @Param leftcol The left column to be joined
  * @Param rightcol The right column to be joined
  * @Param output_l The left index output of join
  * @Param output_r The right index output of join
  * @Param flip_indices Flag that indicates whether the left and right tables have been
  * flipped, meaning the output indices should also be flipped.
  * @tparam join_type The type of join to be performed
  * @tparam output_index_type The datatype used for the output indices
  *
  * @Returns
  */
 /* ----------------------------------------------------------------------------*/
template <JoinType join_type, 
          typename output_index_type>
gdf_error sort_join(gdf_column *leftcol, gdf_column *rightcol,
                    gdf_column * const output_l,
                    gdf_column * const output_r,
                    bool flip_indices = false)
{
  if ((leftcol->null_count != 0) || (rightcol->null_count != 0)) {
      return GDF_VALIDITY_UNSUPPORTED;
  }

  compute_sort_join<join_type, output_index_type> join_call;
  return cudf::type_dispatcher(
          leftcol->dtype,
          join_call,
          output_l, output_r,
          leftcol, rightcol,
          flip_indices);
}
