/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include "copy_if.cuh"
#include <cudf/table.hpp>
 
namespace {

// Returns true if the valid mask is true for index i
// Note we use a functor here so we can cast to a bitmask_t __restrict__
// pointer on the host side, which we can't do with a lambda.
struct valid_column_filter
{
  valid_column_filter(gdf_column const & column) :
    size{column.size},
    bitmask{reinterpret_cast<bit_mask_t *>(column.valid)}
    { CUDF_EXPECTS(nullptr != column.valid, "Null valid bitmask");}

  __device__ inline 
  bool operator()(gdf_index_type i)
  {
    if (i < size) {
      bool valid = bit_mask::is_valid(bitmask, i);
      return valid;
    }
    return false;
  }

  gdf_size_type size;
  bit_mask_t const  * __restrict__ bitmask;
};


// Returns true if the valid mask is true for index i in all columns of the table
// Note we use a functor here so we can cast to bitmask_t __restrict__ *
// pointer on the host side, which we can't do with a lambda.
struct valid_table_filter
{
  valid_table_filter(cudf::table const & table,
                     bit_mask_t **masks,
                     cudf::any_or_all drop_if) 
  : num_rows(table.num_rows()),
    num_columns(table.num_columns()),
    d_masks(masks),
    drop_if(drop_if) {}

  __device__ inline 
  bool operator()(gdf_index_type i)
  {
    if (i < num_rows) {
      
      int c = 0;
      if (drop_if == cudf::ALL) {
        while (c < num_columns)
          if (bit_mask::is_valid(d_masks[c++], i)) return true;
        return false;
      }
      else { // drop_if == cudf::ANY => all columns must be valid
        while (c < num_columns)
          if (not bit_mask::is_valid(d_masks[c++], i)) return false;
        return true;
      }
    }
    return false;
  }

  cudf::any_or_all drop_if;
  gdf_size_type num_rows;
  gdf_size_type num_columns;
  bit_mask_t **d_masks;

};

bit_mask_t** get_bitmasks(cudf::table const &table,
                          cudaStream_t stream = 0) {
  bit_mask_t** h_masks = new bit_mask_t*[table.num_columns()];
  
  for (int i = 0; i < table.num_columns(); ++i) {
    h_masks[i] = reinterpret_cast<bit_mask_t*>(table.get_column(i)->valid);
  }

  size_t masks_size = sizeof(bit_mask_t*) * table.num_columns();

  bit_mask_t **d_masks = nullptr;
  RMM_ALLOC(&d_masks, masks_size, stream);
  cudaMemcpyAsync(d_masks, h_masks, masks_size, cudaMemcpyHostToDevice, stream);
  CHECK_STREAM(stream);

  return d_masks;
}

valid_table_filter make_valid_table_filter(cudf::table const &table, 
                                           cudf::any_or_all drop_if,
                                           cudaStream_t stream=0)
{
  return valid_table_filter(table, get_bitmasks(table, stream),
                            drop_if);
}

void destroy_valid_table_filter(valid_table_filter const& filter,
                                cudaStream_t stream = 0) {
  RMM_FREE(filter.d_masks, stream);
}

}  // namespace

namespace cudf {

/*
 * Filters a column to remove null elements.
 */
gdf_column drop_nulls(gdf_column const &input) {
  if (input.valid != nullptr && input.null_count != 0)
    return detail::copy_if(input, valid_column_filter{input});
  else // no null bitmask, so just copy
    return cudf::copy(input);
}

/*
 * Filters a table to remove null elements.
 */
table drop_nulls(table const &input, any_or_all drop_if) {
  if (cudf::has_nulls(input)) {
    valid_table_filter filter =
      make_valid_table_filter(input, drop_if);
    table result = detail::copy_if(input, filter);
    destroy_valid_table_filter(filter);
    return result;
  }
  else
    return cudf::copy(input);
}

}  // namespace cudf
