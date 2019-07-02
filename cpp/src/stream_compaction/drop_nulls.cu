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
  void get_bitmasks(bit_mask_t **masks,
                    cudf::table const &table,
                    cudaStream_t stream = 0) {
    RMM_ALLOC(masks, sizeof(bit_mask_t*) * table.num_rows(), stream);
    
    for (int i = 0; i < table.num_rows(); ++i) {
      cudaMemcpyAsync(masks[i], 
                      reinterpret_cast<const void*>(&(table.get_column(i)->valid)),
                      sizeof(bit_mask_t), cudaMemcpyHostToDevice, stream);    
    }
  }

  valid_table_filter(cudf::table const & table, cudaStream_t stream = 0) :
    size(table.num_rows()), stream(stream)
  {
    get_bitmasks(d_masks, table, stream);
  }

  ~valid_table_filter() {
    RMM_FREE(d_masks, stream);
  }

  __device__ inline 
  bool operator()(gdf_index_type i)
  {
    if (i < size) {
      bool valid = true;
      int c = 0;
      while (valid) {
        valid = valid and bit_mask::is_valid(d_masks[c++], i);
      }
      return valid;
    }
    return false;
  }

  gdf_size_type size;
  bit_mask_t **d_masks;
  cudaStream_t stream;
};

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
table drop_nulls(table const &input) {
  if (cudf::has_nulls(input))
    return detail::copy_if(input, valid_table_filter{input});
  else
    return cudf::copy(input);
}

}  // namespace cudf
