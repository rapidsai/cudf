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
#include <cudf/legacy/table.hpp>
#include <thrust/logical.h>
 
namespace {

// Returns false if the valid mask is false for index i in ANY/ALL columns of
// table indicated by column_indices, where ANY/ALL is the value of drop_if.
// Columns not indexed by column_indices are not checked
struct valid_table_filter
{
  valid_table_filter(bit_mask_t **masks,
                     gdf_size_type num_columns,
                     cudf::any_or_all drop_if) 
  : drop_if(drop_if),
    num_columns(num_columns),
    d_masks(masks) {}

  __device__ inline 
  bool operator()(gdf_index_type i)
  {
    auto valid = [i](auto mask) { 
      return (mask == nullptr) || bit_mask::is_valid(mask, i);
    };

    if (drop_if == cudf::ALL) // drop rows that have a null in all columns
      return thrust::any_of(thrust::seq, d_masks, d_masks + num_columns, valid);
    else // drop_if == cudf::ANY => drop rows that have any nulls
      return thrust::all_of(thrust::seq, d_masks, d_masks + num_columns, valid); 
  }

  cudf::any_or_all drop_if;
  gdf_size_type num_columns;
  bit_mask_t **d_masks;
};

bit_mask_t** get_bitmasks(cudf::table const &table, cudaStream_t stream = 0) {
  bit_mask_t** h_masks = new bit_mask_t*[table.num_columns()];
  
  int i = 0;
  for (auto col : table) {
    h_masks[i++] = reinterpret_cast<bit_mask_t*>(col->valid);
  }

  size_t masks_size = sizeof(bit_mask_t*) * table.num_columns();

  bit_mask_t **d_masks = nullptr;
  RMM_TRY(RMM_ALLOC(&d_masks, masks_size, stream));
  CUDA_TRY(cudaMemcpyAsync(d_masks, h_masks, masks_size,
                           cudaMemcpyHostToDevice, stream));
  CHECK_STREAM(stream);

  return d_masks;
}

valid_table_filter make_valid_table_filter(cudf::table const &table,
                                           cudf::any_or_all drop_if,
                                           cudaStream_t stream=0)
{
  return valid_table_filter(get_bitmasks(table, stream),
                            table.num_columns(),
                            drop_if);
}

void destroy_valid_table_filter(valid_table_filter const& filter,
                                cudaStream_t stream = 0) {
  RMM_FREE(filter.d_masks, stream);
}

}  // namespace

namespace cudf {

/*
 * Filters a table to remove null elements.
 */
table drop_nulls(table const &input,
                 table const &keys,
                 any_or_all drop_if) {
  if (keys.num_columns() == 0 || keys.num_rows() == 0 ||
      not cudf::has_nulls(keys))
    return cudf::copy(input);

  CUDF_EXPECTS(keys.num_rows() <= input.num_rows(), 
               "Column size mismatch");
  
  valid_table_filter filter =
    make_valid_table_filter(keys, drop_if);
  table result = detail::copy_if(input, filter);
  destroy_valid_table_filter(filter);
  return result;
}

}  // namespace cudf
