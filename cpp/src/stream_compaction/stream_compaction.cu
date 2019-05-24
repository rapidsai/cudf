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
#include <table.hpp>
#include "table/device_table.cuh"
#include <table/device_table_row_operators.cuh>
#include <rmm/thrust_rmm_allocator.h>
 
namespace {

// Returns true if the mask is true and valid (non-null) for index i
// This is the filter functor for apply_boolean_mask
// Note we use a functor here so we can cast to a bitmask_t __restrict__
// pointer on the host side, which we can't do with a lambda.
template <bool has_data, bool has_nulls>
struct boolean_mask_filter
{
  boolean_mask_filter(gdf_column const & boolean_mask) :
    size{boolean_mask.size},
    data{reinterpret_cast<cudf::bool8 *>(boolean_mask.data)},
    bitmask{reinterpret_cast<bit_mask_t *>(boolean_mask.valid)}
    {}

  __device__ inline 
  bool operator()(gdf_index_type i)
  {
    if (i < size) {
      bool valid = !has_nulls || bit_mask::is_valid(bitmask, i);
      bool is_true = !has_data || (cudf::true_v == data[i]);
      return is_true && valid;
    }
    return false;
  }

  gdf_size_type size;
  cudf::bool8 const * __restrict__ data;
  bit_mask_t const  * __restrict__ bitmask;
};

// Returns true if the valid mask is true for index i
// Note we use a functor here so we can cast to a bitmask_t __restrict__
// pointer on the host side, which we can't do with a lambda.
struct valid_filter
{
  valid_filter(gdf_column const & column) :
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

}  // namespace

namespace cudf {

/*
 * Filters a column using a column of boolean values as a mask.
 *
 * calls apply_filter() with the `boolean_mask_filter` functor.
 */
gdf_column apply_boolean_mask(gdf_column const &input,
                              gdf_column const &boolean_mask) {
  CUDF_EXPECTS(boolean_mask.dtype == GDF_BOOL8, "Mask must be Boolean type");
  CUDF_EXPECTS(boolean_mask.data != nullptr ||
               boolean_mask.valid != nullptr, "Null boolean_mask");
  // zero-size inputs are OK, but otherwise input size must match mask size
  CUDF_EXPECTS(input.size == 0 || input.size == boolean_mask.size, 
               "Column size mismatch");

  if (boolean_mask.data == nullptr)
    return detail::copy_if(input, boolean_mask_filter<false, true>{boolean_mask});
  else if (boolean_mask.valid == nullptr || boolean_mask.null_count == 0)
    return detail::copy_if(input, boolean_mask_filter<true, false>{boolean_mask});
  else
    return detail::copy_if(input, boolean_mask_filter<true, true>{boolean_mask});
}

/*
 * Filters a column to remove null elements.
 */
gdf_column drop_nulls(gdf_column const &input) {
  if (input.valid != nullptr && input.null_count != 0)
    return detail::copy_if(input, valid_filter{input});
  else { // no null bitmask, so just copy
    return cudf::copy(input);
  }
}

rmm::device_vector<gdf_index_type>
get_unique_ordered_indices(const cudf::table& key_columns,
                           const bool keep_first)
{
  gdf_size_type ncols = key_columns.num_columns();
  gdf_size_type nrows = key_columns.num_rows();

  // sort only indices
  rmm::device_vector<gdf_size_type> sorted_indices(nrows);
  gdf_context context;
  gdf_column sorted_indices_col;
  CUDF_TRY(gdf_column_view(&sorted_indices_col, (void*)(sorted_indices.data().get()),
        nullptr, nrows, GDF_INT32));
  CUDF_TRY(gdf_order_by(key_columns.begin(),
        nullptr,
        key_columns.num_columns(),
        &sorted_indices_col,
        &context));

  // extract unique indices 
  rmm::device_vector<gdf_index_type> unique_indices(nrows);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  auto exec = rmm::exec_policy(stream)->on(stream);

  auto device_input_table = device_table::create(key_columns, stream);
  auto comp = row_equality_comparator<true>(*device_input_table, true);

  gdf_index_type* result_end;
  if (keep_first) {
      result_end = thrust::unique_copy(exec, 
              sorted_indices.begin(),
              sorted_indices.end(),
              unique_indices.data().get(),
              comp
              );
  } else {
      result_end = thrust::unique_copy(exec, 
              sorted_indices.rbegin(),
              sorted_indices.rend(),
              unique_indices.data().get(),
              comp
              );
  }
  // reorder unique indices
  thrust::sort(exec, unique_indices.data().get(), result_end);
  unique_indices.resize(thrust::distance(unique_indices.data().get(), result_end));
  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  return unique_indices;
}

cudf::table drop_duplicates(const cudf::table& input_table,
                            const cudf::table& key_columns,
                            const bool keep_first)
{
  //CUDF_EXPECTS(nullptr != key_col_indices, "key_col_indices is null");
  //CUDF_EXPECTS(0 < num_key_cols, "number of key colums should be greater than zero");

  if (0 == input_table.num_rows() || 
      0 == input_table.num_columns() ||
      0 == key_columns.num_columns() 
      ) {
    return cudf::table();
  }
  rmm::device_vector<gdf_index_type> unique_indices = get_unique_ordered_indices(key_columns, keep_first);
  // Allocate output columns
  cudf::table destination_table(unique_indices.size(), cudf::column_dtypes(input_table), true);
  // run gather operation to establish new order
  cudf::gather(&input_table, unique_indices.data().get(), &destination_table);
  return destination_table;
}
}  // namespace cudf
