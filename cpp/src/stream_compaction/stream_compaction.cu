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
#include "table/device_table.cuh"
#include <table/device_table_row_operators.cuh>
#include <rmm/thrust_rmm_allocator.h>
#include <string/nvcategory_util.hpp>
#include <nvstrings/NVCategory.h>
 
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
  if (boolean_mask.size == 0 || input.size == 0)
      return cudf::empty_like(input);

  // for non-zero-length masks we expect one of the pointers to be non-null    
  CUDF_EXPECTS(boolean_mask.data != nullptr ||
               boolean_mask.valid != nullptr, "Null boolean_mask");
  CUDF_EXPECTS(boolean_mask.dtype == GDF_BOOL8, "Mask must be Boolean type");
  
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


namespace detail {

/*
 * unique_copy copies elements from the range [first, last) to a range beginning
 * with output, except that in a consecutive group of duplicate elements only
 * depending on last argument keep, only the first one is copied, or the last
 * one is copied or neither is copied. The return value is the end of the range
 * to which the elements are copied.
 */
template<typename DerivedPolicy,
         typename InputIterator,
         typename OutputIterator,
         typename BinaryPredicate,
    typename IndexType = typename
  thrust::iterator_difference<InputIterator>::type>
  OutputIterator unique_copy(thrust::execution_policy<DerivedPolicy> &exec,
                             InputIterator first,
                             InputIterator last,
                             OutputIterator output,
                             BinaryPredicate comp,
                             const duplicate_keep_option keep)
{
  IndexType n = (last-first)-1;
  if (keep == duplicate_keep_option::KEEP_FIRST) {
      return thrust::copy_if(exec,
              first,
              last,
              thrust::counting_iterator<IndexType>(0),
              output,
              [first, comp, n] __device__ (const IndexType i) {
              return (i == 0 || !comp(first[i], first[i-1]));
              });
  } else if (keep == duplicate_keep_option::KEEP_LAST) {
      return thrust::copy_if(exec,
              first,
              last,
              thrust::counting_iterator<IndexType>(0),
              output,
              [first, comp, n] __device__ (const IndexType i) {
              return (i == n || !comp(first[i], first[i+1]));
              });
  } else {
      return thrust::copy_if(exec,
              first,
              last,
              thrust::counting_iterator<IndexType>(0),
              output,
              [first, comp, n] __device__ (const IndexType i) {
              return (i == 0 || !comp(first[i], first[i-1])) 
                  && (i == n || !comp(first[i], first[i+1]));
              });
  }
}

auto 
get_unique_ordered_indices(const cudf::table& key_columns,
                           const duplicate_keep_option keep,
                           const bool nulls_are_equal = true,
                           cudaStream_t stream=0)
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
  auto exec = rmm::exec_policy(stream)->on(stream);
  auto device_input_table = device_table::create(key_columns, stream);
  rmm::device_vector<gdf_size_type>::iterator result_end;

  bool nullable = device_input_table->has_nulls();
  if(nullable) {
    auto comp = row_equality_comparator<true>(*device_input_table,
        nulls_are_equal);
    result_end = unique_copy(exec,
        sorted_indices.begin(),
        sorted_indices.end(),
        unique_indices.begin(),
        comp,
        keep);
  } else {
    auto comp = row_equality_comparator<false>(*device_input_table,
        nulls_are_equal);
    result_end = unique_copy(exec,
        sorted_indices.begin(),
        sorted_indices.end(),
        unique_indices.begin(),
        comp,
        keep);
  }
  //not resizing vector to avoid copy

  return std::make_pair(unique_indices, 
                        thrust::distance(unique_indices.begin(), result_end));
}
} //namespace detail

cudf::table drop_duplicates(const cudf::table& input_table,
                            const cudf::table& key_columns,
                            const duplicate_keep_option keep,
                            const bool nulls_are_equal)
{
  CUDF_EXPECTS( input_table.num_rows() == key_columns.num_rows(), "number of \
rows in input table should be equal to number of rows in key colums table");

  if (0 == input_table.num_rows() || 
      0 == input_table.num_columns() ||
      0 == key_columns.num_columns() 
      ) {
    return cudf::empty_like(input_table);
  }
  rmm::device_vector<gdf_index_type> unique_indices;
  gdf_size_type unique_count; 
  std::tie(unique_indices, unique_count) =
    detail::get_unique_ordered_indices(key_columns, keep, nulls_are_equal);
  // Allocate output columns
  cudf::table destination_table(unique_count, cudf::column_dtypes(input_table), true);
  // run gather operation to establish new order
  cudf::gather(&input_table, unique_indices.data().get(), &destination_table);
  nvcategory_gather_table(input_table, destination_table);
  return destination_table;
}
}  // namespace cudf
