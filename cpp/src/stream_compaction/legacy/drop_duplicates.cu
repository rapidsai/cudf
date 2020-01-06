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


/*#include <cudf/cudf.h>
#include <cudf/types.hpp>
#include <cudf/legacy/copying.hpp>
#include <cudf/legacy/stream_compaction.hpp>
#include <cudf/table.hpp>
#include "table/device_table.cuh"
#include <table/device_table_row_operators.cuh>*/

#include "copy_if.cuh"
#include <cudf/legacy/table.hpp>
#include <table/legacy/device_table.cuh>
#include <table/legacy/device_table_row_operators.cuh>
#include <cudf/legacy/transform.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <cudf/utilities/legacy/nvcategory_util.hpp>
#include <nvstrings/NVCategory.h>

namespace cudf {
namespace detail {

/*
 * unique_copy copies elements from the range [first, last) to a range beginning
 * with output, except that in a consecutive group of duplicate elements only
 * depending on last argument keep, only the first one is copied, or the last
 * one is copied or neither is copied. The return value is the end of the range
 * to which the elements are copied.
 */
template<typename Exec,
         typename InputIterator,
         typename OutputIterator,
         typename BinaryPredicate,
    typename IndexType = typename
  thrust::iterator_difference<InputIterator>::type>
  OutputIterator unique_copy(Exec&& exec,
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
get_unique_ordered_indices(const cudf::table& keys,
                           const duplicate_keep_option keep,
                           const bool nulls_are_equal = true,
                           cudaStream_t stream=0)
{
  cudf::size_type ncols = keys.num_columns();
  cudf::size_type nrows = keys.num_rows();

  // sort only indices
  rmm::device_vector<cudf::size_type> sorted_indices(nrows);
  gdf_context context;
  gdf_column sorted_indices_col;
  CUDF_TRY(gdf_column_view(&sorted_indices_col, (void*)(sorted_indices.data().get()),
        nullptr, nrows, GDF_INT32));
  CUDF_TRY(gdf_order_by(keys.begin(),
        nullptr,
        keys.num_columns(),
        &sorted_indices_col,
        &context));

  // extract unique indices 
  rmm::device_vector<cudf::size_type> unique_indices(nrows);
  auto device_input_table = device_table::create(keys, stream);
  rmm::device_vector<cudf::size_type>::iterator result_end;

  if(cudf::has_nulls(keys)) {
    auto comp = row_equality_comparator<true>(*device_input_table,
        nulls_are_equal);
    result_end = unique_copy(rmm::exec_policy(stream)->on(stream),
        sorted_indices.begin(),
        sorted_indices.end(),
        unique_indices.begin(),
        comp,
        keep);
  } else {
    auto comp = row_equality_comparator<false>(*device_input_table,
        nulls_are_equal);
    result_end = unique_copy(rmm::exec_policy(stream)->on(stream),
        sorted_indices.begin(),
        sorted_indices.end(),
        unique_indices.begin(),
        comp,
        keep);
  }
  //not resizing vector to avoid copy

  return std::make_pair(std::move(unique_indices), 
                        thrust::distance(unique_indices.begin(), result_end));
}

cudf::size_type unique_count(const cudf::table& keys,
             const bool nulls_are_equal = true,
             cudaStream_t stream=0)
{
  cudf::size_type ncols = keys.num_columns();
  cudf::size_type nrows = keys.num_rows();

  // sort only indices
  rmm::device_vector<cudf::size_type> sorted_indices(nrows);
  gdf_context context;
  gdf_column sorted_indices_col;
  CUDF_TRY(gdf_column_view(&sorted_indices_col, static_cast<void*>(sorted_indices.data().get()),
        nullptr, nrows, GDF_INT32));
  CUDF_TRY(gdf_order_by(keys.begin(),
        nullptr,
        keys.num_columns(),
        &sorted_indices_col,
        &context));

  // count unique elements
  auto sorted_row_index = sorted_indices.begin();
  auto device_input_table = device_table::create(keys, stream);

  if(cudf::has_nulls(keys)) {
    auto comp = row_equality_comparator<true>(*device_input_table,
        nulls_are_equal);
    return thrust::count_if(rmm::exec_policy(stream)->on(stream),
              thrust::counting_iterator<cudf::size_type>(0),
              thrust::counting_iterator<cudf::size_type>(nrows),
              [sorted_row_index, comp] 
              __device__ (const cudf::size_type i) {
              return (i == 0 || !comp(sorted_row_index[i], sorted_row_index[i-1]));
              });
  } else {
    auto comp = row_equality_comparator<false>(*device_input_table,
        nulls_are_equal);
    return thrust::count_if(rmm::exec_policy(stream)->on(stream),
              thrust::counting_iterator<cudf::size_type>(0),
              thrust::counting_iterator<cudf::size_type>(nrows),
              [sorted_row_index, comp]
              __device__ (const cudf::size_type i) {
              return (i == 0 || !comp(sorted_row_index[i], sorted_row_index[i-1]));
              });
  }
}

} //namespace detail

cudf::table drop_duplicates(const cudf::table& input,
                            const cudf::table& keys,
                            const duplicate_keep_option keep,
                            const bool nulls_are_equal)
{
  CUDF_EXPECTS( input.num_rows() == keys.num_rows(), "number of \
rows in input table should be equal to number of rows in key colums table");

  if (0 == input.num_rows() || 
      0 == input.num_columns() ||
      0 == keys.num_columns() 
      ) {
    return cudf::empty_like(input);
  }
  rmm::device_vector<cudf::size_type> unique_indices;
  cudf::size_type unique_count; 
  std::tie(unique_indices, unique_count) =
    detail::get_unique_ordered_indices(keys, keep, nulls_are_equal);

  // Allocate output columns
  cudf::table destination_table(unique_count,
                                cudf::column_dtypes(input),
                                cudf::column_dtype_infos(input), true);
  // Ensure column names are preserved. Ideally we could call cudf::allocate_like
  // here, but the above constructor allocates and fills null bitmaps differently
  // than allocate_like. Doing this for now because the impending table + column
  // re-design will handle this better than another cudf::allocate_like overload
  std::transform(
    input.begin(), input.end(),
    destination_table.begin(), destination_table.begin(),
    [](const gdf_column* inp_col, gdf_column* out_col) {
      // a rather roundabout way to do a strcpy...
      gdf_column_view_augmented(out_col,
                                out_col->data, out_col->valid,
                                out_col->size, out_col->dtype,
                                out_col->null_count,
                                out_col->dtype_info,
                                inp_col->col_name);
      return out_col;
  });

  // run gather operation to establish new order
  cudf::gather(&input, unique_indices.data().get(), &destination_table);
  return destination_table;
}

cudf::size_type unique_count(gdf_column const& input,
                           bool const ignore_nulls,
                           bool const nan_as_null)
{
  if (0 == input.size || input.null_count == input.size) {
    return 0;
  }
  gdf_column col{input};
  //TODO: remove after NaN support to equality operator is added
  //if (nan_as_null)
  if ((col.dtype == GDF_FLOAT32 || col.dtype == GDF_FLOAT64)) {
    auto temp = nans_to_nulls(col);
    col.valid = reinterpret_cast<cudf::valid_type*>(temp.first);
    col.null_count = temp.second;
  }
  bool const has_nans{col.null_count > input.null_count};
  
  auto count = detail::unique_count({const_cast<gdf_column*>(&col)}, true);
  if ((col.dtype == GDF_FLOAT32 || col.dtype == GDF_FLOAT64))
    bit_mask::destroy_bit_mask(reinterpret_cast<bit_mask::bit_mask_t*>(col.valid));

  //TODO: remove after NaN support to equality operator is added
  // if nan is counted as null when null is already present.
  if (not nan_as_null and has_nans and cudf::has_nulls(input))
    ++count;

  if (ignore_nulls and cudf::has_nulls(input))
    return --count;
  else
    return count;
}

}  // namespace cudf
