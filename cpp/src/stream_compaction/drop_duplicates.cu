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
#include <cudf/copying.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table.hpp>
#include "table/device_table.cuh"
#include <table/device_table_row_operators.cuh>*/

#include "copy_if.cuh"
#include <cudf/legacy/table.hpp>
#include <table/legacy/device_table.cuh>
#include <table/legacy/device_table_row_operators.cuh>

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
  cudf::table destination_table(unique_count,
                                cudf::column_dtypes(input_table),
                                cudf::column_dtype_infos(input_table), true);
  // Ensure column names are preserved. Ideally we could call cudf::allocate_like
  // here, but the above constructor allocates and fills null bitmaps differently
  // than allocate_like. Doing this for now because the impending table + column
  // re-design will handle this better than another cudf::allocate_like overload
  std::transform(
    input_table.begin(), input_table.end(),
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
  cudf::gather(&input_table, unique_indices.data().get(), &destination_table);
  nvcategory_gather_table(input_table, destination_table);
  return destination_table;
}
}  // namespace cudf
