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

#include <cudf/cudf.h>
#include <cudf/legacy/copying.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/legacy/nvcategory_util.hpp>
#include <table/legacy/device_table.cuh>
#include <table/legacy/device_table_row_operators.cuh>
#include <cudf/utilities/error.hpp>

#include <thrust/fill.h>
#include <algorithm>
#include <cassert>

gdf_column gdf_unique_indices(cudf::table const& input_table,
                              gdf_context const& context) {
  cudf::size_type ncols = input_table.num_columns();
  cudf::size_type nrows = input_table.num_rows();

  rmm::device_vector<void*> d_cols(ncols);
  rmm::device_vector<int> d_types(ncols, 0);
  void** d_col_data = d_cols.data().get();
  int* d_col_types = d_types.data().get();

  cudf::size_type* result_end;
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  // Allocating memory for GDF column
  gdf_column unique_indices{};
  RMM_TRY(
      RMM_ALLOC(&unique_indices.data, sizeof(cudf::size_type) * nrows, nullptr));
  unique_indices.dtype = cudf::gdf_dtype_of<cudf::size_type>();

  auto counting_iter = thrust::make_counting_iterator<cudf::size_type>(0);
  auto device_input_table = device_table::create(input_table);
  bool nullable = device_input_table.get()->has_nulls();
  if (nullable) {
    auto comp = row_equality_comparator<true>(*device_input_table, true);
    result_end = thrust::unique_copy(rmm::exec_policy(stream)->on(stream),
        counting_iter, counting_iter + nrows,
        static_cast<cudf::size_type*>(unique_indices.data), comp);
  } else {
    auto comp = row_equality_comparator<false>(*device_input_table, true);
    result_end = thrust::unique_copy(rmm::exec_policy(stream)->on(stream),
         counting_iter, counting_iter + nrows,
        static_cast<cudf::size_type*>(unique_indices.data), comp);
  }

  // size of the GDF column is being resized
  unique_indices.size = thrust::distance(
      static_cast<cudf::size_type*>(unique_indices.data), result_end);
  gdf_column resized_unique_indices = cudf::copy(unique_indices);
  // Free old column, as we have resized (implicitly)
  gdf_column_free(&unique_indices);

  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  return resized_unique_indices;
}

std::pair<cudf::table, gdf_column> gdf_group_by_without_aggregations(
    cudf::table const& input_table, cudf::size_type num_key_cols,
    cudf::size_type const* key_col_indices, gdf_context* context) {
  CUDF_EXPECTS(nullptr != key_col_indices, "key_col_indices is null");
  CUDF_EXPECTS(0 < num_key_cols,
               "number of key colums should be greater than zero");
  gdf_column unique_indices{};
  unique_indices.dtype = cudf::gdf_dtype_of<cudf::size_type>();

  if (0 == input_table.num_rows()) {
    return std::make_pair(cudf::empty_like(input_table), std::move(unique_indices));
  }

  cudf::size_type nrows = input_table.num_rows();

  // Ask if input table has nulls
  std::vector<gdf_column *> key_columns(num_key_cols);
  std::transform(
      key_col_indices, 
      key_col_indices + num_key_cols, 
      key_columns.begin(),
      [&input_table](cudf::size_type target_index) { 
        return const_cast<gdf_column*>(input_table.get_column(target_index)); 
      }
  );
  // Allocate output columns
  auto allocate_bitmasks =  cudf::has_nulls( cudf::table(key_columns) );
  cudf::table destination_table(nrows,
                                cudf::column_dtypes(input_table),
                                cudf::column_dtype_infos(input_table),
                                allocate_bitmasks);

  std::vector<gdf_column*> key_cols_vect(num_key_cols);
  std::transform(
      key_col_indices, key_col_indices + num_key_cols, key_cols_vect.begin(),
      [&input_table](cudf::size_type const index) {
        return const_cast<gdf_column*>(input_table.get_column(index));
      });
  cudf::table key_col_table(key_cols_vect.data(), key_cols_vect.size());

  rmm::device_vector<cudf::size_type> sorted_indices(nrows);
  gdf_column sorted_indices_col{};
  CUDF_TRY(gdf_column_view(&sorted_indices_col,
                           (void*)(sorted_indices.data().get()), nullptr, nrows,
                           GDF_INT32));

  if (context->flag_groupby_include_nulls ||
      !cudf::has_nulls(key_col_table)) {  // SQL style
    CUDF_TRY(gdf_order_by(key_col_table.begin(), nullptr,
                          key_col_table.num_columns(), &sorted_indices_col,
                          context));
  } else {  // Pandas style

    // Pandas style ignores groups that have nulls in their keys, so we want to
    // filter them out. We will create a bitmask (key_cols_bitmask) that
    // represents if there is any null in any of they key columns. We create a
    // modified set of key columns (modified_key_col_table), where the first key
    // column will take this bitmask (key_cols_bitmask) Then if we set
    // flag_null_sort_behavior = GDF_NULL_AS_LARGEST, then when we sort by the
    // key columns, then all the rows where any of the key columns contained a
    // null, these will be at the end of the sorted set. Then we can figure out
    // how many of those rows contained any nulls and adjust the size of our
    // sorted data set to ignore the rows where there were any nulls in the key
    // columns

    auto key_cols_bitmask = row_bitmask(key_col_table);

    gdf_column modified_fist_key_col{};
    modified_fist_key_col.data = key_cols_vect[0]->data;
    modified_fist_key_col.size = key_cols_vect[0]->size;
    modified_fist_key_col.dtype = key_cols_vect[0]->dtype;
    modified_fist_key_col.null_count = key_cols_vect[0]->null_count;
    modified_fist_key_col.valid =
        reinterpret_cast<cudf::valid_type*>(key_cols_bitmask.data().get());

    std::vector<gdf_column*> modified_key_cols_vect = key_cols_vect;
    modified_key_cols_vect[0] = &modified_fist_key_col;
    cudf::table modified_key_col_table(modified_key_cols_vect.data(),
                                       modified_key_cols_vect.size());

    gdf_context temp_ctx;
    temp_ctx.flag_null_sort_behavior = GDF_NULL_AS_LARGEST;

    CUDF_TRY(gdf_order_by(modified_key_col_table.begin(), nullptr,
                          modified_key_col_table.num_columns(),
                          &sorted_indices_col, &temp_ctx));

    int valid_count;
    CUDF_TRY(gdf_count_nonzero_mask(
        reinterpret_cast<cudf::valid_type*>(key_cols_bitmask.data().get()), nrows,
        &valid_count));

    std::for_each(destination_table.begin(), destination_table.end(),
                  [valid_count](gdf_column* col) { col->size = valid_count; });
  }

  // run gather operation to establish new order
  cudf::gather(&input_table, sorted_indices.data().get(), &destination_table);

  std::vector<gdf_column*> key_cols_vect_out(num_key_cols);
  std::transform(key_col_indices, key_col_indices + num_key_cols,
                 key_cols_vect_out.begin(),
                 [&destination_table](cudf::size_type const index) {
                   return destination_table.get_column(index);
                 });
  cudf::table key_col_sorted_table(key_cols_vect_out.data(),
                                   key_cols_vect_out.size());

  return std::make_pair(std::move(destination_table),
                        gdf_unique_indices(key_col_sorted_table, *context));
}
