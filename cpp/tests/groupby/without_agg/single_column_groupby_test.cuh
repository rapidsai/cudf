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

#ifndef _SORT_SINGLE_COLUMN_GROUPBY_TEST_CUH
#define _SORT_SINGLE_COLUMN_GROUPBY_TEST_CUH

#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/legacy/table.hpp>
#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/compare_column_wrappers.cuh>
#include <table/legacy/device_table.cuh>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <utility>

namespace cudf {
namespace test {
namespace detail {

/**---------------------------------------------------------------------------*
 * @brief Performs a sort-by-key on tables.
 *
 * Sorts the rows of `keys` into ascending order and reorders the corresponding
 * rows of `values` to match the sorted order.
 *
 * @param keys The keys to sort
 * @param values The values to reorder
 * @return std::pair<table, table> A pair whose first element contains the
 * sorted keys and second element contains reordered values.
 *---------------------------------------------------------------------------**/
inline table  sort_by_key(cudf::table const& keys ) {
  rmm::device_vector<gdf_index_type> sorted_indices(keys.num_rows());
  gdf_column gdf_sorted_indices;
  gdf_column_view(&gdf_sorted_indices, sorted_indices.data().get(), nullptr,
                  sorted_indices.size(), GDF_INT32);
  gdf_context context;
  context.flag_groupby_include_nulls = true; // for sql
  context.flag_null_sort_behavior = GDF_NULL_AS_LARGEST;
  gdf_order_by(keys.begin(), nullptr, keys.num_columns(), &gdf_sorted_indices,
               &context);

  cudf::table sorted_output_keys = cudf::allocate_like(keys, true);

  cudf::gather(&keys, sorted_indices.data().get(), &sorted_output_keys);
  EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());

  return sorted_output_keys;
}

struct column_equality {
  template <typename T>
  bool operator()(gdf_column lhs, gdf_column rhs) const {
    std::unique_ptr<column_wrapper<T>> lhs_col;
    std::unique_ptr<column_wrapper<T>> rhs_col;
    lhs_col.reset(new column_wrapper<T>(lhs));
    rhs_col.reset(new column_wrapper<T>(rhs));   
    expect_columns_are_equal(*lhs_col, *rhs_col);
    return true;
  }
};

/**---------------------------------------------------------------------------*
 * @brief Verifies the equality of two tables
 *
 * @param lhs The first table
 * @param rhs The second table
 *---------------------------------------------------------------------------**/
inline void expect_tables_are_equal(cudf::table const& lhs,
                                    cudf::table const& rhs) {
  EXPECT_EQ(lhs.num_columns(), rhs.num_columns());
  EXPECT_EQ(lhs.num_rows(), rhs.num_rows());
  EXPECT_TRUE(
      std::equal(lhs.begin(), lhs.end(), rhs.begin(),
                 [](gdf_column const* lhs_col, gdf_column const* rhs_col) {
                   return cudf::type_dispatcher(
                       lhs_col->dtype, column_equality{}, *lhs_col, *rhs_col);
                 }));
}

inline void destroy_table(table* t) {
  std::for_each(t->begin(), t->end(), [](gdf_column* col) {
    gdf_column_free(col);
    delete col;
  });
}
}  // namespace detail


cudf::table compose_inputs(cudf::table input_table, gdf_column* col) {
  std::vector<gdf_column*> output(input_table.num_columns());  
  std::transform(input_table.begin(), input_table.end(), output.begin(), [](const gdf_column *item){
    return (gdf_column *)item;
  }); 
  output.push_back(col);

  gdf_column **group_by_input_key = output.data();
  return cudf::table{group_by_input_key, input_table.num_columns() + 1};
}

cudf::table compose_output_keys(cudf::table input_table) {
  std::vector<gdf_column*> output(input_table.num_columns() - 1);  
  std::transform(input_table.begin(), input_table.end() - 1, output.begin(), [](const gdf_column *item){
    return (gdf_column *)item;
  }); 
  return cudf::table {output};
}

rmm::device_vector<gdf_size_type> get_last_column (cudf::table current_table) {
  auto num_column = current_table.num_columns();
  gdf_column * sorted_column = current_table.get_column(num_column - 1);
  rmm::device_vector<gdf_size_type> returned_vector(current_table.num_rows());
  cudaMemcpy(returned_vector.data().get(), sorted_column->data, sorted_column->size * sizeof(gdf_size_type), cudaMemcpyDeviceToDevice); 
  return returned_vector;
}

cudf::table groupby_wo_agg(cudf::table const& input_keys) {
  gdf_context context;
  auto ignore_null_keys = true;
  if (not ignore_null_keys) { // SQL
    context.flag_groupby_include_nulls = true;
    context.flag_null_sort_behavior = GDF_NULL_AS_LARGEST;
  } else { // PANDAS
    context.flag_groupby_include_nulls = false;
    context.flag_null_sort_behavior = GDF_NULL_AS_LARGEST;
  }

  std::vector<int> groupby_col_indices;
  for (gdf_size_type i = 0; i < input_keys.num_columns(); i++)
    groupby_col_indices.push_back(i);

  cudf::table sorted_keys_table;
  gdf_column group_indices_col;
  rmm::device_vector<gdf_size_type> gpu_sorted_indices;
  
  auto nrows = input_keys.num_rows();
  rmm::device_vector<gdf_size_type> d_sorted_indices(nrows);
  thrust::sequence(d_sorted_indices.begin(), d_sorted_indices.end(), 0, 1);

  gdf_column sorted_indices_col{};
  CUDF_TRY(gdf_column_view(&sorted_indices_col,
                           (void *)(d_sorted_indices.data().get()), nullptr,
                           nrows, GDF_INT32));

  auto input_table = compose_inputs(input_keys, &sorted_indices_col);
  std::tie(sorted_keys_table,
                        group_indices_col) = gdf_group_by_without_aggregations(input_table,
                                                                          groupby_col_indices.size(),
                                                                          groupby_col_indices.data(),
                                                                          &context);
  cudf::table output_keys = compose_output_keys(sorted_keys_table);
  cudf::table destination_table(group_indices_col.size,
                                  cudf::column_dtypes(output_keys),
                                  cudf::column_dtype_infos(output_keys),
                                  cudf::has_nulls(input_keys));
    
  cudf::gather(&output_keys, (gdf_index_type *)group_indices_col.data,
                &destination_table); 
                                                                                              
  return  destination_table;
}


template <typename Key>
void single_column_groupby_test(column_wrapper<Key> keys,
                                column_wrapper<Key> expected_keys) {
  using namespace cudf::test;

  cudf::table input_keys{keys.get()};

  cudf::table actual_keys_table  = groupby_wo_agg(input_keys);
  
  if (actual_keys_table.num_rows() == 0) {
    return ;
  }
  cudf::table sorted_expected_keys = 
      detail::sort_by_key({expected_keys.get()});

  EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());

  CUDF_EXPECT_NO_THROW(detail::expect_tables_are_equal(actual_keys_table,
                                                       sorted_expected_keys));

  detail::destroy_table(&actual_keys_table); 
  detail::destroy_table(&sorted_expected_keys);
}

}  // namespace test
}  // namespace cudf
#endif
