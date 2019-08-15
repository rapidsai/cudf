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
#include <utilities/type_dispatcher.hpp>
#include "type_info.hpp"
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
inline std::pair<table, table> sort_by_key(cudf::table const& keys,
                                           cudf::table const& values) {
  CUDF_EXPECTS(keys.num_rows() == values.num_rows(),
               "Size mismatch between keys and values");
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
  cudf::table sorted_output_values = cudf::allocate_like(values, true);

  cudf::gather(&keys, sorted_indices.data().get(), &sorted_output_keys);
  cudf::gather(&values, sorted_indices.data().get(), &sorted_output_values);

  EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());

  return std::make_pair(sorted_output_keys, sorted_output_values);
}

struct column_equality {
  template <typename T>
  bool operator()(gdf_column lhs, gdf_column rhs) const {
    std::unique_ptr<column_wrapper<T>> lhs_col;
    std::unique_ptr<column_wrapper<T>> rhs_col;
    lhs_col.reset(new column_wrapper<T>(lhs));
    rhs_col.reset(new column_wrapper<T>(rhs));
    printf("lhs: %p,%d\n", lhs.valid, lhs.null_count);
    print_gdf_column(&lhs);
    printf("rhs: %p,%d\n", rhs.valid, rhs.null_count);
    print_gdf_column(&rhs);
    printf("\n");
    
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
  {
    rmm::device_vector<gdf_size_type> sorted_indices_tmp = get_last_column(sorted_keys_table);
    thrust::host_vector<gdf_size_type>  sorted_indices = sorted_indices_tmp;
    std::cout << "sorted_indexes --> " << std::endl;
    for (size_t i = 0; i < sorted_indices.size(); i++) {
       std::cout << "indexes: " << i << " | " << sorted_indices[i] << std::endl;
    }
    
  }
  // the gather !!                                               
  cudf::gather(&output_keys, (gdf_index_type *)group_indices_col.data,
               &output_keys);

  for (gdf_size_type i = 0; i < output_keys.num_columns(); i++) {
    output_keys.get_column(i)->size = group_indices_col.size;
    // if (not cudf::has_nulls(input_keys)) {
    //     output_keys.get_column(i)->valid = nullptr;
    // }
  }                                                                                                 
  return  output_keys;
}


template <cudf::groupby::sort::operators op, typename Key, typename Value,
          typename ResultValue>
void single_column_groupby_test(column_wrapper<Key> keys,
                                column_wrapper<Value> values,
                                column_wrapper<Key> expected_keys,
                                column_wrapper<ResultValue> expected_values) {
  using namespace cudf::test;
  using namespace cudf::groupby::sort;

  std::cout << "keys: \n";
  keys.print();

  std::cout << "values: \n";
  values.print();
  
  static_assert(std::is_same<ResultValue, expected_result_t<Value, op>>::value,
                "Incorrect type for expected_values.");
  ASSERT_EQ(keys.size(), values.size())
      << "Number of keys must be equal to number of values.";
  ASSERT_EQ(expected_keys.size(), expected_values.size())
      << "Number of keys must be equal to number of values.";

  cudf::table input_keys{keys.get()};

  cudf::table input_values{values.get()};

  std::cout << "input_keys: \n";
  print_gdf_column(input_keys.get_column(0));

  std::cout << "input_values: \n";
  print_gdf_column(input_values.get_column(0));
  if (input_values.get_column(0)->valid)
    print_valid_data(input_values.get_column(0)->valid, input_values.get_column(0)->size);
  
  cudf::table actual_values_table {expected_values.get()};

  std::cout << "groupby -->\n";
  cudf::table actual_keys_table  = groupby_wo_agg(input_keys);
  
  if (actual_keys_table.num_rows() == 0) {
    printf("***empty rows!!!\n");
    return ;
  }

  std::cout << "<-- groupby \n";

  EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());

  cudf::table sorted_actual_keys;
  cudf::table sorted_actual_values;
  std::tie(sorted_actual_keys, sorted_actual_values) =
      detail::sort_by_key(actual_keys_table, actual_values_table);

  cudf::table sorted_expected_keys;
  cudf::table sorted_expected_values;
  std::tie(sorted_expected_keys, sorted_expected_values) =
      detail::sort_by_key({expected_keys.get()}, {expected_values.get()});

  // sorted_actual_keys
  std::cout << "gdf output: \n";

  print_gdf_column(actual_keys_table.get_column(0));
  print_gdf_column(actual_values_table.get_column(0));

  std::cout << "expected output: \n";
 
  print_gdf_column(sorted_expected_keys.get_column(0));
  print_gdf_column(sorted_expected_values.get_column(0));

  // sorted_actual_values

  printf("output keys\n");  
  CUDF_EXPECT_NO_THROW(detail::expect_tables_are_equal(sorted_actual_keys,
                                                       sorted_expected_keys));
  printf("output values\n");                                                       
  CUDF_EXPECT_NO_THROW(detail::expect_tables_are_equal(sorted_actual_values,
                                                       sorted_expected_values));

  detail::destroy_table(&sorted_actual_keys);
  detail::destroy_table(&sorted_actual_values);
  detail::destroy_table(&sorted_expected_keys);
  detail::destroy_table(&sorted_expected_values);
}

}  // namespace test
}  // namespace cudf
#endif
