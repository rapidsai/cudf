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

#ifndef _COMMON_GROUPBY_TEST_HPP
#define _COMMON_GROUPBY_TEST_HPP

#include <cudf/legacy/copying.hpp>
#include <cudf/legacy/groupby.hpp>
#include <cudf/legacy/table.hpp>
#include <tests/utilities/legacy/column_wrapper.cuh>
#include <tests/utilities/legacy/compare_column_wrappers.cuh>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include "type_info.hpp"

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
  rmm::device_vector<cudf::size_type> sorted_indices(keys.num_rows());
  gdf_column gdf_sorted_indices;
  gdf_column_view(&gdf_sorted_indices, sorted_indices.data().get(), nullptr,
                  sorted_indices.size(), GDF_INT32);
  gdf_context context;
  context.flag_null_sort_behavior = GDF_NULL_AS_LARGEST;
  gdf_order_by(keys.begin(), nullptr, keys.num_columns(), &gdf_sorted_indices,
               &context);

  cudf::table sorted_output_keys = cudf::allocate_like(keys, RETAIN);
  cudf::table sorted_output_values = cudf::allocate_like(values, RETAIN);

  cudf::gather(&keys, sorted_indices.data().get(), &sorted_output_keys);
  cudf::gather(&values, sorted_indices.data().get(), &sorted_output_values);

  EXPECT_EQ(cudaSuccess, cudaDeviceSynchronize());

  return std::make_pair(std::move(sorted_output_keys), std::move(sorted_output_values));
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
 

/**---------------------------------------------------------------------------*
 * @brief Verifies the equality of two vectors of columns 
 *
 * @param lhs The first array
 * @param rhs The second array
 *---------------------------------------------------------------------------**/
inline void expect_values_are_equal(std::vector<gdf_column*> lhs,
                                    std::vector<gdf_column*> rhs)  {
  EXPECT_EQ(lhs.size(), rhs.size());
  for (size_t i = 0; i < lhs.size(); i++)
  {
    EXPECT_EQ(lhs[i]->size, rhs[i]->size);
  }
  
  EXPECT_TRUE(
      std::equal(lhs.begin(), lhs.end(), rhs.begin(),
                 [](gdf_column const* lhs_col, gdf_column const* rhs_col) {
                   return cudf::type_dispatcher(
                       lhs_col->dtype, column_equality{}, *lhs_col, *rhs_col);
                 }));                                    
}

template<class table_type>
inline void destroy_columns(table_type* t) {
  std::for_each(t->begin(), t->end(), [](gdf_column* col) {
    gdf_column_free(col);
    delete col;
  });
}

}  // namespace detail

}  // namespace test
}  // namespace cudf
#endif
