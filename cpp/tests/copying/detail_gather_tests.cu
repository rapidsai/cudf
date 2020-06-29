/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <tests/strings/utilities.h>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/type_lists.hpp>

template <typename T>
class GatherTest : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(GatherTest, cudf::test::NumericTypes);

// This test exercises using different iterator types as gather map inputs
// to cudf::detail::gather -- device_vector and raw pointers.
TYPED_TEST(GatherTest, GatherDetailDeviceVectorTest)
{
  constexpr cudf::size_type source_size{1000};
  rmm::device_vector<cudf::size_type> gather_map(source_size);
  thrust::sequence(thrust::device, gather_map.begin(), gather_map.end());

  auto data = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i; });
  cudf::test::fixed_width_column_wrapper<TypeParam> source_column(data, data + source_size);

  cudf::table_view source_table({source_column});

  // test with device vector iterators
  {
    std::unique_ptr<cudf::table> result =
      cudf::detail::gather(source_table, gather_map.begin(), gather_map.end());

    for (auto i = 0; i < source_table.num_columns(); ++i) {
      cudf::test::expect_columns_equal(source_table.column(i), result->view().column(i));
    }

    cudf::test::expect_tables_equal(source_table, result->view());
  }

  // test with raw pointers
  {
    std::unique_ptr<cudf::table> result = cudf::detail::gather(
      source_table, gather_map.data().get(), gather_map.data().get() + gather_map.size());

    for (auto i = 0; i < source_table.num_columns(); ++i) {
      cudf::test::expect_columns_equal(source_table.column(i), result->view().column(i));
    }

    cudf::test::expect_tables_equal(source_table, result->view());
  }
}

TYPED_TEST(GatherTest, GatherDetailInvalidIndexTest)
{
  constexpr cudf::size_type source_size{1000};

  auto data = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i; });
  cudf::test::fixed_width_column_wrapper<TypeParam> source_column(data, data + source_size);
  auto gather_map_data =
    cudf::test::make_counting_transform_iterator(0, [](auto i) { return (i % 2) ? -1 : i; });
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(gather_map_data,
                                                             gather_map_data + (source_size * 2));

  cudf::table_view source_table({source_column});
  std::unique_ptr<cudf::table> result =
    cudf::detail::gather(source_table,
                         gather_map,
                         cudf::detail::out_of_bounds_policy::IGNORE,
                         cudf::detail::negative_index_policy::NOT_ALLOWED);

  auto expect_data =
    cudf::test::make_counting_transform_iterator(0, [](auto i) { return (i % 2) ? 0 : i; });
  auto expect_valid = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return (i % 2) || (i >= source_size) ? 0 : 1; });
  cudf::test::fixed_width_column_wrapper<TypeParam> expect_column(
    expect_data, expect_data + (source_size * 2), expect_valid);

  for (auto i = 0; i < source_table.num_columns(); ++i) {
    cudf::test::expect_columns_equal(expect_column, result->view().column(i));
  }
}
