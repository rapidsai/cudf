/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/random.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

template <typename T>
class GatherTest : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(GatherTest, cudf::test::NumericTypes);

TYPED_TEST(GatherTest, IdentityTest)
{
  constexpr cudf::size_type source_size{1000};

  auto data = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });
  cudf::test::fixed_width_column_wrapper<TypeParam> source_column(data, data + source_size);
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(data, data + source_size);

  cudf::table_view source_table({source_column});

  std::unique_ptr<cudf::table> result = cudf::gather(source_table, gather_map);

  for (auto i = 0; i < source_table.num_columns(); ++i) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(source_table.column(i), result->view().column(i));
  }

  CUDF_TEST_EXPECT_TABLES_EQUAL(source_table, result->view());
}

TYPED_TEST(GatherTest, ReverseIdentityTest)
{
  constexpr cudf::size_type source_size{1000};

  auto data = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });
  auto reversed_data =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return source_size - 1 - i; });

  cudf::test::fixed_width_column_wrapper<TypeParam> source_column(data, data + source_size);
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(reversed_data,
                                                             reversed_data + source_size);

  cudf::table_view source_table({source_column});

  std::unique_ptr<cudf::table> result = cudf::gather(source_table, gather_map);
  cudf::test::fixed_width_column_wrapper<TypeParam> expect_column(reversed_data,
                                                                  reversed_data + source_size);

  for (auto i = 0; i < source_table.num_columns(); ++i) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_column, result->view().column(i));
  }
}

TYPED_TEST(GatherTest, EveryOtherNullOdds)
{
  constexpr cudf::size_type source_size{1000};

  // Every other element is valid
  auto data     = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });
  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; });

  cudf::test::fixed_width_column_wrapper<TypeParam> source_column(
    data, data + source_size, validity);

  // Gather odd-valued indices
  auto map_data = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i * 2; });

  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(map_data,
                                                             map_data + (source_size / 2));

  cudf::table_view source_table({source_column});

  std::unique_ptr<cudf::table> result = cudf::gather(source_table, gather_map);

  auto expect_data  = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return 0; });
  auto expect_valid = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return 0; });
  cudf::test::fixed_width_column_wrapper<TypeParam> expect_column(
    expect_data, expect_data + source_size / 2, expect_valid);

  for (auto i = 0; i < source_table.num_columns(); ++i) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_column, result->view().column(i));
  }
}

TYPED_TEST(GatherTest, EveryOtherNullEvens)
{
  constexpr cudf::size_type source_size{1000};

  // Every other element is valid
  auto data     = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });
  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; });

  cudf::test::fixed_width_column_wrapper<TypeParam> source_column(
    data, data + source_size, validity);

  // Gather even-valued indices
  auto map_data =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i * 2 + 1; });

  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(map_data,
                                                             map_data + (source_size / 2));

  cudf::table_view source_table({source_column});

  std::unique_ptr<cudf::table> result = cudf::gather(source_table, gather_map);

  auto expect_data =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i * 2 + 1; });
  auto expect_valid = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return 1; });
  cudf::test::fixed_width_column_wrapper<TypeParam> expect_column(
    expect_data, expect_data + source_size / 2, expect_valid);

  for (auto i = 0; i < source_table.num_columns(); ++i) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_column, result->view().column(i));
  }
}

TYPED_TEST(GatherTest, AllNull)
{
  constexpr cudf::size_type source_size{1000};

  // Every element is invalid
  auto data     = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });
  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return 0; });

  // Create a gather map that gathers to random locations
  std::vector<cudf::size_type> host_map_data(source_size);
  std::iota(host_map_data.begin(), host_map_data.end(), 0);
  std::mt19937 g(0);
  std::shuffle(host_map_data.begin(), host_map_data.end(), g);

  cudf::test::fixed_width_column_wrapper<TypeParam> source_column{
    data, data + source_size, validity};
  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(host_map_data.begin(),
                                                             host_map_data.end());

  cudf::table_view source_table({source_column});

  std::unique_ptr<cudf::table> result = cudf::gather(source_table, gather_map);

  // Check that the result is also all invalid
  CUDF_TEST_EXPECT_TABLES_EQUAL(source_table, result->view());
}

TYPED_TEST(GatherTest, MultiColReverseIdentityTest)
{
  constexpr cudf::size_type source_size{1000};

  constexpr cudf::size_type n_cols = 3;

  auto data = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });
  auto reversed_data =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return source_size - 1 - i; });

  std::vector<cudf::test::fixed_width_column_wrapper<TypeParam>> source_column_wrappers;
  std::vector<cudf::column_view> source_columns;

  for (int i = 0; i < n_cols; ++i) {
    source_column_wrappers.push_back(
      cudf::test::fixed_width_column_wrapper<TypeParam>(data, data + source_size));
    source_columns.push_back(source_column_wrappers[i]);
  }

  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(reversed_data,
                                                             reversed_data + source_size);

  cudf::table_view source_table{source_columns};

  std::unique_ptr<cudf::table> result = cudf::gather(source_table, gather_map);

  cudf::test::fixed_width_column_wrapper<TypeParam> expect_column(reversed_data,
                                                                  reversed_data + source_size);

  for (auto i = 0; i < source_table.num_columns(); ++i) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_column, result->view().column(i));
  }
}

TYPED_TEST(GatherTest, MultiColNulls)
{
  constexpr cudf::size_type source_size{1000};

  static_assert(0 == source_size % 2, "Size of source data must be a multiple of 2.");

  constexpr cudf::size_type n_cols = 3;

  auto data     = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i; });
  auto validity = cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2; });

  std::vector<cudf::test::fixed_width_column_wrapper<TypeParam>> source_column_wrappers;
  std::vector<cudf::column_view> source_columns;

  for (int i = 0; i < n_cols; ++i) {
    source_column_wrappers.push_back(
      cudf::test::fixed_width_column_wrapper<TypeParam>(data, data + source_size, validity));
    source_columns.push_back(source_column_wrappers[i]);
  }

  auto reversed_data =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return source_size - 1 - i; });

  cudf::test::fixed_width_column_wrapper<int32_t> gather_map(reversed_data,
                                                             reversed_data + source_size);

  cudf::table_view source_table{source_columns};

  std::unique_ptr<cudf::table> result = cudf::gather(source_table, gather_map);

  // Expected data
  auto expect_data =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return source_size - i - 1; });
  auto expect_valid =
    cudf::detail::make_counting_transform_iterator(0, [](auto i) { return (i + 1) % 2; });

  cudf::test::fixed_width_column_wrapper<TypeParam> expect_column(
    expect_data, expect_data + source_size, expect_valid);

  for (auto i = 0; i < source_table.num_columns(); ++i) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expect_column, result->view().column(i));
  }
}
