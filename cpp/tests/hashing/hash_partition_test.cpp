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
#include <cudf/hashing.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/type_lists.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

using cudf::test::fixed_width_column_wrapper;
using cudf::test::strings_column_wrapper;
using cudf::test::expect_columns_equal;
using cudf::test::expect_tables_equal;

// Transform vector of column wrappers to vector of column views
template <typename T>
auto make_view_vector(std::vector<T> const& columns)
{
  std::vector<cudf::column_view> views(columns.size());
  std::transform(columns.begin(), columns.end(), views.begin(),
    [](auto const& col) { return static_cast<cudf::column_view>(col); });
  return views;
}

class HashPartition : public cudf::test::BaseFixture {};

TEST_F(HashPartition, InvalidColumnsToHash)
{
  fixed_width_column_wrapper<float> floats({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
  fixed_width_column_wrapper<int16_t> integers({1, 2, 3, 4, 5, 6, 7, 8});
  strings_column_wrapper strings({"a", "bb", "ccc", "d", "ee", "fff", "gg", "h"});
  auto input = cudf::table_view({floats, integers, strings});

  auto columns_to_hash = std::vector<cudf::size_type>({-1});

  cudf::size_type const num_partitions = 3;
  EXPECT_THROW(cudf::hash_partition(input, columns_to_hash, num_partitions), std::out_of_range);
}

TEST_F(HashPartition, ZeroPartitions)
{
  fixed_width_column_wrapper<float> floats({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
  fixed_width_column_wrapper<int16_t> integers({1, 2, 3, 4, 5, 6, 7, 8});
  strings_column_wrapper strings({"a", "bb", "ccc", "d", "ee", "fff", "gg", "h"});
  auto input = cudf::table_view({floats, integers, strings});

  auto columns_to_hash = std::vector<cudf::size_type>({2});

  cudf::size_type const num_partitions = 0;
  auto result = cudf::hash_partition(input, columns_to_hash, num_partitions);

  EXPECT_EQ(0, result.size());
}

TEST_F(HashPartition, ZeroRows)
{
  fixed_width_column_wrapper<float> floats({});
  fixed_width_column_wrapper<int16_t> integers({});
  strings_column_wrapper strings({});
  auto input = cudf::table_view({floats, integers, strings});

  auto columns_to_hash = std::vector<cudf::size_type>({2});

  cudf::size_type const num_partitions = 3;
  auto result = cudf::hash_partition(input, columns_to_hash, num_partitions);

  EXPECT_EQ(0, result.size());
}

TEST_F(HashPartition, ZeroColumns)
{
  auto input = cudf::table_view(std::vector<cudf::column_view>{});

  auto columns_to_hash = std::vector<cudf::size_type>({});

  cudf::size_type const num_partitions = 3;
  auto result = cudf::hash_partition(input, columns_to_hash, num_partitions);

  EXPECT_EQ(0, result.size());
}

TEST_F(HashPartition, MixedColumnTypes)
{
  fixed_width_column_wrapper<float> floats({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
  fixed_width_column_wrapper<int16_t> integers({1, 2, 3, 4, 5, 6, 7, 8});
  strings_column_wrapper strings({"a", "bb", "ccc", "d", "ee", "fff", "gg", "h"});
  auto input = cudf::table_view({floats, integers, strings});

  auto columns_to_hash = std::vector<cudf::size_type>({0, 2});

  cudf::size_type const num_partitions = 3;
  auto result1 = cudf::hash_partition(input, columns_to_hash, num_partitions);
  auto result2 = cudf::hash_partition(input, columns_to_hash, num_partitions);

  // Expect output to have size num_partitions
  EXPECT_EQ(static_cast<size_t>(num_partitions), result1.size());
  EXPECT_EQ(result1.size(), result2.size());

  // Expect deterministic result from hashing the same input
  for (cudf::size_type i = 0; i < num_partitions; ++i) {
    expect_tables_equal(result1[i]->view(), result2[i]->view());
  }
}

TEST_F(HashPartition, ColumnsToHash)
{
  fixed_width_column_wrapper<int32_t> to_hash({1, 2, 3, 4, 5, 6});
  fixed_width_column_wrapper<int32_t> first_col({7, 8, 9, 10, 11, 12});
  fixed_width_column_wrapper<int32_t> second_col({13, 14, 15, 16, 17, 18});
  auto first_input = cudf::table_view({to_hash, first_col});
  auto second_input = cudf::table_view({to_hash, second_col});

  auto columns_to_hash = std::vector<cudf::size_type>({0});

  cudf::size_type const num_partitions = 3;
  auto first_result = cudf::hash_partition(first_input, columns_to_hash, num_partitions);
  auto second_result = cudf::hash_partition(second_input, columns_to_hash, num_partitions);

  // Expect output to have size num_partitions
  EXPECT_EQ(static_cast<size_t>(num_partitions), first_result.size());
  EXPECT_EQ(first_result.size(), second_result.size());

  // Expect same result for the hashed columns
  for (cudf::size_type i = 0; i < num_partitions; ++i) {
    expect_columns_equal(first_result[i]->get_column(0).view(),
      second_result[i]->get_column(0).view());
  }
}

template <typename T>
class HashPartitionFixedWidth : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(HashPartitionFixedWidth, cudf::test::FixedWidthTypes);

template <typename T>
void run_fixed_width_test(size_t cols, size_t rows, cudf::size_type num_partitions)
{
  std::vector<fixed_width_column_wrapper<T>> columns(cols);
  std::generate(columns.begin(), columns.end(), [rows]() {
      auto iter = thrust::make_counting_iterator(0);
      return fixed_width_column_wrapper<T>(iter, iter + rows);
    });
  auto input = cudf::table_view(make_view_vector(columns));

  auto columns_to_hash = std::vector<cudf::size_type>(cols);
  std::iota(columns_to_hash.begin(), columns_to_hash.end(), 0);

  auto result1 = cudf::hash_partition(input, columns_to_hash, num_partitions);
  auto result2 = cudf::hash_partition(input, columns_to_hash, num_partitions);

  // Expect output to have size num_partitions
  EXPECT_EQ(static_cast<size_t>(num_partitions), result1.size());
  EXPECT_EQ(result1.size(), result2.size());

  // Expect deterministic result from hashing the same input
  for (cudf::size_type i = 0; i < num_partitions; ++i) {
    expect_tables_equal(result1[i]->view(), result2[i]->view());
  }
}

TYPED_TEST(HashPartitionFixedWidth, MorePartitionsThanRows)
{
  run_fixed_width_test<TypeParam>(5, 10, 50);
}

TYPED_TEST(HashPartitionFixedWidth, LargeInput)
{
  run_fixed_width_test<TypeParam>(10, 1000, 10);
}
