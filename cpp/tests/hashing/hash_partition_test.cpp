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

TEST_F(HashPartition, NoColumnsToHash)
{
  auto floats = fixed_width_column_wrapper<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
  auto integers = fixed_width_column_wrapper<int16_t>({1, 2, 3, 4, 5, 6, 7, 8});
  auto strings = strings_column_wrapper({"a", "bb", "ccc", "d", "ee", "fff", "gg", "h"});
  auto input = cudf::table_view({floats, integers, strings});

  auto columns_to_hash = std::vector<cudf::size_type>();

  cudf::size_type const num_partitions = 3;
  EXPECT_THROW(cudf::hash_partition(input, columns_to_hash, num_partitions), cudf::logic_error);
}

TEST_F(HashPartition, ZeroPartitions)
{
  auto floats = fixed_width_column_wrapper<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
  auto integers = fixed_width_column_wrapper<int16_t>({1, 2, 3, 4, 5, 6, 7, 8});
  auto strings = strings_column_wrapper({"a", "bb", "ccc", "d", "ee", "fff", "gg", "h"});
  auto input = cudf::table_view({floats, integers, strings});

  auto columns_to_hash = std::vector<cudf::size_type>({2});

  cudf::size_type const num_partitions = 0;
  EXPECT_THROW(cudf::hash_partition(input, columns_to_hash, num_partitions), cudf::logic_error);
}

TEST_F(HashPartition, MixedColumnTypes)
{
  auto floats = fixed_width_column_wrapper<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
  auto integers = fixed_width_column_wrapper<int16_t>({1, 2, 3, 4, 5, 6, 7, 8});
  auto strings = strings_column_wrapper({"a", "bb", "ccc", "d", "ee", "fff", "gg", "h"});
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

TYPED_TEST(HashPartitionFixedWidth, ZeroRows)
{
  run_fixed_width_test<TypeParam>(1, 0, 1);
}

TYPED_TEST(HashPartitionFixedWidth, LargeInput)
{
  run_fixed_width_test<TypeParam>(10, 1000, 10);
}
