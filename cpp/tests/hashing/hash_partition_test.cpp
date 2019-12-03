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

class HashPartition : public cudf::test::BaseFixture {};

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
  EXPECT_EQ(size_t{num_partitions}, result1.size());
  EXPECT_EQ(size_t{num_partitions}, result2.size());

  // Expect deterministic result from hashing the same input
  for (cudf::size_type i = 0; i < num_partitions; ++i) {
    expect_tables_equal(result1[i]->view(), result2[i]->view());
  }
}

template <typename T>
class HashPartitionFixedWidth : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(HashPartitionFixedWidth, cudf::test::FixedWidthTypes);

TYPED_TEST(HashPartitionFixedWidth, MorePartitionsThanRows)
{
  auto first = fixed_width_column_wrapper<TypeParam>({1, 2, 3, 4, 5, 6});
  auto second = fixed_width_column_wrapper<TypeParam>({7, 8, 9, 10, 11, 12});
  auto input = cudf::table_view({first, second});

  auto columns_to_hash = std::vector<cudf::size_type>({0, 1});

  cudf::size_type const num_partitions = 10;
  auto result1 = cudf::hash_partition(input, columns_to_hash, num_partitions);
  auto result2 = cudf::hash_partition(input, columns_to_hash, num_partitions);

  // Expect output to have size num_partitions
  EXPECT_EQ(size_t{num_partitions}, result1.size());
  EXPECT_EQ(size_t{num_partitions}, result2.size());

  // Expect deterministic result from hashing the same input
  for (cudf::size_type i = 0; i < num_partitions; ++i) {
    expect_tables_equal(result1[i]->view(), result2[i]->view());
  }
}
