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
#include <cudf/partitioning.hpp>
#include <cudf/table/table.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/type_lists.hpp>

template <typename T>
class PartitionTest : public cudf::test::BaseFixture {
  using value_type = cudf::test::GetType<T, 0>;
  using map_type = cudf::test::GetType<T, 1>;
};

using types = cudf::test::CrossProduct<cudf::test::FixedWidthTypes,
                                       cudf::test::IntegralTypes>;

// using types = cudf::test::Types<cudf::test::Types<int32_t, int32_t> >;

TYPED_TEST_CASE(PartitionTest, types);

using cudf::test::fixed_width_column_wrapper;
using cudf::test::strings_column_wrapper;

// Exceptional cases
TYPED_TEST(PartitionTest, EmptyInputs) {
  using value_type = cudf::test::GetType<TypeParam, 0>;
  using map_type = cudf::test::GetType<TypeParam, 1>;

  fixed_width_column_wrapper<value_type> empty_column{};
  fixed_width_column_wrapper<map_type> empty_map{};

  auto result = cudf::experimental::partition(cudf::table_view{{empty_column}},
                                              empty_map, 10);

  auto result_offsets = result.second;

  EXPECT_TRUE(result_offsets.empty());

  cudf::test::expect_columns_equal(empty_column, result.first->get_column(0));
}

TYPED_TEST(PartitionTest, MapInputSizeMismatch) {
  using value_type = cudf::test::GetType<TypeParam, 0>;
  using map_type = cudf::test::GetType<TypeParam, 1>;

  fixed_width_column_wrapper<value_type> input{1, 2, 3};
  fixed_width_column_wrapper<map_type> map{1, 2};

  EXPECT_THROW(cudf::experimental::partition(cudf::table_view{{input}}, map, 3),
               cudf::logic_error);
}

TYPED_TEST(PartitionTest, MapWithNullsThrows) {
  using value_type = cudf::test::GetType<TypeParam, 0>;
  using map_type = cudf::test::GetType<TypeParam, 1>;

  fixed_width_column_wrapper<value_type> input{1, 2, 3};
  fixed_width_column_wrapper<map_type> map{{1, 2}, {1, 0}};

  EXPECT_THROW(cudf::experimental::partition(cudf::table_view{{input}}, map, 3),
               cudf::logic_error);
}

void run_partition_test(cudf::table_view table_to_partition,
                        cudf::column_view partition_map,
                        cudf::size_type num_partitions,
                        cudf::table_view expected_partitioned_table,
                        std::vector<cudf::size_type> const& expected_offsets) {
  auto result = cudf::experimental::partition(table_to_partition, partition_map,
                                              num_partitions);
  auto const& actual_partitioned_table = result.first;
  auto const& actual_offsets = result.second;
  EXPECT_EQ(actual_offsets, expected_offsets);

  cudf::test::expect_tables_equal(*actual_partitioned_table,
                                  expected_partitioned_table);
}

// Normal cases
TYPED_TEST(PartitionTest, Identity) {
  using value_type = cudf::test::GetType<TypeParam, 0>;
  using map_type = cudf::test::GetType<TypeParam, 1>;

  fixed_width_column_wrapper<value_type> first{0, 1, 2, 3, 4, 5};
  strings_column_wrapper strings{"this", "is", "a", "column", "of", "strings"};
  auto table_to_partition = cudf::table_view{{first, strings}};

  fixed_width_column_wrapper<map_type> map{0, 1, 2, 3, 4, 5};

  std::vector<cudf::size_type> expected_offsets{0, 1, 2, 3, 4, 5, 6};

  run_partition_test(table_to_partition, map, 6, table_to_partition,
                     expected_offsets);
}

TYPED_TEST(PartitionTest, Reverse) {
  using value_type = cudf::test::GetType<TypeParam, 0>;
  using map_type = cudf::test::GetType<TypeParam, 1>;

  fixed_width_column_wrapper<value_type> first{0, 1, 3, 7, 5, 13};
  strings_column_wrapper strings{"this", "is", "a", "column", "of", "strings"};
  auto table_to_partition = cudf::table_view{{first, strings}};

  fixed_width_column_wrapper<map_type> map{5, 4, 3, 2, 1, 0};

  std::vector<cudf::size_type> expected_offsets{0, 1, 2, 3, 4, 5, 6};

  fixed_width_column_wrapper<value_type> expected_first{13, 5, 7, 3, 1, 0};
  strings_column_wrapper expected_strings{"strings", "of", "column",
                                          "a",       "is", "this"};
  auto expected_partitioned_table =
      cudf::table_view{{expected_first, expected_strings}};

  run_partition_test(table_to_partition, map, 6, expected_partitioned_table,
                     expected_offsets);
}