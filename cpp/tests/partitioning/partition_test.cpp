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
#include <cudf/copying.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/table/table.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/type_lists.hpp>

#include "cudf/sorting.hpp"

template <typename T>
class PartitionTest : public cudf::test::BaseFixture {
  using value_type = cudf::test::GetType<T, 0>;
  using map_type   = cudf::test::GetType<T, 1>;
};

using types =
  cudf::test::CrossProduct<cudf::test::FixedWidthTypes, cudf::test::IntegralTypesNotBool>;

// using types = cudf::test::Types<cudf::test::Types<int32_t, int32_t> >;

TYPED_TEST_CASE(PartitionTest, types);

using cudf::test::fixed_width_column_wrapper;
using cudf::test::strings_column_wrapper;

// Exceptional cases
TYPED_TEST(PartitionTest, EmptyInputs)
{
  using value_type = cudf::test::GetType<TypeParam, 0>;
  using map_type   = cudf::test::GetType<TypeParam, 1>;

  fixed_width_column_wrapper<value_type> empty_column{};
  fixed_width_column_wrapper<map_type> empty_map{};

  auto result = cudf::partition(cudf::table_view{{empty_column}}, empty_map, 10);

  auto result_offsets = result.second;

  EXPECT_TRUE(result_offsets.empty());

  cudf::test::expect_columns_equal(empty_column, result.first->get_column(0));
}

TYPED_TEST(PartitionTest, MapInputSizeMismatch)
{
  using value_type = cudf::test::GetType<TypeParam, 0>;
  using map_type   = cudf::test::GetType<TypeParam, 1>;

  fixed_width_column_wrapper<value_type, int32_t> input({1, 2, 3});
  fixed_width_column_wrapper<map_type> map{1, 2};

  EXPECT_THROW(cudf::partition(cudf::table_view{{input}}, map, 3), cudf::logic_error);
}

TYPED_TEST(PartitionTest, MapWithNullsThrows)
{
  using value_type = cudf::test::GetType<TypeParam, 0>;
  using map_type   = cudf::test::GetType<TypeParam, 1>;

  fixed_width_column_wrapper<value_type, int32_t> input({1, 2, 3});
  fixed_width_column_wrapper<map_type> map{{1, 2}, {1, 0}};

  EXPECT_THROW(cudf::partition(cudf::table_view{{input}}, map, 3), cudf::logic_error);
}

/**
 * @brief Verifies that partitions indicated by `offsets` are equal between
 * `expected` and `actual`.
 *
 * The order of rows within each partition may be different, so each partition
 * is first sorted before being compared for equality.
 *
 */
void expect_equal_partitions(cudf::table_view expected,
                             cudf::table_view actual,
                             std::vector<cudf::size_type> const& offsets)
{
  // Need to convert partition offsets into split points by dropping the first
  // and last element
  std::vector<cudf::size_type> split_points;
  std::copy(std::next(offsets.begin()), std::prev(offsets.end()), std::back_inserter(split_points));

  // Split the partitions, sort each partition, then compare for equality
  auto actual_split   = cudf::split(actual, split_points);
  auto expected_split = cudf::split(expected, split_points);
  std::equal(expected_split.begin(),
             expected_split.end(),
             actual_split.begin(),
             [](cudf::table_view expected, cudf::table_view actual) {
               auto sorted_expected = cudf::sort(expected);
               auto sorted_actual   = cudf::sort(actual);
               cudf::test::expect_tables_equal(*sorted_expected, *sorted_actual);
               return true;
             });
}

void run_partition_test(cudf::table_view table_to_partition,
                        cudf::column_view partition_map,
                        cudf::size_type num_partitions,
                        cudf::table_view expected_partitioned_table,
                        std::vector<cudf::size_type> const& expected_offsets)
{
  auto result = cudf::partition(table_to_partition, partition_map, num_partitions);
  auto const& actual_partitioned_table = result.first;
  auto const& actual_offsets           = result.second;
  EXPECT_EQ(actual_offsets, expected_offsets);

  expect_equal_partitions(expected_partitioned_table, *actual_partitioned_table, expected_offsets);
}

// Normal cases
TYPED_TEST(PartitionTest, Identity)
{
  using value_type = cudf::test::GetType<TypeParam, 0>;
  using map_type   = cudf::test::GetType<TypeParam, 1>;

  fixed_width_column_wrapper<value_type, int32_t> first({0, 1, 2, 3, 4, 5});
  strings_column_wrapper strings{"this", "is", "a", "column", "of", "strings"};
  auto table_to_partition = cudf::table_view{{first, strings}};

  fixed_width_column_wrapper<map_type> map{0, 1, 2, 3, 4, 5};

  std::vector<cudf::size_type> expected_offsets{0, 1, 2, 3, 4, 5, 6};

  run_partition_test(table_to_partition, map, 6, table_to_partition, expected_offsets);
}

TYPED_TEST(PartitionTest, Reverse)
{
  using value_type = cudf::test::GetType<TypeParam, 0>;
  using map_type   = cudf::test::GetType<TypeParam, 1>;

  fixed_width_column_wrapper<value_type, int32_t> first({0, 1, 3, 7, 5, 13});
  strings_column_wrapper strings{"this", "is", "a", "column", "of", "strings"};
  auto table_to_partition = cudf::table_view{{first, strings}};

  fixed_width_column_wrapper<map_type> map{5, 4, 3, 2, 1, 0};

  std::vector<cudf::size_type> expected_offsets{0, 1, 2, 3, 4, 5, 6};

  fixed_width_column_wrapper<value_type, int32_t> expected_first({13, 5, 7, 3, 1, 0});
  strings_column_wrapper expected_strings{"strings", "of", "column", "a", "is", "this"};
  auto expected_partitioned_table = cudf::table_view{{expected_first, expected_strings}};

  run_partition_test(table_to_partition, map, 6, expected_partitioned_table, expected_offsets);
}

TYPED_TEST(PartitionTest, SinglePartition)
{
  using value_type = cudf::test::GetType<TypeParam, 0>;
  using map_type   = cudf::test::GetType<TypeParam, 1>;

  fixed_width_column_wrapper<value_type, int32_t> first({0, 1, 3, 7, 5, 13});
  strings_column_wrapper strings{"this", "is", "a", "column", "of", "strings"};
  auto table_to_partition = cudf::table_view{{first, strings}};

  fixed_width_column_wrapper<map_type> map{0, 0, 0, 0, 0, 0};

  std::vector<cudf::size_type> expected_offsets{0, 6};

  fixed_width_column_wrapper<value_type, int32_t> expected_first({13, 5, 7, 3, 1, 0});
  strings_column_wrapper expected_strings{"strings", "of", "column", "a", "is", "this"};
  auto expected_partitioned_table = cudf::table_view{{expected_first, expected_strings}};

  run_partition_test(table_to_partition, map, 1, expected_partitioned_table, expected_offsets);
}

TYPED_TEST(PartitionTest, EmptyPartitions)
{
  using value_type = cudf::test::GetType<TypeParam, 0>;
  using map_type   = cudf::test::GetType<TypeParam, 1>;

  fixed_width_column_wrapper<value_type, int32_t> first({0, 1, 3, 7, 5, 13});
  strings_column_wrapper strings{"this", "is", "a", "column", "of", "strings"};
  auto table_to_partition = cudf::table_view{{first, strings}};

  fixed_width_column_wrapper<map_type> map{2, 2, 0, 0, 4, 4};

  std::vector<cudf::size_type> expected_offsets{0, 2, 2, 4, 4, 6};

  fixed_width_column_wrapper<value_type, int32_t> expected_first({3, 7, 0, 1, 5, 13});
  strings_column_wrapper expected_strings{"a", "column", "this", "is", "of", "strings"};
  auto expected_partitioned_table = cudf::table_view{{expected_first, expected_strings}};

  run_partition_test(table_to_partition, map, 5, expected_partitioned_table, expected_offsets);
}
