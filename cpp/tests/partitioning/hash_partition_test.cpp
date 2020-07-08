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
#include <cudf/partitioning.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/type_lists.hpp>

using cudf::test::expect_columns_equal;
using cudf::test::expect_table_properties_equal;
using cudf::test::expect_tables_equal;
using cudf::test::fixed_width_column_wrapper;
using cudf::test::strings_column_wrapper;

// Transform vector of column wrappers to vector of column views
template <typename T>
auto make_view_vector(std::vector<T> const& columns)
{
  std::vector<cudf::column_view> views(columns.size());
  std::transform(columns.begin(), columns.end(), views.begin(), [](auto const& col) {
    return static_cast<cudf::column_view>(col);
  });
  return views;
}

class HashPartition : public cudf::test::BaseFixture {
};

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
  std::unique_ptr<cudf::table> output;
  std::vector<cudf::size_type> offsets;
  std::tie(output, offsets) = cudf::hash_partition(input, columns_to_hash, num_partitions);

  // Expect empty table with same number of columns and zero partitions
  EXPECT_EQ(input.num_columns(), output->num_columns());
  EXPECT_EQ(0, output->num_rows());
  EXPECT_EQ(0, offsets.size());
}

TEST_F(HashPartition, ZeroRows)
{
  fixed_width_column_wrapper<float> floats({});
  fixed_width_column_wrapper<int16_t> integers({});
  strings_column_wrapper strings({});
  auto input = cudf::table_view({floats, integers, strings});

  auto columns_to_hash = std::vector<cudf::size_type>({2});

  cudf::size_type const num_partitions = 3;
  std::unique_ptr<cudf::table> output;
  std::vector<cudf::size_type> offsets;
  std::tie(output, offsets) = cudf::hash_partition(input, columns_to_hash, num_partitions);

  // Expect empty table with same number of columns and zero partitions
  EXPECT_EQ(input.num_columns(), output->num_columns());
  EXPECT_EQ(0, output->num_rows());
  EXPECT_EQ(0, offsets.size());
}

TEST_F(HashPartition, ZeroColumns)
{
  auto input = cudf::table_view(std::vector<cudf::column_view>{});

  auto columns_to_hash = std::vector<cudf::size_type>({});

  cudf::size_type const num_partitions = 3;
  std::unique_ptr<cudf::table> output;
  std::vector<cudf::size_type> offsets;
  std::tie(output, offsets) = cudf::hash_partition(input, columns_to_hash, num_partitions);

  // Expect empty table with same number of columns and zero partitions
  EXPECT_EQ(input.num_columns(), output->num_columns());
  EXPECT_EQ(0, output->num_rows());
  EXPECT_EQ(0, offsets.size());
}

TEST_F(HashPartition, MixedColumnTypes)
{
  fixed_width_column_wrapper<float> floats({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
  fixed_width_column_wrapper<int16_t> integers({1, 2, 3, 4, 5, 6, 7, 8});
  strings_column_wrapper strings({"a", "bb", "ccc", "d", "ee", "fff", "gg", "h"});
  auto input = cudf::table_view({floats, integers, strings});

  auto columns_to_hash = std::vector<cudf::size_type>({0, 2});

  cudf::size_type const num_partitions = 3;
  std::unique_ptr<cudf::table> output1, output2;
  std::vector<cudf::size_type> offsets1, offsets2;
  std::tie(output1, offsets1) = cudf::hash_partition(input, columns_to_hash, num_partitions);
  std::tie(output2, offsets2) = cudf::hash_partition(input, columns_to_hash, num_partitions);

  // Expect output to have size num_partitions
  EXPECT_EQ(static_cast<size_t>(num_partitions), offsets1.size());
  EXPECT_EQ(offsets1.size(), offsets2.size());

  // Expect output to have same shape as input
  expect_table_properties_equal(input, output1->view());

  // Expect deterministic result from hashing the same input
  expect_tables_equal(output1->view(), output2->view());
}

TEST_F(HashPartition, NullableStrings)
{
  strings_column_wrapper strings({"a", "bb", "ccc", "d"}, {1, 1, 1, 1});
  cudf::table_view input({strings});

  std::vector<cudf::size_type> const columns_to_hash({0});
  cudf::size_type const num_partitions = 3;

  std::unique_ptr<cudf::table> result;
  std::vector<cudf::size_type> offsets;
  std::tie(result, offsets) = cudf::hash_partition(input, columns_to_hash, num_partitions);

  auto const& col = result->get_column(0);
  EXPECT_EQ(0, col.null_count());
}

TEST_F(HashPartition, ColumnsToHash)
{
  fixed_width_column_wrapper<int32_t> to_hash({1, 2, 3, 4, 5, 6});
  fixed_width_column_wrapper<int32_t> first_col({7, 8, 9, 10, 11, 12});
  fixed_width_column_wrapper<int32_t> second_col({13, 14, 15, 16, 17, 18});
  auto first_input  = cudf::table_view({to_hash, first_col});
  auto second_input = cudf::table_view({to_hash, second_col});

  auto columns_to_hash = std::vector<cudf::size_type>({0});

  cudf::size_type const num_partitions = 3;
  std::unique_ptr<cudf::table> first_result, second_result;
  std::vector<cudf::size_type> first_offsets, second_offsets;
  std::tie(first_result, first_offsets) =
    cudf::hash_partition(first_input, columns_to_hash, num_partitions);
  std::tie(second_result, second_offsets) =
    cudf::hash_partition(second_input, columns_to_hash, num_partitions);

  // Expect offsets to be equal and num_partitions in length
  EXPECT_EQ(static_cast<size_t>(num_partitions), first_offsets.size());
  EXPECT_TRUE(std::equal(
    first_offsets.begin(), first_offsets.end(), second_offsets.begin(), second_offsets.end()));

  // Expect same result for the hashed columns
  expect_columns_equal(first_result->get_column(0).view(), second_result->get_column(0).view());
}

template <typename T>
class HashPartitionFixedWidth : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(HashPartitionFixedWidth, cudf::test::FixedWidthTypes);

TYPED_TEST(HashPartitionFixedWidth, NullableFixedWidth)
{
  fixed_width_column_wrapper<TypeParam> fixed({1, 2, 3, 4}, {1, 1, 1, 1});
  cudf::table_view input({fixed});

  std::vector<cudf::size_type> const columns_to_hash({0});
  cudf::size_type const num_partitions = 3;

  std::unique_ptr<cudf::table> result;
  std::vector<cudf::size_type> offsets;
  std::tie(result, offsets) = cudf::hash_partition(input, columns_to_hash, num_partitions);

  auto const& col = result->get_column(0);
  EXPECT_EQ(0, col.null_count());
}

template <typename T>
void run_fixed_width_test(size_t cols,
                          size_t rows,
                          cudf::size_type num_partitions,
                          bool has_nulls = false)
{
  std::vector<fixed_width_column_wrapper<T>> columns(cols);
  if (has_nulls) {
    std::generate(columns.begin(), columns.end(), [rows]() {
      auto iter   = thrust::make_counting_iterator(0);
      auto valids = thrust::make_transform_iterator(iter, [](auto i) { return i % 4 != 0; });
      return fixed_width_column_wrapper<T>(iter, iter + rows, valids);
    });
  } else {
    std::generate(columns.begin(), columns.end(), [rows]() {
      auto iter = thrust::make_counting_iterator(0);
      return fixed_width_column_wrapper<T>(iter, iter + rows);
    });
  }
  auto input = cudf::table_view(make_view_vector(columns));

  auto columns_to_hash = std::vector<cudf::size_type>(cols);
  std::iota(columns_to_hash.begin(), columns_to_hash.end(), 0);

  std::unique_ptr<cudf::table> output1, output2;
  std::vector<cudf::size_type> offsets1, offsets2;
  std::tie(output1, offsets1) = cudf::hash_partition(input, columns_to_hash, num_partitions);
  std::tie(output2, offsets2) = cudf::hash_partition(input, columns_to_hash, num_partitions);

  // Expect output to have size num_partitions
  EXPECT_EQ(static_cast<size_t>(num_partitions), offsets1.size());
  EXPECT_TRUE(std::equal(offsets1.begin(), offsets1.end(), offsets2.begin()));

  // Expect output to have same shape as input
  expect_table_properties_equal(input, output1->view());
  expect_table_properties_equal(output1->view(), output2->view());

  // Compute number of rows in each partition
  EXPECT_EQ(0, offsets1[0]);
  offsets1.push_back(rows);
  std::adjacent_difference(offsets1.begin() + 1, offsets1.end(), offsets1.begin() + 1);

  // Compute the partition number for each row
  cudf::size_type partition = 0;
  thrust::host_vector<cudf::size_type> partitions;
  std::for_each(offsets1.begin() + 1, offsets1.end(), [&](cudf::size_type const& count) {
    std::fill_n(std::back_inserter(partitions), count, partition++);
  });

  // Make a table view of the partition numbers
  constexpr cudf::data_type dtype{cudf::type_id::INT32};
  rmm::device_vector<cudf::size_type> d_partitions(partitions);
  cudf::column_view partitions_col(dtype, rows, d_partitions.data().get());
  cudf::table_view partitions_table({partitions_col});

  // Sort partition numbers by the corresponding row hashes of each output
  auto hash1 = cudf::hash(output1->view());
  cudf::table_view hash1_table({hash1->view()});
  auto sorted_partitions1 = cudf::sort_by_key(partitions_table, hash1_table);

  auto hash2 = cudf::hash(output2->view());
  cudf::table_view hash2_table({hash2->view()});
  auto sorted_partitions2 = cudf::sort_by_key(partitions_table, hash2_table);

  // After sorting by row hashes, the corresponding partition numbers should be
  // equal
  expect_tables_equal(sorted_partitions1->view(), sorted_partitions2->view());
}

TYPED_TEST(HashPartitionFixedWidth, MorePartitionsThanRows)
{
  run_fixed_width_test<TypeParam>(5, 10, 50);
}

TYPED_TEST(HashPartitionFixedWidth, LargeInput) { run_fixed_width_test<TypeParam>(10, 1000, 10); }

TYPED_TEST(HashPartitionFixedWidth, HasNulls)
{
  run_fixed_width_test<TypeParam>(10, 1000, 10, true);
}

CUDF_TEST_PROGRAM_MAIN()
