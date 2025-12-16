/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/table_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/hashing.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <numeric>

using cudf::test::fixed_width_column_wrapper;
using cudf::test::strings_column_wrapper;
using structs_col = cudf::test::structs_column_wrapper;

using cudf::test::iterators::null_at;
using cudf::test::iterators::nulls_at;

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
  auto [output, offsets] = cudf::hash_partition(input, columns_to_hash, num_partitions);

  // Expect empty table with same number of columns and zero partitions
  EXPECT_EQ(input.num_columns(), output->num_columns());
  EXPECT_EQ(0, output->num_rows());
  EXPECT_EQ(std::size_t{num_partitions}, offsets.size());
}

TEST_F(HashPartition, ZeroRows)
{
  fixed_width_column_wrapper<float> floats({});
  fixed_width_column_wrapper<int16_t> integers({});
  strings_column_wrapper strings;
  auto input = cudf::table_view({floats, integers, strings});

  auto columns_to_hash = std::vector<cudf::size_type>({2});

  cudf::size_type const num_partitions = 3;
  auto [output, offsets] = cudf::hash_partition(input, columns_to_hash, num_partitions);

  // Expect empty table with same number of columns and same number of partitions
  EXPECT_EQ(input.num_columns(), output->num_columns());
  EXPECT_EQ(0, output->num_rows());
  EXPECT_EQ(std::size_t{num_partitions}, offsets.size());
}

TEST_F(HashPartition, ZeroColumns)
{
  auto input = cudf::table_view(std::vector<cudf::column_view>{});

  auto columns_to_hash = std::vector<cudf::size_type>({});

  cudf::size_type const num_partitions = 3;
  auto [output, offsets] = cudf::hash_partition(input, columns_to_hash, num_partitions);

  // Expect empty table with same number of columns and same number of partitions
  EXPECT_EQ(input.num_columns(), output->num_columns());
  EXPECT_EQ(0, output->num_rows());
  EXPECT_EQ(std::size_t{num_partitions}, offsets.size());
}

TEST_F(HashPartition, MixedColumnTypes)
{
  fixed_width_column_wrapper<float> floats({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
  fixed_width_column_wrapper<int16_t> integers({1, 2, 3, 4, 5, 6, 7, 8});
  strings_column_wrapper strings({"a", "bb", "ccc", "d", "ee", "fff", "gg", "h"});
  auto input = cudf::table_view({floats, integers, strings});

  auto columns_to_hash = std::vector<cudf::size_type>({0, 2});

  cudf::size_type const num_partitions = 3;
  auto [output1, offsets1] = cudf::hash_partition(input, columns_to_hash, num_partitions);
  auto [output2, offsets2] = cudf::hash_partition(input, columns_to_hash, num_partitions);

  // Expect output to have size num_partitions
  EXPECT_EQ(static_cast<size_t>(num_partitions), offsets1.size());
  EXPECT_EQ(offsets1.size(), offsets2.size());

  // Expect output to have same shape as input
  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(input, output1->view());

  // Expect deterministic result from hashing the same input
  CUDF_TEST_EXPECT_TABLES_EQUAL(output1->view(), output2->view());
}

TEST_F(HashPartition, NullableStrings)
{
  strings_column_wrapper strings({"a", "bb", "ccc", "d"}, {true, true, true, true});
  cudf::table_view input({strings});

  std::vector<cudf::size_type> const columns_to_hash({0});
  cudf::size_type const num_partitions = 3;

  auto [result, offsets] = cudf::hash_partition(input, columns_to_hash, num_partitions);

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
  auto [first_result, first_offsets] =
    cudf::hash_partition(first_input, columns_to_hash, num_partitions);
  auto [second_result, second_offsets] =
    cudf::hash_partition(second_input, columns_to_hash, num_partitions);

  // Expect offsets to be equal and num_partitions in length
  EXPECT_EQ(static_cast<size_t>(num_partitions), first_offsets.size());
  EXPECT_TRUE(std::equal(
    first_offsets.begin(), first_offsets.end(), second_offsets.begin(), second_offsets.end()));

  // Expect same result for the hashed columns
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(first_result->get_column(0).view(),
                                 second_result->get_column(0).view());
}

TEST_F(HashPartition, IdentityHashFailure)
{
  fixed_width_column_wrapper<float> floats({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
  fixed_width_column_wrapper<int16_t> integers({1, 2, 3, 4, 5, 6, 7, 8});
  strings_column_wrapper strings({"a", "bb", "ccc", "d", "ee", "fff", "gg", "h"});
  auto input = cudf::table_view({floats, integers, strings});

  auto columns_to_hash = std::vector<cudf::size_type>({2});

  cudf::size_type const num_partitions = 3;
  EXPECT_THROW(
    cudf::hash_partition(input, columns_to_hash, num_partitions, cudf::hash_id::HASH_IDENTITY),
    cudf::logic_error);
}

TEST_F(HashPartition, CustomSeedValue)
{
  fixed_width_column_wrapper<float> floats({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
  fixed_width_column_wrapper<int16_t> integers({1, 2, 3, 4, 5, 6, 7, 8});
  strings_column_wrapper strings({"a", "bb", "ccc", "d", "ee", "fff", "gg", "h"});
  auto input = cudf::table_view({floats, integers, strings});

  auto columns_to_hash = std::vector<cudf::size_type>({0, 2});

  cudf::size_type const num_partitions = 3;
  auto [output1, offsets1]             = cudf::hash_partition(
    input, columns_to_hash, num_partitions, cudf::hash_id::HASH_MURMUR3, 12345);
  auto [output2, offsets2] = cudf::hash_partition(
    input, columns_to_hash, num_partitions, cudf::hash_id::HASH_MURMUR3, 12345);

  // Expect output to have size num_partitions
  EXPECT_EQ(static_cast<size_t>(num_partitions), offsets1.size());
  EXPECT_EQ(offsets1.size(), offsets2.size());

  // Expect output to have same shape as input
  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(input, output1->view());

  // Expect deterministic result from hashing the same input
  CUDF_TEST_EXPECT_TABLES_EQUAL(output1->view(), output2->view());
}

template <typename T>
class HashPartitionFixedWidth : public cudf::test::BaseFixture {};

TYPED_TEST_SUITE(HashPartitionFixedWidth, cudf::test::FixedWidthTypesWithoutFixedPoint);

TYPED_TEST(HashPartitionFixedWidth, NullableFixedWidth)
{
  fixed_width_column_wrapper<TypeParam, int32_t> fixed({1, 2, 3, 4}, {1, 1, 1, 1});
  cudf::table_view input({fixed});

  std::vector<cudf::size_type> const columns_to_hash({0});
  cudf::size_type const num_partitions = 3;

  auto [result, offsets] = cudf::hash_partition(input, columns_to_hash, num_partitions);

  auto const& col = result->get_column(0);
  EXPECT_EQ(0, col.null_count());
}

template <typename T>
void run_fixed_width_test(size_t cols,
                          size_t rows,
                          cudf::size_type num_partitions,
                          cudf::hash_id hash_function,
                          bool has_nulls = false)
{
  std::vector<fixed_width_column_wrapper<T, int32_t>> columns;
  columns.reserve(cols);
  if (has_nulls) {
    std::generate_n(std::back_inserter(columns), cols, [rows]() {
      auto iter   = thrust::make_counting_iterator(0);
      auto valids = thrust::make_transform_iterator(iter, [](auto i) { return i % 4 != 0; });
      return fixed_width_column_wrapper<T, int32_t>(iter, iter + rows, valids);
    });
  } else {
    std::generate_n(std::back_inserter(columns), cols, [rows]() {
      auto iter = thrust::make_counting_iterator(0);
      return fixed_width_column_wrapper<T, int32_t>(iter, iter + rows);
    });
  }
  auto input = cudf::table_view(make_view_vector(columns));

  auto columns_to_hash = std::vector<cudf::size_type>(cols);
  std::iota(columns_to_hash.begin(), columns_to_hash.end(), 0);

  auto [output1, offsets1] = cudf::hash_partition(input, columns_to_hash, num_partitions);
  auto [output2, offsets2] = cudf::hash_partition(input, columns_to_hash, num_partitions);

  // Expect output to have size num_partitions
  EXPECT_EQ(static_cast<size_t>(num_partitions), offsets1.size());
  EXPECT_TRUE(std::equal(offsets1.begin(), offsets1.end(), offsets2.begin()));

  // Expect output to have same shape as input
  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(input, output1->view());
  CUDF_TEST_EXPECT_TABLE_PROPERTIES_EQUAL(output1->view(), output2->view());

  // Compute number of rows in each partition
  EXPECT_EQ(0, offsets1[0]);
  offsets1.push_back(rows);
  std::adjacent_difference(offsets1.begin() + 1, offsets1.end(), offsets1.begin() + 1);

  // Compute the partition number for each row
  cudf::size_type partition = 0;
  std::vector<cudf::size_type> partitions;
  std::for_each(offsets1.begin() + 1, offsets1.end(), [&](cudf::size_type const& count) {
    std::fill_n(std::back_inserter(partitions), count, partition++);
  });

  // Make a table view of the partition numbers
  constexpr cudf::data_type dtype{cudf::type_id::INT32};
  auto d_partitions = cudf::detail::make_device_uvector(
    partitions, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  cudf::column_view partitions_col(dtype, rows, d_partitions.data(), nullptr, 0);
  cudf::table_view partitions_table({partitions_col});

  // Sort partition numbers by the corresponding row hashes of each output
  auto hash1 = cudf::hashing::murmurhash3_x86_32(output1->view());
  cudf::table_view hash1_table({hash1->view()});
  auto sorted_partitions1 = cudf::sort_by_key(partitions_table, hash1_table);

  auto hash2 = cudf::hashing::murmurhash3_x86_32(output2->view());
  cudf::table_view hash2_table({hash2->view()});
  auto sorted_partitions2 = cudf::sort_by_key(partitions_table, hash2_table);

  // After sorting by row hashes, the corresponding partition numbers should be
  // equal
  CUDF_TEST_EXPECT_TABLES_EQUAL(sorted_partitions1->view(), sorted_partitions2->view());
}

TYPED_TEST(HashPartitionFixedWidth, MorePartitionsThanRows)
{
  run_fixed_width_test<TypeParam>(5, 10, 50, cudf::hash_id::HASH_MURMUR3);
  run_fixed_width_test<TypeParam>(5, 10, 50, cudf::hash_id::HASH_IDENTITY);
}

TYPED_TEST(HashPartitionFixedWidth, LargeInput)
{
  run_fixed_width_test<TypeParam>(10, 1000, 10, cudf::hash_id::HASH_MURMUR3);
  run_fixed_width_test<TypeParam>(10, 1000, 10, cudf::hash_id::HASH_IDENTITY);
}

TYPED_TEST(HashPartitionFixedWidth, HasNulls)
{
  run_fixed_width_test<TypeParam>(10, 1000, 10, cudf::hash_id::HASH_MURMUR3, true);
  run_fixed_width_test<TypeParam>(10, 1000, 10, cudf::hash_id::HASH_IDENTITY, true);
}

TEST_F(HashPartition, FixedPointColumnsToHash)
{
  fixed_width_column_wrapper<int32_t> to_hash({1});
  cudf::test::fixed_point_column_wrapper<int64_t> first_col({7}, numeric::scale_type{-1});
  cudf::test::fixed_point_column_wrapper<__int128_t> second_col({77}, numeric::scale_type{0});

  auto input = cudf::table_view({to_hash, first_col, second_col});

  auto columns_to_hash = std::vector<cudf::size_type>({0});

  cudf::size_type const num_partitions = 1;
  auto [result, offsets] = cudf::hash_partition(input, columns_to_hash, num_partitions);

  // Expect offsets to be equal and num_partitions in length
  EXPECT_EQ(static_cast<size_t>(num_partitions), offsets.size());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->get_column(0).view(), input.column(0));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->get_column(1).view(), input.column(1));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result->get_column(2).view(), input.column(2));
}

TEST_F(HashPartition, ListWithNulls)
{
  using lcw = cudf::test::lists_column_wrapper<int32_t>;

  lcw to_hash{{{1}, {2, 2}, {0}, {1}, {2, 2}, {0}}, nulls_at({2, 5})};
  fixed_width_column_wrapper<int32_t> first_col({7, 8, 9, 10, 11, 12});
  fixed_width_column_wrapper<int32_t> second_col({13, 14, 15, 16, 17, 18});
  auto const first_input  = cudf::table_view({to_hash, first_col});
  auto const second_input = cudf::table_view({to_hash, second_col});

  auto const columns_to_hash = std::vector<cudf::size_type>({0});

  cudf::size_type const num_partitions = 3;
  auto const [first_result, first_offsets] =
    cudf::hash_partition(first_input, columns_to_hash, num_partitions);
  auto const [second_result, second_offsets] =
    cudf::hash_partition(second_input, columns_to_hash, num_partitions);

  // Expect offsets to be equal and num_partitions in length
  EXPECT_EQ(static_cast<size_t>(num_partitions), first_offsets.size());
  EXPECT_TRUE(std::equal(
    first_offsets.begin(), first_offsets.end(), second_offsets.begin(), second_offsets.end()));

  // Expect same result for the hashed columns
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(first_result->get_column(0).view(),
                                 second_result->get_column(0).view());
}

TEST_F(HashPartition, StructofStructWithNulls)
{
  //  +-----------------+
  //  |  s1{s2{a,b}, c} |
  //  +-----------------+
  // 0 |  { {1, 1}, 5}  |
  // 1 |  { {1, 1}, 5}  |
  // 2 |  { {1, 2}, 4}  |
  // 3 |  Null          |
  // 4 |  { Null,   4}  |
  // 5 |  { Null,   4}  |

  auto const to_hash = [&] {
    auto a  = fixed_width_column_wrapper<int32_t>{1, 1, 1, 0, 0, 0};
    auto b  = fixed_width_column_wrapper<int32_t>{1, 1, 2, 0, 0, 0};
    auto s2 = structs_col{{a, b}, nulls_at({4, 5})};

    auto c = fixed_width_column_wrapper<int32_t>{5, 5, 4, 0, 4, 4};
    return structs_col({s2, c}, null_at(3));
  }();

  fixed_width_column_wrapper<int32_t> first_col({7, 8, 9, 10, 11, 12});
  fixed_width_column_wrapper<int32_t> second_col({13, 14, 15, 16, 17, 18});
  auto const first_input  = cudf::table_view({to_hash, first_col});
  auto const second_input = cudf::table_view({to_hash, second_col});

  auto const columns_to_hash = std::vector<cudf::size_type>({0});

  cudf::size_type const num_partitions = 3;
  auto const [first_result, first_offsets] =
    cudf::hash_partition(first_input, columns_to_hash, num_partitions);
  auto const [second_result, second_offsets] =
    cudf::hash_partition(second_input, columns_to_hash, num_partitions);

  // Expect offsets to be equal and num_partitions in length
  EXPECT_EQ(static_cast<size_t>(num_partitions), first_offsets.size());
  EXPECT_TRUE(std::equal(
    first_offsets.begin(), first_offsets.end(), second_offsets.begin(), second_offsets.end()));

  // Expect same result for the hashed columns
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(first_result->get_column(0).view(),
                                 second_result->get_column(0).view());
}

// Tests for hash_partition_indices
class HashPartitionIndices : public cudf::test::BaseFixture {};

TEST_F(HashPartitionIndices, ZeroPartitions)
{
  fixed_width_column_wrapper<float> floats({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
  fixed_width_column_wrapper<int16_t> integers({1, 2, 3, 4, 5, 6, 7, 8});
  auto input = cudf::table_view({floats, integers});

  auto columns_to_hash = std::vector<cudf::size_type>({0});

  cudf::size_type const num_partitions = 0;
  auto result = cudf::hash_partition_indices(input, columns_to_hash, num_partitions);

  EXPECT_EQ(std::size_t{0}, result.size());
}

TEST_F(HashPartitionIndices, ZeroRows)
{
  fixed_width_column_wrapper<float> floats({});
  fixed_width_column_wrapper<int16_t> integers({});
  auto input = cudf::table_view({floats, integers});

  auto columns_to_hash = std::vector<cudf::size_type>({0});

  cudf::size_type const num_partitions = 3;
  auto result = cudf::hash_partition_indices(input, columns_to_hash, num_partitions);

  EXPECT_EQ(std::size_t{num_partitions}, result.size());
  for (auto const& col : result) {
    EXPECT_EQ(0, col->size());
  }
}

TEST_F(HashPartitionIndices, ZeroColumns)
{
  auto input = cudf::table_view(std::vector<cudf::column_view>{});

  auto columns_to_hash = std::vector<cudf::size_type>({});

  cudf::size_type const num_partitions = 3;
  auto result = cudf::hash_partition_indices(input, columns_to_hash, num_partitions);

  EXPECT_EQ(std::size_t{num_partitions}, result.size());
  for (auto const& col : result) {
    EXPECT_EQ(0, col->size());
  }
}

TEST_F(HashPartitionIndices, SinglePartition)
{
  fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4, 5, 6, 7, 8});
  auto input = cudf::table_view({col});

  auto columns_to_hash = std::vector<cudf::size_type>({0});

  cudf::size_type const num_partitions = 1;
  auto result = cudf::hash_partition_indices(input, columns_to_hash, num_partitions);

  EXPECT_EQ(std::size_t{1}, result.size());
  EXPECT_EQ(8, result[0]->size());

  // All row indices should be present
  auto gathered      = cudf::gather(input, result[0]->view());
  auto sorted_input  = cudf::sort(input);
  auto sorted_output = cudf::sort(gathered->view());
  CUDF_TEST_EXPECT_TABLES_EQUAL(sorted_input->view(), sorted_output->view());
}

TEST_F(HashPartitionIndices, PowerOfTwoPartitions)
{
  fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
  auto input = cudf::table_view({col});

  auto columns_to_hash = std::vector<cudf::size_type>({0});

  cudf::size_type const num_partitions = 4;  // Power of 2
  auto result = cudf::hash_partition_indices(input, columns_to_hash, num_partitions);

  EXPECT_EQ(std::size_t{num_partitions}, result.size());

  // Total indices should equal total rows
  cudf::size_type total = 0;
  for (auto const& col : result) {
    total += col->size();
  }
  EXPECT_EQ(16, total);
}

TEST_F(HashPartitionIndices, NonPowerOfTwoPartitions)
{
  fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
  auto input = cudf::table_view({col});

  auto columns_to_hash = std::vector<cudf::size_type>({0});

  cudf::size_type const num_partitions = 5;  // Non-power of 2
  auto result = cudf::hash_partition_indices(input, columns_to_hash, num_partitions);

  EXPECT_EQ(std::size_t{num_partitions}, result.size());

  // Total indices should equal total rows
  cudf::size_type total = 0;
  for (auto const& col : result) {
    total += col->size();
  }
  EXPECT_EQ(15, total);
}

TEST_F(HashPartitionIndices, ConsistentWithHashPartition)
{
  // Verify that gathering with the indices produces the same result as hash_partition
  fixed_width_column_wrapper<int32_t> col1({1, 2, 3, 4, 5, 6, 7, 8});
  fixed_width_column_wrapper<float> col2({1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f});
  auto input = cudf::table_view({col1, col2});

  auto columns_to_hash = std::vector<cudf::size_type>({0});

  cudf::size_type const num_partitions = 3;

  // Get partition indices
  auto indices = cudf::hash_partition_indices(input, columns_to_hash, num_partitions);

  // Get hash_partition result
  auto [partitioned_table, offsets] = cudf::hash_partition(input, columns_to_hash, num_partitions);

  // For each partition, gather from input and compare with the corresponding slice
  for (cudf::size_type p = 0; p < num_partitions; ++p) {
    auto gathered = cudf::gather(input, indices[p]->view());

    // Sort both for comparison (order within partition is undefined)
    auto sorted_gathered = cudf::sort(gathered->view());

    auto const start = offsets[p];
    auto const end   = (p + 1 < num_partitions) ? offsets[p + 1] : partitioned_table->num_rows();

    std::vector<cudf::column_view> sliced_cols;
    for (cudf::size_type c = 0; c < partitioned_table->num_columns(); ++c) {
      sliced_cols.push_back(cudf::slice(partitioned_table->get_column(c), {start, end})[0]);
    }
    auto sliced       = cudf::table_view(sliced_cols);
    auto sorted_slice = cudf::sort(sliced);

    CUDF_TEST_EXPECT_TABLES_EQUAL(sorted_gathered->view(), sorted_slice->view());
  }
}

TEST_F(HashPartitionIndices, StringColumn)
{
  strings_column_wrapper strings({"apple", "banana", "cherry", "date", "elderberry", "fig"});
  auto input = cudf::table_view({strings});

  auto columns_to_hash = std::vector<cudf::size_type>({0});

  cudf::size_type const num_partitions = 3;
  auto result = cudf::hash_partition_indices(input, columns_to_hash, num_partitions);

  EXPECT_EQ(std::size_t{num_partitions}, result.size());

  cudf::size_type total = 0;
  for (auto const& col : result) {
    total += col->size();
  }
  EXPECT_EQ(6, total);
}

TEST_F(HashPartitionIndices, WithNulls)
{
  fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4, 5, 6}, {1, 0, 1, 1, 0, 1});
  auto input = cudf::table_view({col});

  auto columns_to_hash = std::vector<cudf::size_type>({0});

  cudf::size_type const num_partitions = 3;
  auto result = cudf::hash_partition_indices(input, columns_to_hash, num_partitions);

  EXPECT_EQ(std::size_t{num_partitions}, result.size());

  cudf::size_type total = 0;
  for (auto const& col : result) {
    total += col->size();
  }
  EXPECT_EQ(6, total);
}

TEST_F(HashPartitionIndices, StructColumn)
{
  auto const to_hash = [&] {
    auto a = fixed_width_column_wrapper<int32_t>{1, 1, 1, 2, 2, 2};
    auto b = fixed_width_column_wrapper<int32_t>{1, 2, 3, 1, 2, 3};
    return structs_col{{a, b}};
  }();

  auto input = cudf::table_view({to_hash});

  auto columns_to_hash = std::vector<cudf::size_type>({0});

  cudf::size_type const num_partitions = 3;
  auto result = cudf::hash_partition_indices(input, columns_to_hash, num_partitions);

  EXPECT_EQ(std::size_t{num_partitions}, result.size());

  cudf::size_type total = 0;
  for (auto const& col : result) {
    total += col->size();
  }
  EXPECT_EQ(6, total);
}

TEST_F(HashPartitionIndices, ListColumn)
{
  using lcw = cudf::test::lists_column_wrapper<int32_t>;

  lcw to_hash{{1}, {2, 2}, {3, 3, 3}, {1}, {2, 2}, {3, 3, 3}};
  auto input = cudf::table_view({to_hash});

  auto columns_to_hash = std::vector<cudf::size_type>({0});

  cudf::size_type const num_partitions = 3;
  auto result = cudf::hash_partition_indices(input, columns_to_hash, num_partitions);

  EXPECT_EQ(std::size_t{num_partitions}, result.size());

  cudf::size_type total = 0;
  for (auto const& col : result) {
    total += col->size();
  }
  EXPECT_EQ(6, total);
}

TEST_F(HashPartitionIndices, MorePartitionsThanRows)
{
  fixed_width_column_wrapper<int32_t> col({1, 2, 3});
  auto input = cudf::table_view({col});

  auto columns_to_hash = std::vector<cudf::size_type>({0});

  cudf::size_type const num_partitions = 10;
  auto result = cudf::hash_partition_indices(input, columns_to_hash, num_partitions);

  EXPECT_EQ(std::size_t{num_partitions}, result.size());

  cudf::size_type total       = 0;
  cudf::size_type empty_count = 0;
  for (auto const& col : result) {
    total += col->size();
    if (col->size() == 0) { empty_count++; }
  }
  EXPECT_EQ(3, total);
  EXPECT_GE(empty_count, 7);  // At least 7 partitions should be empty
}

TEST_F(HashPartitionIndices, DeterministicResults)
{
  fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4, 5, 6, 7, 8, 9, 10});
  auto input = cudf::table_view({col});

  auto columns_to_hash = std::vector<cudf::size_type>({0});

  cudf::size_type const num_partitions = 4;

  auto result1 = cudf::hash_partition_indices(input, columns_to_hash, num_partitions);
  auto result2 = cudf::hash_partition_indices(input, columns_to_hash, num_partitions);

  EXPECT_EQ(result1.size(), result2.size());
  for (std::size_t i = 0; i < result1.size(); ++i) {
    // Sort the indices within each partition for comparison
    // (order within partition is undefined)
    auto sorted1 = cudf::sort(cudf::table_view({result1[i]->view()}));
    auto sorted2 = cudf::sort(cudf::table_view({result2[i]->view()}));
    CUDF_TEST_EXPECT_TABLES_EQUAL(sorted1->view(), sorted2->view());
  }
}

TEST_F(HashPartitionIndices, CustomSeed)
{
  fixed_width_column_wrapper<int32_t> col({1, 2, 3, 4, 5, 6, 7, 8});
  auto input = cudf::table_view({col});

  auto columns_to_hash = std::vector<cudf::size_type>({0});

  cudf::size_type const num_partitions = 3;

  auto result1 = cudf::hash_partition_indices(
    input, columns_to_hash, num_partitions, cudf::hash_id::HASH_MURMUR3, 12345);
  auto result2 = cudf::hash_partition_indices(
    input, columns_to_hash, num_partitions, cudf::hash_id::HASH_MURMUR3, 12345);

  EXPECT_EQ(result1.size(), result2.size());
  for (std::size_t i = 0; i < result1.size(); ++i) {
    auto sorted1 = cudf::sort(cudf::table_view({result1[i]->view()}));
    auto sorted2 = cudf::sort(cudf::table_view({result2[i]->view()}));
    CUDF_TEST_EXPECT_TABLES_EQUAL(sorted1->view(), sorted2->view());
  }
}

TEST_F(HashPartitionIndices, IdentityHash)
{
  fixed_width_column_wrapper<int32_t> col({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto input = cudf::table_view({col});

  auto columns_to_hash = std::vector<cudf::size_type>({0});

  cudf::size_type const num_partitions = 5;

  auto result = cudf::hash_partition_indices(
    input, columns_to_hash, num_partitions, cudf::hash_id::HASH_IDENTITY);

  EXPECT_EQ(std::size_t{num_partitions}, result.size());

  cudf::size_type total = 0;
  for (auto const& col : result) {
    total += col->size();
  }
  EXPECT_EQ(10, total);
}

CUDF_TEST_PROGRAM_MAIN()
