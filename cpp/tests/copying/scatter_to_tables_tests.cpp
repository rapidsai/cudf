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
#include <cudf/copying.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/type_lists.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

#include <thrust/iterator/constant_iterator.h>

using cudf::test::fixed_width_column_wrapper;
using cudf::test::strings_column_wrapper;
using cudf::test::expect_tables_equal;

namespace {

// Return vector of gather maps per partition
template <typename T>
auto make_gather_maps(std::vector<T> const& partition_map)
{
  auto const max = *std::max_element(partition_map.begin(), partition_map.end());
  std::vector<std::vector<size_t>> gather_maps(max + 1);
  for (size_t i = 0; i < partition_map.size(); ++i) {
    auto const partition = static_cast<size_t>(partition_map[i]);
    gather_maps.at(partition).push_back(i);
  }
  return gather_maps;
}

// Return vector of columns per partition
template <typename T>
auto gather_fixed_width(std::vector<T> const& values,
    std::vector<std::vector<size_t>> const& gather_maps)
{
  std::vector<fixed_width_column_wrapper<T>> columns(gather_maps.size());

  std::transform(gather_maps.begin(), gather_maps.end(), columns.begin(),
    [&values](auto const& gather_map) {
      auto gather_iter = thrust::make_permutation_iterator(
        values.begin(), gather_map.begin());
      return fixed_width_column_wrapper<T>(gather_iter,
        gather_iter + gather_map.size());
    });

  return columns;
}

// Return vector of columns per partition
template <typename T, typename ValidIterator>
auto gather_fixed_width(std::vector<T> const& values, ValidIterator valid_iterator,
    std::vector<std::vector<size_t>> const& gather_maps)
{
  std::vector<fixed_width_column_wrapper<T>> columns(gather_maps.size());

  std::transform(gather_maps.begin(), gather_maps.end(), columns.begin(),
    [&values, &valid_iterator](auto const& gather_map) {
      auto gather_iter = thrust::make_permutation_iterator(
        values.begin(), gather_map.begin());
      auto valid_permuted = thrust::make_permutation_iterator(
        valid_iterator, gather_map.begin());
      return fixed_width_column_wrapper<T>(gather_iter,
        gather_iter + gather_map.size(), valid_permuted);
    });

  return columns;
}

// Return vector of columns per partition
template <bool nullable>
auto gather_strings(std::vector<char const*> const& strings,
    std::vector<std::vector<size_t>> const& gather_maps)
{
  // No default constructor so reserve and push_back
  std::vector<strings_column_wrapper> columns;
  columns.reserve(gather_maps.size());

  for (auto const& gather_map : gather_maps) {
    auto gather_iter = thrust::make_permutation_iterator(
      strings.begin(), gather_map.begin());
    if (nullable) {
      auto valid_iter = thrust::make_transform_iterator(gather_iter,
        [](char const* ptr) { return ptr != nullptr; });
      columns.push_back(strings_column_wrapper(gather_iter,
        gather_iter + gather_map.size(), valid_iter));
    } else {
      columns.push_back(strings_column_wrapper(gather_iter,
        gather_iter + gather_map.size()));
    }
  };

  return columns;
}

// Transform vector of column wrappers to vector of column views
template <typename T>
auto make_view_vector(std::vector<T> const& columns)
{
  std::vector<cudf::column_view> views(columns.size());
  std::transform(columns.begin(), columns.end(), views.begin(),
    [](auto const& col) { return static_cast<cudf::column_view>(col); });
  return views;
}

// Splice vector of partitioned columns into vector of tables
auto make_table_view_vector(std::vector<std::vector<cudf::column_view>> const& partitions) {
  auto const num_cols = partitions.size();
  auto const num_parts = partitions.front().size();

  // No default constructor so reserve and push_back
  std::vector<cudf::table_view> views;
  views.reserve(num_parts);

  std::vector<cudf::column_view> cols(num_cols);
  for (size_t i_part = 0; i_part < num_parts; ++i_part) {
    for (size_t i_col = 0; i_col < num_cols; ++i_col) {
      cols.at(i_col) = partitions.at(i_col).at(i_part);
    }
    views.push_back(cudf::table_view(cols));
  }

  return views;
}

}  // namespace

template <typename T>
class ScatterToTablesInvalidIndexTypes : public cudf::test::BaseFixture {};

// NOTE string types hit static assert in fixed_width_column_wrapper
using InvalidIndexTypes = cudf::test::Concat<
    cudf::test::Types<float, double, cudf::experimental::bool8>,
    cudf::test::TimestampTypes>;
TYPED_TEST_CASE(ScatterToTablesInvalidIndexTypes, InvalidIndexTypes);

TYPED_TEST(ScatterToTablesInvalidIndexTypes, InvalidPartitionMap)
{
  auto integers = std::vector<int16_t>({1, 2, 3, 4, 5, 6});
  auto partition_map = std::vector<TypeParam>({3, 1, 1, 4, 1, 3});

  // Assemble input table
  auto integers_in = fixed_width_column_wrapper<int16_t>(integers.begin(), integers.end());
  auto input = cudf::table_view({integers_in});

  auto const partition_col = fixed_width_column_wrapper<TypeParam>(
    partition_map.begin(), partition_map.end());

  EXPECT_THROW(cudf::experimental::scatter_to_tables(input, partition_col), cudf::logic_error);
}

template <typename T>
class ScatterToTablesIndexTypes : public cudf::test::BaseFixture {};

using IndexTypes = cudf::test::Types<int8_t, int16_t, int32_t, int64_t>;
TYPED_TEST_CASE(ScatterToTablesIndexTypes, IndexTypes);

TYPED_TEST(ScatterToTablesIndexTypes, MapTooManyRows)
{
  auto integers = std::vector<int16_t>({1, 2, 3, 4, 5, 6});
  auto partition_map = std::vector<TypeParam>({3, 1, 1, 4, 1, 3, 4}); // one row too many

  // Assemble input table
  auto integers_in = fixed_width_column_wrapper<int16_t>(integers.begin(), integers.end());
  auto input = cudf::table_view({integers_in});

  auto const partition_col = fixed_width_column_wrapper<TypeParam>(
    partition_map.begin(), partition_map.end());

  EXPECT_THROW(cudf::experimental::scatter_to_tables(input, partition_col), cudf::logic_error);
}

TYPED_TEST(ScatterToTablesIndexTypes, MapHasNulls)
{
  auto integers = std::vector<int16_t>({1, 2, 3, 4, 5, 6});
  auto partition_map = std::vector<TypeParam>({3, 1, 1, 4, 1, 3});

  // Assemble input table
  auto integers_in = fixed_width_column_wrapper<int16_t>(integers.begin(), integers.end());
  auto input = cudf::table_view({integers_in});

  // Add nulls to partition column
  auto nulls_iterator = thrust::make_constant_iterator(0);
  auto const partition_col = fixed_width_column_wrapper<TypeParam>(
    partition_map.begin(), partition_map.end(), nulls_iterator);

  EXPECT_THROW(cudf::experimental::scatter_to_tables(input, partition_col), cudf::logic_error);
}

TYPED_TEST(ScatterToTablesIndexTypes, MultipleTypes)
{
  auto floats = std::vector<float>({1.f, 2.f, 3.f, 4.f, 5.f, 6.f});
  auto integers = std::vector<int16_t>({1, 2, 3, 4, 5, 6});
  auto strings = std::vector<char const*>({"a", "bb", "ccc", "d", "ee", "fff"});
  auto partition_map = std::vector<TypeParam>({3, 1, 1, 4}); // shorter than input is ok
  auto validity = cudf::test::make_counting_transform_iterator(0, [](auto i){return 1;});

  // Assemble input table
  auto floats_in = fixed_width_column_wrapper<float>(floats.begin(), floats.end(), validity);
  auto integers_in = fixed_width_column_wrapper<int16_t>(integers.begin(), integers.end(), validity);
  auto strings_in = strings_column_wrapper(strings.begin(), strings.end(), validity);
  auto input = cudf::table_view({floats_in, integers_in, strings_in});

  auto const partition_col = fixed_width_column_wrapper<TypeParam>(
    partition_map.begin(), partition_map.end());

  // Compute expected tables
  auto gather_maps = make_gather_maps(partition_map);
  auto floats_cols = gather_fixed_width(floats, validity, gather_maps);
  auto integers_cols = gather_fixed_width(integers, validity, gather_maps);
  auto strings_cols = gather_strings<true>(strings, gather_maps);

  auto floats_views = make_view_vector(floats_cols);
  auto integers_views = make_view_vector(integers_cols);
  auto strings_views = make_view_vector(strings_cols);
  auto expected = make_table_view_vector({floats_views, integers_views, strings_views});

  auto result = cudf::experimental::scatter_to_tables(input, partition_col);
  EXPECT_EQ(expected.size(), result.size());
  for (size_t i = 0; i < result.size(); ++i) {
    expect_tables_equal(expected[i], result[i]->view());
  }
}

TYPED_TEST(ScatterToTablesIndexTypes, StringsNulls)
{
  auto strings = std::vector<char const*>({"a", "bb", nullptr, "d", nullptr, "fff"});
  auto partition_map = std::vector<TypeParam>({3, 1, 1, 4, 1, 3});

  // Assemble input table
  auto strings_in = strings_column_wrapper(strings.begin(), strings.end(),
    thrust::make_transform_iterator(strings.begin(), [](auto str) { return str != nullptr; }));
  auto input = cudf::table_view({strings_in});

  auto const partition_col = fixed_width_column_wrapper<TypeParam>(
    partition_map.begin(), partition_map.end());

  // Compute expected tables
  auto gather_maps = make_gather_maps(partition_map);
  auto strings_cols = gather_strings<true>(strings, gather_maps);

  auto strings_views = make_view_vector(strings_cols);
  auto expected = make_table_view_vector({strings_views});

  auto result = cudf::experimental::scatter_to_tables(input, partition_col);
  EXPECT_EQ(expected.size(), result.size());
  for (size_t i = 0; i < result.size(); ++i) {
    expect_tables_equal(expected[i], result[i]->view());
  }
}

template <typename T>
class ScatterToTablesFixedWidth : public cudf::test::BaseFixture {};

TYPED_TEST_CASE(ScatterToTablesFixedWidth, cudf::test::FixedWidthTypes);

TYPED_TEST(ScatterToTablesFixedWidth, MixedNulls)
{
  auto first = std::vector<TypeParam>({1, 2, 3, 4, 5, 6});
  auto second = std::vector<TypeParam>({7, 8, 9, 10, 11, 12});
  auto partition_map = std::vector<int32_t>({3, 1, 1, 4, 1, 3});

  auto valid_iter = thrust::make_transform_iterator(thrust::make_counting_iterator(0),
    [](auto index) { return index % 3 > 0; });

  // Assemble input table
  auto first_in = fixed_width_column_wrapper<TypeParam>(first.begin(), first.end());
  auto second_in = fixed_width_column_wrapper<TypeParam>(second.begin(), second.end(), valid_iter);
  auto input = cudf::table_view({first_in, second_in});

  auto const partition_col = fixed_width_column_wrapper<int32_t>(
    partition_map.begin(), partition_map.end());

  // Compute expected tables
  auto gather_maps = make_gather_maps(partition_map);
  auto first_cols = gather_fixed_width(first, gather_maps);
  auto second_cols = gather_fixed_width(second, valid_iter, gather_maps);

  auto first_views = make_view_vector(first_cols);
  auto second_views = make_view_vector(second_cols);
  auto expected = make_table_view_vector({first_views, second_views});

  auto result = cudf::experimental::scatter_to_tables(input, partition_col);
  EXPECT_EQ(expected.size(), result.size());
  for (size_t i = 0; i < result.size(); ++i) {
    expect_tables_equal(expected[i], result[i]->view());
  }
}
