/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/transpose.hpp>

#include <algorithm>
#include <random>
#include <string>

namespace {

template <typename T, typename F>
auto generate_vectors(size_t ncols, size_t nrows, F generator)
{
  std::vector<std::vector<T>> values(ncols);

  std::for_each(values.begin(), values.end(), [generator, nrows](std::vector<T>& col) {
    col.resize(nrows);
    std::generate(col.begin(), col.end(), generator);
  });

  return values;
}

template <typename T>
auto transpose_vectors(std::vector<std::vector<T>> const& input)
{
  if (input.empty()) { return input; }
  size_t ncols = input.size();
  size_t nrows = input.front().size();

  std::vector<std::vector<T>> transposed(nrows);
  std::for_each(
    transposed.begin(), transposed.end(), [=](std::vector<T>& col) { col.resize(ncols); });

  for (size_t col = 0; col < input.size(); ++col) {
    for (size_t row = 0; row < nrows; ++row) {
      transposed[row][col] = input[col][row];
    }
  }

  return transposed;
}

template <typename T, typename ColumnWrapper>
auto make_columns(std::vector<std::vector<T>> const& values)
{
  std::vector<ColumnWrapper> columns;
  columns.reserve(values.size());

  for (auto const& value_col : values) {
    columns.emplace_back(value_col.begin(), value_col.end());
  }

  return columns;
}

template <typename T, typename ColumnWrapper>
auto make_columns(std::vector<std::vector<T>> const& values,
                  std::vector<std::vector<cudf::size_type>> const& valids)
{
  std::vector<ColumnWrapper> columns;
  columns.reserve(values.size());

  for (size_t col = 0; col < values.size(); ++col) {
    columns.emplace_back(values[col].begin(), values[col].end(), valids[col].begin());
  }

  return columns;
}

template <typename ColumnWrapper>
auto make_table_view(std::vector<ColumnWrapper> const& cols)
{
  std::vector<cudf::column_view> views(cols.size());

  std::transform(cols.begin(), cols.end(), views.begin(), [](auto const& col) {
    return static_cast<cudf::column_view>(col);
  });

  return cudf::table_view(views);
}

template <typename T>
void run_test(size_t ncols, size_t nrows, bool add_nulls)
{
  using ColumnWrapper = std::conditional_t<std::is_same_v<T, std::string>,
                                           cudf::test::strings_column_wrapper,
                                           cudf::test::fixed_width_column_wrapper<T>>;

  std::mt19937 rng(1);

  // Generate values as vector of vectors
  auto const values = generate_vectors<T>(
    ncols, nrows, [&rng]() { return cudf::test::make_type_param_scalar<T>(rng()); });
  auto const valuesT = transpose_vectors(values);

  std::vector<ColumnWrapper> input_cols;
  std::vector<ColumnWrapper> expected_cols;
  std::vector<cudf::size_type> expected_nulls(nrows);

  if (add_nulls) {
    // Generate null mask as vector of vectors
    auto const valids = generate_vectors<cudf::size_type>(
      ncols, nrows, [&rng]() { return static_cast<cudf::size_type>(rng() % 3 > 0 ? 1 : 0); });
    auto const validsT = transpose_vectors(valids);

    // Compute the null counts over each transposed column
    std::transform(validsT.begin(),
                   validsT.end(),
                   expected_nulls.begin(),
                   [ncols](std::vector<cudf::size_type> const& vec) {
                     // num nulls = num elems - num valids
                     return ncols - std::accumulate(vec.begin(), vec.end(), 0);
                   });

    // Create column wrappers from vector of vectors
    input_cols    = make_columns<T, ColumnWrapper>(values, valids);
    expected_cols = make_columns<T, ColumnWrapper>(valuesT, validsT);
  } else {
    input_cols    = make_columns<T, ColumnWrapper>(values);
    expected_cols = make_columns<T, ColumnWrapper>(valuesT);
  }

  // Create table views from column wrappers
  auto input_view    = make_table_view(input_cols);
  auto expected_view = make_table_view(expected_cols);

  auto result      = transpose(input_view);
  auto result_view = std::get<1>(result);

  ASSERT_EQ(result_view.num_columns(), expected_view.num_columns());
  for (cudf::size_type i = 0; i < result_view.num_columns(); ++i) {
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result_view.column(i), expected_view.column(i));
    EXPECT_EQ(result_view.column(i).null_count(), expected_nulls[i]);
  }
}

}  // namespace

template <typename T>
class TransposeTest : public cudf::test::BaseFixture {};

// Using std::string here instead of cudf::test::StringTypes allows us to
// use std::vector<T> utilities in this file just like the fixed-width types.
// Should consider changing cudf::test::StringTypes to std::string instead of cudf::string_view.
using StdStringType  = cudf::test::Types<std::string>;
using TransposeTypes = cudf::test::Concat<cudf::test::FixedWidthTypes, StdStringType>;

TYPED_TEST_SUITE(TransposeTest, TransposeTypes);  // cudf::test::FixedWidthTypes);

TYPED_TEST(TransposeTest, SingleValue) { run_test<TypeParam>(1, 1, false); }

TYPED_TEST(TransposeTest, SingleColumn) { run_test<TypeParam>(1, 1000, false); }

TYPED_TEST(TransposeTest, SingleColumnNulls) { run_test<TypeParam>(1, 1000, true); }

TYPED_TEST(TransposeTest, Square) { run_test<TypeParam>(100, 100, false); }

TYPED_TEST(TransposeTest, SquareNulls) { run_test<TypeParam>(100, 100, true); }

TYPED_TEST(TransposeTest, Slim) { run_test<TypeParam>(10, 1000, false); }

TYPED_TEST(TransposeTest, SlimNulls) { run_test<TypeParam>(10, 1000, true); }

TYPED_TEST(TransposeTest, Fat) { run_test<TypeParam>(1000, 10, false); }

TYPED_TEST(TransposeTest, FatNulls) { run_test<TypeParam>(1000, 10, true); }

TYPED_TEST(TransposeTest, EmptyTable) { run_test<TypeParam>(0, 0, false); }

TYPED_TEST(TransposeTest, EmptyColumns) { run_test<TypeParam>(10, 0, false); }

class TransposeTestError : public cudf::test::BaseFixture {};

TEST_F(TransposeTestError, MismatchedColumns)
{
  cudf::test::fixed_width_column_wrapper<uint32_t, int32_t> col1({1, 2, 3});
  cudf::test::fixed_width_column_wrapper<int8_t> col2{{4, 5, 6}};
  cudf::test::fixed_width_column_wrapper<float> col3{{7, 8, 9}};
  cudf::table_view input{{col1, col2, col3}};
  EXPECT_THROW(cudf::transpose(input), cudf::logic_error);
}

CUDF_TEST_PROGRAM_MAIN()
