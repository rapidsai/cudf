/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <cudf_test/default_stream.hpp>
#include <cudf_test/testing_main.hpp>

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

template <typename ColumnWrapper>
auto make_table_view(std::vector<ColumnWrapper> const& cols)
{
  std::vector<cudf::column_view> views(cols.size());

  std::transform(cols.begin(), cols.end(), views.begin(), [](auto const& col) {
    return static_cast<cudf::column_view>(col);
  });

  return cudf::table_view(views);
}

}  // namespace

class TransposeTest : public cudf::test::BaseFixture {};

TEST_F(TransposeTest, StringTest)
{
  using ColumnWrapper = cudf::test::strings_column_wrapper;

  size_t ncols = 10;
  size_t nrows = 10;

  std::mt19937 rng(1);

  // Generate values as vector of vectors
  auto const values =
    generate_vectors<std::string>(ncols, nrows, [&rng]() { return std::to_string(rng()); });

  auto input_cols = make_columns<std::string, ColumnWrapper>(values);

  // Create table views from column wrappers
  auto input_view = make_table_view(input_cols);

  auto result = transpose(input_view, cudf::test::get_default_stream());
}

CUDF_TEST_PROGRAM_MAIN()
