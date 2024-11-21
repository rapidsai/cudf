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

class TransposeTest : public cudf::test::BaseFixture {};

TEST_F(TransposeTest, StringTest)
{
  using ColumnWrapper = cudf::test::strings_column_wrapper;
  size_t ncols        = 10;
  size_t nrows        = 10;

  std::mt19937 rng(1);

  auto const values = [&rng, nrows, ncols]() {
    std::vector<std::vector<std::string>> values(ncols);
    std::for_each(values.begin(), values.end(), [&rng, nrows](std::vector<std::string>& col) {
      col.resize(nrows);
      std::generate(col.begin(), col.end(), [&rng]() { return std::to_string(rng()); });
    });
    return values;
  }();

  auto input_cols = [&values]() {
    std::vector<ColumnWrapper> columns;
    columns.reserve(values.size());
    for (auto const& value_col : values) {
      columns.emplace_back(value_col.begin(), value_col.end());
    }
    return columns;
  }();

  auto input_view = [&input_cols]() {
    std::vector<cudf::column_view> views(input_cols.size());
    std::transform(input_cols.begin(), input_cols.end(), views.begin(), [](auto const& col) {
      return static_cast<cudf::column_view>(col);
    });
    return cudf::table_view(views);
  }();

  auto result = transpose(input_view, cudf::test::get_default_stream());
}

CUDF_TEST_PROGRAM_MAIN()
