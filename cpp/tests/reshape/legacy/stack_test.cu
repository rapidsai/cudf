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

#include <cudf/legacy/reshape.hpp>
#include <tests/utilities/legacy/column_wrapper.cuh>
#include <tests/utilities/legacy/column_wrapper_factory.hpp>
#include <tests/utilities/legacy/scalar_wrapper.cuh>
#include <tests/utilities/legacy/cudf_test_fixtures.h>

#include <numeric>
#include <random>

template <typename T>
using column_wrapper = cudf::test::column_wrapper<T>;

template <typename T>
struct StackTest : GdfTest {};

using test_types =
  ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double,
                   cudf::bool8, cudf::nvstring_category>;
TYPED_TEST_CASE(StackTest, test_types);

TYPED_TEST(StackTest, StackOrdered)
{
  using T = TypeParam;
  using wrapper = cudf::test::column_wrapper<T>;

  // Making the ranges that will be filled
  gdf_size_type num_cols = 3;
  gdf_size_type num_rows = 10;

  std::vector<int> labels(num_cols);
  std::vector<std::vector<int> > values(num_cols);

  // initialize these vectors
  for (auto &&v : values) {
    v = std::vector<int>(num_rows);
    std::iota(v.begin(), v.end(), 0);
  }

  // set your expectations
  std::vector<int> expect_data;
  for (gdf_size_type i = 0; i < num_rows; i++) {
    for (gdf_size_type j = 0; j < num_cols; j++) {
      expect_data.push_back(values[j][i]);
    }
  }

  cudf::test::column_wrapper_factory<T> factory;
  std::vector<wrapper> columns;

  auto make_table = [&] (std::vector<wrapper>& cols,
                         gdf_size_type col_size) -> cudf::table
  {
    for (gdf_size_type i = 0; i < num_cols; i++) {
      cols.emplace_back(factory.make(num_rows,
        [&](gdf_size_type j) { return values[i][j]; })
      );
    }

    std::vector<gdf_column*> raw_cols(num_cols, nullptr);
    std::transform(cols.begin(), cols.end(), raw_cols.begin(),
                   [](wrapper &c) { return c.get(); });

    return cudf::table{raw_cols.data(), num_cols};
  };

  auto input = make_table(columns, num_rows);

  wrapper expect = factory.make(num_cols * num_rows,
                                [&](auto i){ return expect_data[i]; });

  wrapper result(cudf::stack(input));

  EXPECT_EQ(expect, result) << "  Actual:" << result.to_str()
                            << "Expected:" << expect.to_str();
}
