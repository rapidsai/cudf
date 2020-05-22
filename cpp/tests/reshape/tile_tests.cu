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

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/type_lists.hpp>

#include <cudf/reshape.hpp>
#include <cudf/table/table.hpp>
#include "cudf/utilities/error.hpp"

using namespace cudf::test;

template <typename T>
struct TileTest : public BaseFixture {
};

TYPED_TEST_CASE(TileTest, cudf::test::AllTypes);

TYPED_TEST(TileTest, NoColumns)
{
  using T = TypeParam;

  cudf::table_view in(std::vector<cudf::column_view>{});

  auto expected = in;

  auto actual = cudf::tile(in, 10);

  cudf::test::expect_tables_equal(expected, actual->view());
}

TYPED_TEST(TileTest, NoRows)
{
  using T = TypeParam;

  fixed_width_column_wrapper<T> in_a({});
  cudf::table_view in(std::vector<cudf::column_view>{in_a});

  auto expected = in;

  auto actual = cudf::tile(in, 10);

  cudf::test::expect_tables_equal(expected, actual->view());
}

TYPED_TEST(TileTest, OneColumn)
{
  using T = TypeParam;

  fixed_width_column_wrapper<T> in_a({-1, 0, 1});
  cudf::table_view in(std::vector<cudf::column_view>{in_a});

  fixed_width_column_wrapper<T> expected_a({-1, 0, 1, -1, 0, 1});
  cudf::table_view expected(std::vector<cudf::column_view>{expected_a});

  auto actual = cudf::tile(in, 2);

  cudf::test::expect_tables_equal(expected, actual->view());
}

TYPED_TEST(TileTest, OneColumnNullable)
{
  using T = TypeParam;

  fixed_width_column_wrapper<T> in_a({-1, 0, 1}, {1, 0, 0});
  cudf::table_view in(std::vector<cudf::column_view>{in_a});

  fixed_width_column_wrapper<T> expected_a({-1, 0, 1, -1, 0, 1}, {1, 0, 0, 1, 0, 0});
  cudf::table_view expected(std::vector<cudf::column_view>{expected_a});

  auto actual = cudf::tile(in, 2);

  cudf::test::expect_tables_equal(expected, actual->view());
}

TYPED_TEST(TileTest, OneColumnNegativeCount)
{
  using T = TypeParam;

  fixed_width_column_wrapper<T> in_a({-1, 0, 1}, {1, 0, 0});
  cudf::table_view in(std::vector<cudf::column_view>{in_a});

  EXPECT_THROW(cudf::tile(in, -1), cudf::logic_error);
}

TYPED_TEST(TileTest, OneColumnZeroCount)
{
  using T = TypeParam;

  fixed_width_column_wrapper<T> in_a({-1, 0, 1}, {1, 0, 0});
  cudf::table_view in(std::vector<cudf::column_view>{in_a});

  std::vector<T> vals{};
  std::vector<bool> mask{};

  fixed_width_column_wrapper<T> expected_a(vals.begin(), vals.end(), mask.begin());

  cudf::table_view expected(std::vector<cudf::column_view>{expected_a});

  auto actual = cudf::tile(in, 0);

  cudf::test::expect_tables_equal(expected, actual->view());
}
