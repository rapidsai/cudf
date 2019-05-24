/*
 * Copyright 2018 BlazingDB, Inc.
 *     Copyright 2018 Alexander Ocsa <cristhian@blazingdb.com>
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

#include <replace.hpp>

#include <utilities/error_utils.hpp>

#include <tests/utilities/column_wrapper.cuh>
#include <tests/utilities/cudf_test_fixtures.h>


template <typename T>
struct ReplaceNullsTest : GdfTest {};

using test_types =
    ::testing::Types<int8_t, int16_t, int32_t, int64_t, float, double>;

TYPED_TEST_CASE(ReplaceNullsTest, test_types);

template <typename T>
void ReplaceNulls(cudf::test::column_wrapper<T> input,
                  cudf::test::column_wrapper<T> replacement_values,
                  cudf::test::column_wrapper<T> expected)
{
  gdf_column result;
  EXPECT_NO_THROW(result = cudf::replace_nulls(input, replacement_values));

  EXPECT_TRUE(expected == result);
}

TYPED_TEST(ReplaceNullsTest, ReplaceColumn)
{
  ReplaceNulls<TypeParam>(
    cudf::test::column_wrapper<TypeParam> {8,
      [](gdf_index_type row) { return row; },
      [](gdf_index_type row) { return (row < 4) ? false : true; }},
    cudf::test::column_wrapper<TypeParam> {8,
      [](gdf_index_type row) { return 1; },
      [](gdf_index_type row) { return true; }},
    cudf::test::column_wrapper<TypeParam> {8,
      [](gdf_index_type row) { return (row < 4) ? 1 : row; },
      [](gdf_index_type row) { return true; }});
}

