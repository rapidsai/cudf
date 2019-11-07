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

#include <cudf/replace.hpp>

#include <cudf/utilities/error.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_lists.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <cudf/cudf.h>
#include <gtest/gtest.h>
#include <cudf/scalar/scalar.hpp>

template <typename T>
struct ReplaceNullsTest : public cudf::test::BaseFixture {};

using test_types = cudf::test::Types<int8_t, int16_t, int32_t, int64_t, float, double>;

TYPED_TEST_CASE(ReplaceNullsTest, test_types);

template <typename T>
void ReplaceNullsColumn(cudf::test::fixed_width_column_wrapper<T> input,
                        cudf::test::fixed_width_column_wrapper<T> replacement_values,
                        cudf::test::fixed_width_column_wrapper<T> expected)
{
  std::unique_ptr<cudf::column> result;
  ASSERT_NO_THROW(result = cudf::experimental::replace_nulls(input, replacement_values));
  expect_columns_equal(expected, *result);
}

template <typename T>
void ReplaceNullsScalar(cudf::test::fixed_width_column_wrapper<T> input,
                        cudf::scalar const* replacement_value,
                        cudf::test::fixed_width_column_wrapper<T> expected)
{
  std::unique_ptr<cudf::column> result;
  ASSERT_NO_THROW(result = cudf::experimental::replace_nulls(input, replacement_value));
  expect_columns_equal(expected, *result);
}

TYPED_TEST(ReplaceNullsTest, ReplaceColumn)
{
  std::vector<TypeParam> inputColumn{0,1,2,3,4,5,6,7,8,9};
  std::vector<cudf::valid_type> inputValid{0,0,0,0,0,1,1,1,1,1};
  std::vector<TypeParam> replacementColumn{0,1,2,3,4,5,6,7,8,9};



  ReplaceNullsColumn<TypeParam>(
    cudf::test::fixed_width_column_wrapper<TypeParam> {inputColumn.begin(), inputColumn.end(), inputValid.begin()},
    cudf::test::fixed_width_column_wrapper<TypeParam> {replacementColumn.begin(), replacementColumn.end()},
    cudf::test::fixed_width_column_wrapper<TypeParam> {replacementColumn.begin(), replacementColumn.end()});
}


TYPED_TEST(ReplaceNullsTest, ReplaceScalar)
{
  std::vector<TypeParam> inputColumn {0,1,2,3,4,5,6,7,8,9};
  std::vector<cudf::valid_type> inputValid {0,0,0,0,0,1,1,1,1,1};
  std::vector<TypeParam> expectedColumn{1,1,1,1,1,5,6,7,8,9};
  std::unique_ptr<cudf::scalar> replacement(new cudf::numeric_scalar<TypeParam>(1));

  ReplaceNullsScalar<TypeParam>(
    cudf::test::fixed_width_column_wrapper<TypeParam> {inputColumn.begin(), inputColumn.end(), inputValid.begin()},
    replacement.get(),
    cudf::test::fixed_width_column_wrapper<TypeParam> {expectedColumn.begin(), expectedColumn.end()});
}
