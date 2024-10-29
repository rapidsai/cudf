/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <cudf/replace.hpp>
#include <cudf/scalar/scalar.hpp>

struct ReplaceNaNsErrorTest : public cudf::test::BaseFixture {};

// Error: old-values and new-values size mismatch
TEST_F(ReplaceNaNsErrorTest, SizeMismatch)
{
  cudf::test::fixed_width_column_wrapper<float> input_column{7, 5, 6, 3, 1, 8, 4};
  cudf::test::fixed_width_column_wrapper<float> replacement_column{{10, 11, 12, 13}};

  EXPECT_THROW(cudf::replace_nans(input_column, replacement_column), cudf::logic_error);
}

// Error : column type mismatch
TEST_F(ReplaceNaNsErrorTest, TypeMismatch)
{
  cudf::test::fixed_width_column_wrapper<float> input_column{7, 5, 6, 3, 1, 2, 8, 4};
  cudf::test::fixed_width_column_wrapper<double> replacement_column{10, 11, 12, 13, 14, 15, 16, 17};

  EXPECT_THROW(cudf::replace_nans(input_column, replacement_column), cudf::logic_error);
}

// Error: column type mismatch
TEST_F(ReplaceNaNsErrorTest, TypeMismatchScalar)
{
  cudf::test::fixed_width_column_wrapper<double> input_column{7, 5, 6, 3, 1, 2, 8, 4};
  cudf::numeric_scalar<float> replacement(1);

  EXPECT_THROW(cudf::replace_nans(input_column, replacement), cudf::logic_error);
}

// Error: column type mismatch
TEST_F(ReplaceNaNsErrorTest, NonFloatType)
{
  cudf::test::fixed_width_column_wrapper<int32_t> input_column{7, 5, 6, 3, 1, 2, 8, 4};
  cudf::numeric_scalar<float> replacement(1);

  EXPECT_THROW(cudf::replace_nans(input_column, replacement), cudf::logic_error);
}

template <typename T>
struct ReplaceNaNsTest : public cudf::test::BaseFixture {};

using test_types = cudf::test::Types<float, double>;

TYPED_TEST_SUITE(ReplaceNaNsTest, test_types);

template <typename T>
void ReplaceNaNsColumn(cudf::test::fixed_width_column_wrapper<T> input,
                       cudf::test::fixed_width_column_wrapper<T> replacement_values,
                       cudf::test::fixed_width_column_wrapper<T> expected)
{
  std::unique_ptr<cudf::column> result;
  ASSERT_NO_THROW(result = cudf::replace_nans(input, replacement_values));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

template <typename T>
void ReplaceNaNsScalar(cudf::test::fixed_width_column_wrapper<T> input,
                       cudf::scalar const& replacement_value,
                       cudf::test::fixed_width_column_wrapper<T> expected)
{
  std::unique_ptr<cudf::column> result;
  ASSERT_NO_THROW(result = cudf::replace_nans(input, replacement_value));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *result);
}

TYPED_TEST(ReplaceNaNsTest, ReplaceColumn)
{
  using T = TypeParam;

  auto nan = std::numeric_limits<double>::quiet_NaN();
  auto inputColumn =
    cudf::test::make_type_param_vector<T>({nan, 1.0, nan, 3.0, 4.0, nan, nan, 7.0, 8.0, 9.0});
  auto replacement =
    cudf::test::make_type_param_vector<T>({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});

  ReplaceNaNsColumn<T>(
    cudf::test::fixed_width_column_wrapper<T>(inputColumn.begin(), inputColumn.end()),
    cudf::test::fixed_width_column_wrapper<T>(replacement.begin(), replacement.end()),
    cudf::test::fixed_width_column_wrapper<T>(replacement.begin(), replacement.end()));
}

TYPED_TEST(ReplaceNaNsTest, ReplaceColumnNullable)
{
  using T = TypeParam;

  auto nan = std::numeric_limits<double>::quiet_NaN();
  auto inputColumn =
    cudf::test::make_type_param_vector<T>({nan, 1.0, nan, 3.0, 4.0, nan, nan, 7.0, 8.0, 9.0});
  auto inputValid = std::vector<int>{0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
  auto replacement =
    cudf::test::make_type_param_vector<T>({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});

  // Nulls should be untouched as they are considered not nan.
  ReplaceNaNsColumn<T>(
    cudf::test::fixed_width_column_wrapper<T>(
      inputColumn.begin(), inputColumn.end(), inputValid.begin()),
    cudf::test::fixed_width_column_wrapper<T>(replacement.begin(), replacement.end()),
    cudf::test::fixed_width_column_wrapper<T>(
      replacement.begin(), replacement.end(), inputValid.begin()));
}

TYPED_TEST(ReplaceNaNsTest, ReplacementHasNulls)
{
  using T = TypeParam;

  auto nan = std::numeric_limits<double>::quiet_NaN();
  auto input_column =
    cudf::test::make_type_param_vector<T>({7.0, nan, 6.0, 3.0, nan, 2.0, 8.0, 4.0});
  auto replace_data =
    cudf::test::make_type_param_vector<T>({4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0, 1.0});
  auto replace_valid = std::vector<int>{1, 0, 1, 1, 1, 1, 1, 1};
  auto result_data =
    cudf::test::make_type_param_vector<T>({7.0, 5.0, 6.0, 3.0, 8.0, 2.0, 8.0, 4.0});
  auto result_valid = std::vector<int>{1, 0, 1, 1, 1, 1, 1, 1};

  ReplaceNaNsColumn<T>(
    cudf::test::fixed_width_column_wrapper<T>(input_column.begin(), input_column.end()),
    cudf::test::fixed_width_column_wrapper<T>(
      replace_data.begin(), replace_data.end(), replace_valid.begin()),
    cudf::test::fixed_width_column_wrapper<T>(
      result_data.begin(), result_data.end(), result_valid.begin()));
}

TYPED_TEST(ReplaceNaNsTest, ReplaceColumn_Empty)
{
  ReplaceNaNsColumn<TypeParam>(cudf::test::fixed_width_column_wrapper<TypeParam>{},
                               cudf::test::fixed_width_column_wrapper<TypeParam>{},
                               cudf::test::fixed_width_column_wrapper<TypeParam>{});
}

TYPED_TEST(ReplaceNaNsTest, ReplaceScalar)
{
  using T = TypeParam;

  auto nan = std::numeric_limits<double>::quiet_NaN();
  auto input_data =
    cudf::test::make_type_param_vector<T>({nan, 1.0, nan, 3.0, 4.0, nan, nan, 7.0, 8.0, 9.0});
  auto input_valid = std::vector<int>{0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
  auto expect_data =
    cudf::test::make_type_param_vector<T>({0.0, 1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 7.0, 8.0, 9.0});
  cudf::numeric_scalar<T> replacement(1);

  ReplaceNaNsScalar<T>(cudf::test::fixed_width_column_wrapper<T>(
                         input_data.begin(), input_data.end(), input_valid.begin()),
                       replacement,
                       cudf::test::fixed_width_column_wrapper<T>(
                         expect_data.begin(), expect_data.end(), input_valid.begin()));
}

TYPED_TEST(ReplaceNaNsTest, ReplaceNullScalar)
{
  using T = TypeParam;

  auto nan = std::numeric_limits<double>::quiet_NaN();
  auto input_data =
    cudf::test::make_type_param_vector<T>({nan, 1.0, nan, 3.0, 4.0, nan, nan, 7.0, 8.0, 9.0});
  auto input_valid = std::vector<int>{0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
  auto expect_data =
    cudf::test::make_type_param_vector<T>({0.0, 1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 7.0, 8.0, 9.0});
  auto expect_valid = std::vector<int>{0, 0, 0, 0, 0, 0, 0, 1, 1, 1};
  cudf::numeric_scalar<T> replacement(1, false);

  ReplaceNaNsScalar<T>(cudf::test::fixed_width_column_wrapper<T>(
                         input_data.begin(), input_data.end(), input_valid.begin()),
                       replacement,
                       cudf::test::fixed_width_column_wrapper<T>(
                         expect_data.begin(), expect_data.end(), expect_valid.begin()));
}

CUDF_TEST_PROGRAM_MAIN()
