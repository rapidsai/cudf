/*
 * Copyright 2020, NVIDIA CORPORATION.
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

#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/error.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_lists.hpp>

struct ReplaceNaNsErrorTest : public cudf::test::BaseFixture {
};

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

namespace cudf {
namespace test {

template <typename T>
struct ReplaceNaNsTest : public BaseFixture {
};

using test_types = Types<float, double>;

TYPED_TEST_CASE(ReplaceNaNsTest, test_types);

template <typename T>
void ReplaceNaNsColumn(fixed_width_column_wrapper<T> input,
                       fixed_width_column_wrapper<T> replacement_values,
                       fixed_width_column_wrapper<T> expected)
{
  std::unique_ptr<column> result;
  ASSERT_NO_THROW(result = replace_nans(input, replacement_values));
  expect_columns_equal(expected, *result);
}

template <typename T>
void ReplaceNaNsScalar(fixed_width_column_wrapper<T> input,
                       scalar const& replacement_value,
                       fixed_width_column_wrapper<T> expected)
{
  std::unique_ptr<column> result;
  ASSERT_NO_THROW(result = replace_nans(input, replacement_value));
  expect_columns_equal(expected, *result);
}

TYPED_TEST(ReplaceNaNsTest, ReplaceColumn)
{
  using T = TypeParam;

  auto nan         = std::numeric_limits<double>::quiet_NaN();
  auto inputColumn = make_type_param_vector<T>({nan, 1.0, nan, 3.0, 4.0, nan, nan, 7.0, 8.0, 9.0});
  auto replacement = make_type_param_vector<T>({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});

  ReplaceNaNsColumn<T>(fixed_width_column_wrapper<T>(inputColumn.begin(), inputColumn.end()),
                       fixed_width_column_wrapper<T>(replacement.begin(), replacement.end()),
                       fixed_width_column_wrapper<T>(replacement.begin(), replacement.end()));
}

TYPED_TEST(ReplaceNaNsTest, ReplaceColumnNullable)
{
  using T = TypeParam;

  auto nan         = std::numeric_limits<double>::quiet_NaN();
  auto inputColumn = make_type_param_vector<T>({nan, 1.0, nan, 3.0, 4.0, nan, nan, 7.0, 8.0, 9.0});
  auto inputValid  = std::vector<int>{0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
  auto replacement = make_type_param_vector<T>({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0});

  // Nulls should be untouched as they are considered not nan.
  ReplaceNaNsColumn<T>(
    fixed_width_column_wrapper<T>(inputColumn.begin(), inputColumn.end(), inputValid.begin()),
    fixed_width_column_wrapper<T>(replacement.begin(), replacement.end()),
    fixed_width_column_wrapper<T>(replacement.begin(), replacement.end(), inputValid.begin()));
}

TYPED_TEST(ReplaceNaNsTest, ReplacementHasNulls)
{
  using T = TypeParam;

  auto nan           = std::numeric_limits<double>::quiet_NaN();
  auto input_column  = make_type_param_vector<T>({7.0, nan, 6.0, 3.0, nan, 2.0, 8.0, 4.0});
  auto replace_data  = make_type_param_vector<T>({4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 0.0, 1.0});
  auto replace_valid = std::vector<int>{1, 0, 1, 1, 1, 1, 1, 1};
  auto result_data   = make_type_param_vector<T>({7.0, 5.0, 6.0, 3.0, 8.0, 2.0, 8.0, 4.0});
  auto result_valid  = std::vector<int>{1, 0, 1, 1, 1, 1, 1, 1};

  ReplaceNaNsColumn<T>(
    fixed_width_column_wrapper<T>(input_column.begin(), input_column.end()),
    fixed_width_column_wrapper<T>(replace_data.begin(), replace_data.end(), replace_valid.begin()),
    fixed_width_column_wrapper<T>(result_data.begin(), result_data.end(), result_valid.begin()));
}

TYPED_TEST(ReplaceNaNsTest, ReplaceColumn_Empty)
{
  ReplaceNaNsColumn<TypeParam>(fixed_width_column_wrapper<TypeParam>{},
                               fixed_width_column_wrapper<TypeParam>{},
                               fixed_width_column_wrapper<TypeParam>{});
}

TYPED_TEST(ReplaceNaNsTest, ReplaceScalar)
{
  using T = TypeParam;

  auto nan         = std::numeric_limits<double>::quiet_NaN();
  auto input_data  = make_type_param_vector<T>({nan, 1.0, nan, 3.0, 4.0, nan, nan, 7.0, 8.0, 9.0});
  auto input_valid = std::vector<int>{0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
  auto expect_data = make_type_param_vector<T>({0.0, 1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 7.0, 8.0, 9.0});
  numeric_scalar<T> replacement(1);

  ReplaceNaNsScalar<T>(
    fixed_width_column_wrapper<T>(input_data.begin(), input_data.end(), input_valid.begin()),
    replacement,
    fixed_width_column_wrapper<T>(expect_data.begin(), expect_data.end(), input_valid.begin()));
}

TYPED_TEST(ReplaceNaNsTest, ReplaceNullScalar)
{
  using T = TypeParam;

  auto nan          = std::numeric_limits<double>::quiet_NaN();
  auto input_data   = make_type_param_vector<T>({nan, 1.0, nan, 3.0, 4.0, nan, nan, 7.0, 8.0, 9.0});
  auto input_valid  = std::vector<int>{0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
  auto expect_data  = make_type_param_vector<T>({0.0, 1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 7.0, 8.0, 9.0});
  auto expect_valid = std::vector<int>{0, 0, 0, 0, 0, 0, 0, 1, 1, 1};
  numeric_scalar<T> replacement(1, false);

  ReplaceNaNsScalar<T>(
    fixed_width_column_wrapper<T>(input_data.begin(), input_data.end(), input_valid.begin()),
    replacement,
    fixed_width_column_wrapper<T>(expect_data.begin(), expect_data.end(), expect_valid.begin()));
}

}  // namespace test
}  // namespace cudf

CUDF_TEST_PROGRAM_MAIN()
