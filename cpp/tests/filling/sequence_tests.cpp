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

#include <cudf/scalar/scalar.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/cudf_gtest.hpp>
#include <tests/utilities/type_lists.hpp>

#include <cudf/filling.hpp>

using namespace cudf;
using namespace cudf::test;

template <typename T>
class SequenceTypedTestFixture : public cudf::test::BaseFixture {
};

class SequenceTestFixture : public cudf::test::BaseFixture {
};

using NumericTypesNoBool = cudf::test::Types<int8_t, int16_t, int32_t, int64_t, float, double>;

TYPED_TEST_CASE(SequenceTypedTestFixture, NumericTypesNoBool);

TYPED_TEST(SequenceTypedTestFixture, Incrementing)
{
  using T = TypeParam;

  numeric_scalar<T> init(0);
  numeric_scalar<T> step(1);

  size_type num_els = 10;

  T expected[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  fixed_width_column_wrapper<T> expected_w(expected, expected + num_els);

  auto result = cudf::sequence(num_els, init, step);

  expect_columns_equal(*result, expected_w);
}

TYPED_TEST(SequenceTypedTestFixture, Decrementing)
{
  using T = TypeParam;

  numeric_scalar<T> init(0);
  numeric_scalar<T> step(-5);

  size_type num_els = 10;

  T expected[] = {0, -5, -10, -15, -20, -25, -30, -35, -40, -45};
  fixed_width_column_wrapper<T> expected_w(expected, expected + num_els);

  auto result = cudf::sequence(num_els, init, step);

  expect_columns_equal(*result, expected_w);
}

TYPED_TEST(SequenceTypedTestFixture, EmptyOutput)
{
  using T = TypeParam;

  numeric_scalar<T> init(0);
  numeric_scalar<T> step(-5);

  size_type num_els = 0;

  T expected[] = {};
  fixed_width_column_wrapper<T> expected_w(expected, expected + num_els);

  auto result = cudf::sequence(num_els, init, step);

  expect_columns_equal(*result, expected_w);
}

TEST_F(SequenceTestFixture, BadTypes)
{
  string_scalar string_init("zero");
  string_scalar string_step("???");
  EXPECT_THROW(cudf::sequence(10, string_init, string_step), cudf::logic_error);

  numeric_scalar<bool> bool_init(true);
  numeric_scalar<bool> bool_step(false);
  EXPECT_THROW(cudf::sequence(10, bool_init, bool_step), cudf::logic_error);

  timestamp_scalar<timestamp_s> ts_init(duration_s{10}, true);
  timestamp_scalar<timestamp_s> ts_step(duration_s{10}, true);
  EXPECT_THROW(cudf::sequence(10, ts_init, ts_step), cudf::logic_error);
}

TEST_F(SequenceTestFixture, MismatchedInputs)
{
  numeric_scalar<int> init(0);
  numeric_scalar<float> step(-5);
  EXPECT_THROW(cudf::sequence(10, init, step), cudf::logic_error);

  numeric_scalar<int> init2(0);
  numeric_scalar<int8_t> step2(-5);
  EXPECT_THROW(cudf::sequence(10, init2, step2), cudf::logic_error);

  numeric_scalar<float> init3(0);
  numeric_scalar<double> step3(-5);
  EXPECT_THROW(cudf::sequence(10, init3, step3), cudf::logic_error);
}

TYPED_TEST(SequenceTypedTestFixture, DefaultStep)
{
  using T = TypeParam;

  numeric_scalar<T> init(0);

  size_type num_els = 10;

  T expected[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  fixed_width_column_wrapper<T> expected_w(expected, expected + num_els);

  auto result = cudf::sequence(num_els, init);

  expect_columns_equal(*result, expected_w);
}
