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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/filling.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/unary.hpp>

using namespace cudf;
using namespace cudf::test;

template <typename T>
class SequenceTypedTestFixture : public cudf::test::BaseFixture {
};

class SequenceTestFixture : public cudf::test::BaseFixture {
};

using NumericTypesNoBool = cudf::test::Types<int8_t, int16_t, int32_t, int64_t, float, double>;

TYPED_TEST_SUITE(SequenceTypedTestFixture, NumericTypesNoBool);

TYPED_TEST(SequenceTypedTestFixture, Incrementing)
{
  using T = TypeParam;

  numeric_scalar<T> init(0);
  numeric_scalar<T> step(1);

  size_type num_els = 10;

  T expected[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  fixed_width_column_wrapper<T> expected_w(expected, expected + num_els);

  auto result = cudf::sequence(num_els, init, step);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected_w);
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

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected_w);
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

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected_w);
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

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expected_w);
}

TEST_F(SequenceTestFixture, DateSequenceBasic)
{
  // Timestamp generated using https://www.epochconverter.com/
  timestamp_scalar<timestamp_s> init(1629852896L, true);  // 2021-08-25 00:54:56 GMT
  size_type size{5};
  size_type months{1};

  fixed_width_column_wrapper<timestamp_s, int64_t> expected{
    1629852896L,  // 2021-08-25 00:54:56 GMT
    1632531296L,  // 2021-09-25 00:54:56 GMT
    1635123296L,  // 2021-10-25 00:54:56 GMT
    1637801696L,  // 2021-11-25 00:54:56 GMT
    1640393696L,  // 2021-12-25 00:54:56 GMT
  };

  auto got = calendrical_month_sequence(size, init, months);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *got);
}

TEST_F(SequenceTestFixture, DateSequenceLeapYear)
{
  // Timestamp generated using https://www.epochconverter.com/
  timestamp_scalar<timestamp_s> init(951876379L, true);  // 2000-02-29 02:06:19 GMT
  size_type size{5};
  size_type months{12};

  fixed_width_column_wrapper<timestamp_s, int64_t> expected{
    951876379L,   // 2000-02-29 02:06:19 GMT Leap Year
    983412379L,   // 2001-02-28 02:06:19 GMT
    1014948379L,  // 2002-02-28 02:06:19 GMT
    1046484379L,  // 2003-02-28 02:06:19 GMT
    1078106779L,  // 2004-02-29 02:06:19 GMT Leap Year
  };

  auto got = calendrical_month_sequence(size, init, months);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *got);
}

TEST_F(SequenceTestFixture, DateSequenceBadTypes)
{
  numeric_scalar<int64_t> init(951876379, true);
  size_type size   = 5;
  size_type months = 12;

  EXPECT_THROW(calendrical_month_sequence(size, init, months), cudf::logic_error);
}
