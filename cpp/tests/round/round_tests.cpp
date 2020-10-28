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

#include <cudf/round.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <limits>

struct RoundTests : public cudf::test::BaseFixture {
};

template <typename T>
struct RoundTestsIntegerTypes : public cudf::test::BaseFixture {
};

template <typename T>
struct RoundTestsFloatingPointTypes : public cudf::test::BaseFixture {
};

using IntegerTypes = cudf::test::Types<int32_t, int64_t>;

TYPED_TEST_CASE(RoundTestsIntegerTypes, IntegerTypes);
TYPED_TEST_CASE(RoundTestsFloatingPointTypes, cudf::test::FloatingPointTypes);

TYPED_TEST(RoundTestsIntegerTypes, SimpleFixedPointTest)
{
  using namespace numeric;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<TypeParam>;

  auto const input    = fp_wrapper{{1140, 1150, 1160}, scale_type{-3}};
  auto const expected = fp_wrapper{{11, 12, 12}, scale_type{-1}};

  EXPECT_THROW(cudf::round(input, 1, cudf::round_option::HALF_UP), cudf::logic_error);

  // enable in follow up PR
  // auto const result   = cudf::round(col, 1, cudf::round_option::HALF_UP);
  // CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFloatingPointTypes, SimpleFloatingPointTest0)
{
  using namespace numeric;
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const input    = fw_wrapper{1.4, 1.5, 1.6};
  auto const expected = fw_wrapper{1, 2, 2};
  auto const result   = cudf::round(input, 0, cudf::round_option::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFloatingPointTypes, SimpleFloatingPointTest1)
{
  using namespace numeric;
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const input    = fw_wrapper{1.24, 1.25, 1.26};
  auto const expected = fw_wrapper{1.2, 1.3, 1.3};
  auto const result   = cudf::round(input, 1, cudf::round_option::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFloatingPointTypes, SimpleFloatingPointTestNeg1)
{
  using namespace numeric;
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const input    = fw_wrapper{12, 135, 1454, 1455, 1456};
  auto const expected = fw_wrapper{10, 140, 1450, 1460, 1460};
  auto const result   = cudf::round(input, -1, cudf::round_option::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFloatingPointTypes, SimpleFloatingPointTestNeg2)
{
  using namespace numeric;
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const input    = fw_wrapper{12, 135, 1454, 1455, 1500};
  auto const expected = fw_wrapper{0, 100, 1500, 1500, 1500};
  auto const result   = cudf::round(input, -2, cudf::round_option::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsIntegerTypes, SimpleIntegerTestNeg2)
{
  using namespace numeric;
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const input    = fw_wrapper{12, 135, 1454, 1455, 1500};
  auto const expected = fw_wrapper{0, 100, 1500, 1500, 1500};
  auto const result   = cudf::round(input, -2, cudf::round_option::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsIntegerTypes, SimpleNegativeInteger)
{
  using namespace numeric;
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const input    = fw_wrapper{-12, -135, -1454, -1455, -1500};
  auto const expected = fw_wrapper{0, -100, -1500, -1500, -1500};
  auto const result   = cudf::round(input, -2, cudf::round_option::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(RoundTests, Int64AtBoundary)
{
  using namespace numeric;
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;

  auto const m     = std::numeric_limits<int64_t>::max();  // 9223372036854775807
  auto const input = fw_wrapper{m};

  auto const expected = fw_wrapper{9223372036854775800};
  auto const result   = cudf::round(input, -2, cudf::round_option::HALF_UP);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());

  auto const expected2 = fw_wrapper{9223372036850000000};
  auto const result2   = cudf::round(input, -7, cudf::round_option::HALF_UP);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected2, result2->view());

  auto const expected3 = fw_wrapper{9223372000000000000};
  auto const result3   = cudf::round(input, -11, cudf::round_option::HALF_UP);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected3, result3->view());

  auto const result4 = cudf::round(input, -12, cudf::round_option::HALF_UP);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected3, result4->view());

  auto const expected5 = fw_wrapper{9000000000000000000};
  auto const result5   = cudf::round(input, -18, cudf::round_option::HALF_UP);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected5, result5->view());
}

CUDF_TEST_PROGRAM_MAIN()
