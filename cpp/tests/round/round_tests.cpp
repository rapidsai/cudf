/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/round.hpp>

#include <limits>

struct RoundTests : public cudf::test::BaseFixture {
};

template <typename T>
struct RoundTestsIntegerTypes : public cudf::test::BaseFixture {
};

template <typename T>
struct RoundTestsFixedPointTypes : public cudf::test::BaseFixture {
};

template <typename T>
struct RoundTestsFloatingPointTypes : public cudf::test::BaseFixture {
};

using IntegerTypes = cudf::test::Types<int16_t, int32_t, int64_t>;

TYPED_TEST_CASE(RoundTestsIntegerTypes, IntegerTypes);
TYPED_TEST_CASE(RoundTestsFixedPointTypes, cudf::test::FixedPointTypes);
TYPED_TEST_CASE(RoundTestsFloatingPointTypes, cudf::test::FloatingPointTypes);

TYPED_TEST(RoundTestsFixedPointTypes, SimpleFixedPointTestHalfUpZero)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper{{1140, 1150, 1160, 1240, 1250, 1260}, scale_type{-2}};
  auto const expected = fp_wrapper{{11, 12, 12, 12, 13, 13}, scale_type{0}};
  auto const result   = cudf::round(input);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFixedPointTypes, SimpleFixedPointTestHalfUpZeroNoOp)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input  = fp_wrapper{{1125, 1150, 1160, 1240, 1250, 1260}, scale_type{0}};
  auto const result = cudf::round(input);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, result->view());
}

TYPED_TEST(RoundTestsFixedPointTypes, SimpleFixedPointTestHalfUpZeroNullMask)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const null_mask = std::vector<int>{1, 1, 0, 1};
  auto const input     = fp_wrapper{{1150, 1160, 1240, 1250}, null_mask.cbegin(), scale_type{-2}};
  auto const expected  = fp_wrapper{{12, 12, 1240, 13}, null_mask.cbegin(), scale_type{0}};
  auto const result    = cudf::round(input);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFixedPointTypes, SimpleFixedPointTestHalfEvenZero)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper{{1140, 1150, 1160, 1240, 1250, 1260}, scale_type{-2}};
  auto const expected = fp_wrapper{{11, 12, 12, 12, 12, 13}, scale_type{0}};
  auto const result   = cudf::round(input, 0, cudf::rounding_method::HALF_EVEN);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFixedPointTypes, SimpleFixedPointTestHalfUp)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper{{1140, 1150, 1160}, scale_type{-3}};
  auto const expected = fp_wrapper{{11, 12, 12}, scale_type{-1}};
  auto const result   = cudf::round(input, 1, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFixedPointTypes, SimpleFixedPointTestHalfUp2)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper{{114, 115, 116}, scale_type{-2}};
  auto const expected = fp_wrapper{{11, 12, 12}, scale_type{-1}};
  auto const result   = cudf::round(input, 1, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFixedPointTypes, SimpleFixedPointTestHalfUp3)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper{{1, 2, 3}, scale_type{1}};
  auto const expected = fp_wrapper{{100, 200, 300}, scale_type{-1}};
  auto const result   = cudf::round(input, 1, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFixedPointTypes, SimpleFixedPointTestHalfEven)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper{{1140, 1150, 1160, 1240, 1250, 1260}, scale_type{-3}};
  auto const expected = fp_wrapper{{11, 12, 12, 12, 12, 13}, scale_type{-1}};
  auto const result   = cudf::round(input, 1, cudf::rounding_method::HALF_EVEN);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFixedPointTypes, SimpleFixedPointTestHalfEven2)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper{{114, 115, 116, 124, 125, 126}, scale_type{-2}};
  auto const expected = fp_wrapper{{11, 12, 12, 12, 12, 13}, scale_type{-1}};
  auto const result   = cudf::round(input, 1, cudf::rounding_method::HALF_EVEN);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFixedPointTypes, SimpleFixedPointTestHalfEven3)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper{{1, 2, 3}, scale_type{1}};
  auto const expected = fp_wrapper{{100, 200, 300}, scale_type{-1}};
  auto const result   = cudf::round(input, 1, cudf::rounding_method::HALF_EVEN);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFixedPointTypes, EmptyFixedPointTypeTest)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input         = fp_wrapper{{}, scale_type{1}};
  auto const expected      = fp_wrapper{{}, scale_type{-1}};
  auto const expected_type = cudf::data_type{cudf::type_to_id<decimalXX>(), scale_type{-1}};
  auto const result        = cudf::round(input, 1, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
  EXPECT_EQ(result->view().type(), expected_type);
}

TYPED_TEST(RoundTestsFixedPointTypes, SimpleFixedPointTestNegHalfUp)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper{{14, 15, 16, 24, 25, 26}, scale_type{2}};
  auto const expected = fp_wrapper{{1, 2, 2, 2, 3, 3}, scale_type{3}};
  auto const result   = cudf::round(input, -3, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFixedPointTypes, SimpleFixedPointTestNegHalfUp2)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper{{14, 15, 16, 24, 25, 26}, scale_type{3}};
  auto const expected = fp_wrapper{{1, 2, 2, 2, 3, 3}, scale_type{4}};
  auto const result   = cudf::round(input, -4, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFixedPointTypes, SimpleFixedPointTestHalfNegUp3)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper{{1, 2, 3}, scale_type{2}};
  auto const expected = fp_wrapper{{10, 20, 30}, scale_type{1}};
  auto const result   = cudf::round(input, -1, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFixedPointTypes, SimpleFixedPointTestNegHalfEven)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper{{14, 15, 16, 24, 25, 26}, scale_type{2}};
  auto const expected = fp_wrapper{{1, 2, 2, 2, 2, 3}, scale_type{3}};
  auto const result   = cudf::round(input, -3, cudf::rounding_method::HALF_EVEN);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFixedPointTypes, SimpleFixedPointTestNegHalfEven2)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper{{14, 15, 16, 24, 25, 26}, scale_type{3}};
  auto const expected = fp_wrapper{{1, 2, 2, 2, 2, 3}, scale_type{4}};
  auto const result   = cudf::round(input, -4, cudf::rounding_method::HALF_EVEN);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFixedPointTypes, SimpleFixedPointTestHalfNegEven3)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper{{1, 2, 3}, scale_type{2}};
  auto const expected = fp_wrapper{{10, 20, 30}, scale_type{1}};
  auto const result   = cudf::round(input, -1, cudf::rounding_method::HALF_EVEN);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFloatingPointTypes, SimpleFloatingPointTestHalfUp0)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const input    = fw_wrapper{1.4, 1.5, 1.6};
  auto const expected = fw_wrapper{1, 2, 2};
  auto const result   = cudf::round(input);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFloatingPointTypes, SimpleFloatingPointTestHalfUp1)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const input    = fw_wrapper{1.24, 1.25, 1.26};
  auto const expected = fw_wrapper{1.2, 1.3, 1.3};
  auto const result   = cudf::round(input, 1, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFloatingPointTypes, SimpleFloatingPointTestHalfEven0)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const input    = fw_wrapper{1.4, 1.5, 1.6, 2.4, 2.5, 2.6};
  auto const expected = fw_wrapper{1, 2, 2, 2, 2, 3};
  auto const result   = cudf::round(input, 0, cudf::rounding_method::HALF_EVEN);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFloatingPointTypes, SimpleFloatingPointTestHalfEven1)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const input    = fw_wrapper{1.24, 1.25, 1.26, 1.34, 1.35, 1.36};
  auto const expected = fw_wrapper{1.2, 1.2, 1.3, 1.3, 1.4, 1.4};
  auto const result   = cudf::round(input, 1, cudf::rounding_method::HALF_EVEN);
}

TYPED_TEST(RoundTestsFloatingPointTypes, SimpleFloatingPointTestWithNullsHalfUp)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const input    = fw_wrapper{{1.24, 1.25, 1.26}, {1, 0, 1}};
  auto const expected = fw_wrapper{{1.2, 1.3, 1.3}, {1, 0, 1}};
  auto const result   = cudf::round(input, 1, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFloatingPointTypes, SimpleFloatingPointTestNegHalfUp1)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const input    = fw_wrapper{5, 12, 15, 135, 1454, 1455, 1456};
  auto const expected = fw_wrapper{10, 10, 20, 140, 1450, 1460, 1460};
  auto const result   = cudf::round(input, -1, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFloatingPointTypes, SimpleFloatingPointTestNegHalfEven1)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const input    = fw_wrapper{5, 12, 15, 135, 1454, 1455, 1456};
  auto const expected = fw_wrapper{0, 10, 20, 140, 1450, 1460, 1460};
  auto const result   = cudf::round(input, -1, cudf::rounding_method::HALF_EVEN);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFloatingPointTypes, SimpleFloatingPointTestNegHalfUp2)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const input    = fw_wrapper{12, 135, 1454, 1455, 1500};
  auto const expected = fw_wrapper{0, 100, 1500, 1500, 1500};
  auto const result   = cudf::round(input, -2, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFloatingPointTypes, LargeFloatingPointHalfUp)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto transform   = [](int i) -> float { return i % 2 == 0 ? i + 0.44 : i + 0.56; };
  auto begin       = cudf::detail::make_counting_transform_iterator(0, transform);
  auto const input = fw_wrapper(begin, begin + 2000);

  auto transform2     = [](int i) { return i % 2 == 0 ? i + 0.4 : i + 0.6; };
  auto begin2         = cudf::detail::make_counting_transform_iterator(0, transform2);
  auto const expected = fw_wrapper(begin2, begin2 + 2000);

  auto const result = cudf::round(input, 1, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsIntegerTypes, LargeIntegerHalfEven)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto transform   = [](int i) -> float { return 10 * i + 5; };
  auto begin       = cudf::detail::make_counting_transform_iterator(1, transform);
  auto const input = fw_wrapper(begin, begin + 2000);

  auto transform2     = [](int i) { return i % 2 == 0 ? 10 * i : 10 + 10 * i; };
  auto begin2         = cudf::detail::make_counting_transform_iterator(1, transform2);
  auto const expected = fw_wrapper(begin2, begin2 + 2000);

  auto const result = cudf::round(input, -1, cudf::rounding_method::HALF_EVEN);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsFloatingPointTypes, SameSignificatDigitsHalfUp)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const input    = fw_wrapper{9.87654321};
  auto const expected = fw_wrapper{9.88};
  auto const result   = cudf::round(input, 2, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());

  auto const input2    = fw_wrapper{987.654321};
  auto const expected2 = fw_wrapper{988};
  auto const result2   = cudf::round(input2);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected2, result2->view());

  auto const input3    = fw_wrapper{987654.321};
  auto const expected3 = fw_wrapper{988000};
  auto const result3   = cudf::round(input3, -3, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected3, result3->view());

  auto const input4    = fw_wrapper{9876543.21};
  auto const expected4 = fw_wrapper{9880000};
  auto const result4   = cudf::round(input4, -4, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected4, result4->view());

  auto const input5    = fw_wrapper{0.0000987654321};
  auto const expected5 = fw_wrapper{0.0000988};
  auto const result5   = cudf::round(input5, 7, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected5, result5->view());
}

TEST_F(RoundTests, FloatMaxTestHalfUp)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<float>;

  auto const input    = fw_wrapper{std::numeric_limits<float>::max()};  // 3.40282e+38
  auto const expected = fw_wrapper{3.4e+38};
  auto const result   = cudf::round(input, -37, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(RoundTests, DoubleMaxTestHalfUp)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<double>;

  auto const input    = fw_wrapper{std::numeric_limits<double>::max() - 5e+306};  // 1.74769e+308
  auto const expected = fw_wrapper{1.7e+308};
  auto const result   = cudf::round(input, -307, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(RoundTests, AvoidOverFlowTestHalfUp)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<double>;

  auto const input    = fw_wrapper{std::numeric_limits<double>::max()};
  auto const expected = fw_wrapper{std::numeric_limits<double>::max()};
  auto const result   = cudf::round(input, 2, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsIntegerTypes, SimpleIntegerTestNegHalfUp2)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const input    = fw_wrapper{12, 135, 1454, 1455, 1500};
  auto const expected = fw_wrapper{0, 100, 1500, 1500, 1500};
  auto const result   = cudf::round(input, -2, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsIntegerTypes, SimpleIntegerTestNegHalfEven)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const input    = fw_wrapper{12, 135, 1450, 1550, 1650};
  auto const expected = fw_wrapper{0, 100, 1400, 1600, 1600};
  auto const result   = cudf::round(input, -2, cudf::rounding_method::HALF_EVEN);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsIntegerTypes, SimpleNegativeIntegerHalfUp)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const input    = fw_wrapper{-12, -135, -1454, -1455, -1500, -140, -150, -160};
  auto const expected = fw_wrapper{0, -100, -1500, -1500, -1500, -100, -200, -200};
  auto const result   = cudf::round(input, -2, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsIntegerTypes, SimpleNegativeIntegerHalfEven)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const input    = fw_wrapper{-12, -135, -145, -146, -1454, -1455, -1500};
  auto const expected = fw_wrapper{-10, -140, -140, -150, -1450, -1460, -1500};
  auto const result   = cudf::round(input, -1, cudf::rounding_method::HALF_EVEN);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(RoundTests, SimpleNegativeIntegerWithUnsignedNumbersHalfUp)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<uint32_t>;

  auto const input    = fw_wrapper{12, 135, 1454, 1455, 1500, 140, 150, 160};
  auto const expected = fw_wrapper{0, 100, 1500, 1500, 1500, 100, 200, 200};
  auto const result   = cudf::round(input, -2, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(RoundTests, SimpleNegativeInt8HalfEven)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<int8_t>;

  auto const input    = fw_wrapper{12, 35, 36, 15, 16, 24, 25, 26};
  auto const expected = fw_wrapper{10, 40, 40, 20, 20, 20, 20, 30};
  auto const result   = cudf::round(input, -1, cudf::rounding_method::HALF_EVEN);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(RoundTests, SimpleNegativeInt8HalfUp)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<int8_t>;

  auto const input    = fw_wrapper{12, 35, 36, 15, 16, 24, 25, 26};
  auto const expected = fw_wrapper{10, 40, 40, 20, 20, 20, 30, 30};
  auto const result   = cudf::round(input, -1, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(RoundTestsIntegerTypes, SimplePositiveIntegerHalfUp)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<TypeParam>;

  auto const input    = fw_wrapper{-12, -135, -1454, -1455, -1500};
  auto const expected = fw_wrapper{-12, -135, -1454, -1455, -1500};
  auto const result   = cudf::round(input, 2, cudf::rounding_method::HALF_UP);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TEST_F(RoundTests, Int64AtBoundaryHalfUp)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<int64_t>;

  auto const m     = std::numeric_limits<int64_t>::max();  // 9223372036854775807
  auto const input = fw_wrapper{m};

  auto const expected = fw_wrapper{9223372036854775800};
  auto const result   = cudf::round(input, -2, cudf::rounding_method::HALF_UP);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());

  auto const expected2 = fw_wrapper{9223372036850000000};
  auto const result2   = cudf::round(input, -7, cudf::rounding_method::HALF_UP);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected2, result2->view());

  auto const expected3 = fw_wrapper{9223372000000000000};
  auto const result3   = cudf::round(input, -11, cudf::rounding_method::HALF_UP);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected3, result3->view());

  auto const result4 = cudf::round(input, -12, cudf::rounding_method::HALF_UP);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected3, result4->view());

  auto const expected5 = fw_wrapper{9000000000000000000};
  auto const result5   = cudf::round(input, -18, cudf::rounding_method::HALF_UP);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected5, result5->view());
}

TEST_F(RoundTests, BoolTestHalfUp)
{
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<bool>;

  auto const input = fw_wrapper{0, 1, 0};
  EXPECT_THROW(cudf::round(input, -2, cudf::rounding_method::HALF_UP), cudf::logic_error);
}

CUDF_TEST_PROGRAM_MAIN()
