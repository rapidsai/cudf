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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <climits>
#include <type_traits>
#include <vector>

template <typename T>
inline auto make_data_type()
{
  return cudf::data_type{cudf::type_to_id<T>()};
}

template <typename T>
inline auto make_fixed_point_data_type(int32_t scale)
{
  return cudf::data_type{cudf::type_to_id<T>(), scale};
}

template <typename T>
struct FixedPointTests : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(FixedPointTests, cudf::test::FixedPointTypes);

TYPED_TEST(FixedPointTests, CastToDouble)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<double>;

  auto const input    = fp_wrapper{{1729, 17290, 172900, 1729000}, scale_type{-3}};
  auto const expected = fw_wrapper{1.729, 17.29, 172.9, 1729.0};
  auto const result   = cudf::cast(input, make_data_type<double>());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, CastToDoubleLarge)
{
  using namespace numeric;
  using namespace cudf::test;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<double>;

  auto begin          = make_counting_transform_iterator(0, [](auto i) { return 10 * (i + 0.5); });
  auto begin2         = make_counting_transform_iterator(0, [](auto i) { return i + 0.5; });
  auto const input    = fp_wrapper{begin, begin + 2000, scale_type{-1}};
  auto const expected = fw_wrapper(begin2, begin2 + 2000);
  auto const result   = cudf::cast(input, make_data_type<double>());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, CastToInt32)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<int32_t>;

  auto const input    = fp_wrapper{{1729, 17290, 172900, 1729000}, scale_type{-3}};
  auto const expected = fw_wrapper{1, 17, 172, 1729};
  auto const result   = cudf::cast(input, make_data_type<int32_t>());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, CastToIntLarge)
{
  using namespace numeric;
  using namespace cudf::test;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<int32_t>;

  auto begin          = thrust::make_counting_iterator(0);
  auto begin2         = make_counting_transform_iterator(0, [](auto i) { return 10 * i; });
  auto const input    = fp_wrapper{begin, begin + 2000, scale_type{1}};
  auto const expected = fw_wrapper(begin2, begin2 + 2000);
  auto const result   = cudf::cast(input, make_data_type<int32_t>());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, CastFromDouble)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<double>;

  auto const input    = fw_wrapper{1.729, 17.29, 172.9, 1729.0};
  auto const expected = fp_wrapper{{1729, 17290, 172900, 1729000}, scale_type{-3}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimalXX>(-3));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, CastFromDoubleLarge)
{
  using namespace numeric;
  using namespace cudf::test;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<double>;

  auto begin          = make_counting_transform_iterator(0, [](auto i) { return i + 0.5; });
  auto begin2         = make_counting_transform_iterator(0, [](auto i) { return 10 * (i + 0.5); });
  auto const input    = fw_wrapper(begin, begin + 2000);
  auto const expected = fp_wrapper{begin2, begin2 + 2000, scale_type{-1}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimalXX>(-1));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, CastFromInt)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<int32_t>;

  auto const input    = fw_wrapper{1729, 172, 17, 1};
  auto const expected = fp_wrapper{{17, 1, 0, 0}, scale_type{2}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimalXX>(2));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, CastFromIntLarge)
{
  using namespace numeric;
  using namespace cudf::test;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;
  using fw_wrapper = cudf::test::fixed_width_column_wrapper<int32_t>;

  auto begin          = make_counting_transform_iterator(0, [](auto i) { return 1000 * i; });
  auto begin2         = thrust::make_counting_iterator(0);
  auto const input    = fw_wrapper(begin, begin + 2000);
  auto const expected = fp_wrapper{begin2, begin2 + 2000, scale_type{3}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimalXX>(3));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, FixedPointToFixedPointSameTypeidUp)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper{{1729, 17290, 172900, 1729000}, scale_type{-3}};
  auto const expected = fp_wrapper{{172, 1729, 17290, 172900}, scale_type{-2}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimalXX>(-2));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, FixedPointToFixedPointSameTypeidDown)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper{{1729, 17290, 172900, 1729000}, scale_type{-3}};
  auto const expected = fp_wrapper{{17290, 172900, 1729000, 17290000}, scale_type{-4}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimalXX>(-4));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, FixedPointToFixedPointSameTypeidUpPositive)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper{{1, 12, 123, 1234, 12345, 123456}, scale_type{1}};
  auto const expected = fp_wrapper{{0, 1, 12, 123, 1234, 12345}, scale_type{2}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimalXX>(2));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, FixedPointToFixedPointSameTypeidEmpty)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper{{}, scale_type{1}};
  auto const expected = fp_wrapper{{}, scale_type{2}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimalXX>(2));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, FixedPointToFixedPointSameTypeidDownPositive)
{
  using namespace numeric;
  using decimalXX  = TypeParam;
  using RepType    = cudf::device_storage_type_t<decimalXX>;
  using fp_wrapper = cudf::test::fixed_point_column_wrapper<RepType>;

  auto const input    = fp_wrapper{{0, 1, 12, 123, 1234}, scale_type{2}};
  auto const expected = fp_wrapper{{0, 1000, 12000, 123000, 1234000}, scale_type{-1}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimalXX>(-1));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, FixedPointToFixedPointDifferentTypeid)
{
  using namespace numeric;
  using decimalA    = TypeParam;
  using RepTypeA    = cudf::device_storage_type_t<decimalA>;
  using RepTypeB    = std::conditional_t<std::is_same<RepTypeA, int32_t>::value, int64_t, int32_t>;
  using fp_wrapperA = cudf::test::fixed_point_column_wrapper<RepTypeA>;
  using fp_wrapperB = cudf::test::fixed_point_column_wrapper<RepTypeB>;

  auto const input    = fp_wrapperB{{1729, 17290, 172900, 1729000}, scale_type{-3}};
  auto const expected = fp_wrapperA{{1729, 17290, 172900, 1729000}, scale_type{-3}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimalA>(-3));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, FixedPointToFixedPointDifferentTypeidDown)
{
  using namespace numeric;
  using decimalA    = TypeParam;
  using RepTypeA    = cudf::device_storage_type_t<decimalA>;
  using RepTypeB    = std::conditional_t<std::is_same<RepTypeA, int32_t>::value, int64_t, int32_t>;
  using fp_wrapperA = cudf::test::fixed_point_column_wrapper<RepTypeA>;
  using fp_wrapperB = cudf::test::fixed_point_column_wrapper<RepTypeB>;

  auto const input    = fp_wrapperB{{1729, 17290, 172900, 1729000}, scale_type{-3}};
  auto const expected = fp_wrapperA{{172900, 1729000, 17290000, 172900000}, scale_type{-5}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimalA>(-5));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}

TYPED_TEST(FixedPointTests, FixedPointToFixedPointDifferentTypeidUp)
{
  using namespace numeric;
  using decimalA    = TypeParam;
  using RepTypeA    = cudf::device_storage_type_t<decimalA>;
  using RepTypeB    = std::conditional_t<std::is_same<RepTypeA, int32_t>::value, int64_t, int32_t>;
  using fp_wrapperA = cudf::test::fixed_point_column_wrapper<RepTypeA>;
  using fp_wrapperB = cudf::test::fixed_point_column_wrapper<RepTypeB>;

  auto const input    = fp_wrapperB{{1729, 17290, 172900, 1729000}, scale_type{-3}};
  auto const expected = fp_wrapperA{{1, 17, 172, 1729}, scale_type{0}};
  auto const result   = cudf::cast(input, make_fixed_point_data_type<decimalA>(0));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, result->view());
}
