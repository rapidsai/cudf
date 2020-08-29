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

#include <cstdint>
#include <limits>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>
#include <tests/utilities/table_utilities.hpp>
#include <tests/utilities/type_lists.hpp>

#include <cudf/reshape.hpp>
#include "cudf/detail/reshape.hpp"


using namespace cudf::test;


class ByteCastTest : public cudf::test::BaseFixture {
};

TEST_F(ByteCastTest, PrimitiveValuesNoNulls)
{
  using limits16 = std::numeric_limits<int16_t>;
  // fixed_width_column_wrapper<int16_t> const int16_col({short(0), short(100), short(-100), limits16::min(), limits16::max()});
  fixed_width_column_wrapper<int16_t> const int16_col({short(0), short(100), short(-100), limits16::min(), limits16::max()});
  lists_column_wrapper<uint8_t> const int16_expected({{0x00, 0x00}, {0x00, 0x64}, {0xff, 0x9c}, {0x80, 0x00}, {0x7f, 0xff}});

  using limits32 = std::numeric_limits<int32_t>;
  fixed_width_column_wrapper<int32_t> const int32_col({0, 100, -100, limits32::min(), limits32::max()});
  lists_column_wrapper<uint8_t> const int32_expected({{0x00, 0x00, 0x00, 0x00}, {0x00, 0x00, 0x00, 0x64}, {0xff, 0xff, 0xff, 0x9c}, {0x80, 0x00, 0x00, 0x00}, {0x7f, 0xff, 0xff, 0xff}});

  using limits64 = std::numeric_limits<int64_t>;
  fixed_width_column_wrapper<int64_t> const int64_col({long(0), long(100), long(-100), limits64::min(), limits64::max()});
  lists_column_wrapper<uint8_t> const int64_expected({{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
                                                      {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x64},
                                                      {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x9c},
                                                      {0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
                                                      {0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}});

  using limitsfp32 = std::numeric_limits<float>;
  fixed_width_column_wrapper<float> const fp32_col({float(0.0), float(100.0), float(-100.0), limitsfp32::min(), limitsfp32::max()});
  lists_column_wrapper<uint8_t> const fp32_expected({{0x00, 0x00, 0x00, 0x00}, {0x42, 0xc8, 0x00, 0x00}, {0xc2, 0xc8, 0x00, 0x00}, {0x00, 0x80, 0x00, 0x00}, {0x7f, 0x7f, 0xff, 0xff}});

  using limitsfp64 = std::numeric_limits<double>;
  fixed_width_column_wrapper<double> const fp64_col({double(0.0), double(100.0), double(-100.0), limitsfp64::min(), limitsfp64::max()});
  lists_column_wrapper<uint8_t> const fp64_expected({{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
                                                     {0x40, 0x59, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
                                                     {0xc0, 0x59, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
                                                     {0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
                                                     {0x7f, 0xef, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}});
  
  auto const output_int16 = cudf::byte_cast(int16_col);
  auto const output_int32 = cudf::byte_cast(int32_col);
  auto const output_int64 = cudf::byte_cast(int64_col);
  auto const output_fp32 = cudf::byte_cast(fp32_col);
  auto const output_fp64 = cudf::byte_cast(fp64_col);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_int16->view(), int16_expected, true);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_int32->view(), int32_expected, true);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_int64->view(), int64_expected, true);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_fp32->view(), fp32_expected, true);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_fp64->view(), fp64_expected, true);
}

TEST_F(ByteCastTest, PrimitiveValuesWithNulls)
{
  
  auto even_valids = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? true : false; });
  auto odd_valids = cudf::test::make_counting_transform_iterator(
    0, [](auto i) { return i % 2 == 0 ? false : true; });

  using limits16 = std::numeric_limits<int16_t>;
  fixed_width_column_wrapper<int16_t> const int16_col({short(0), short(100), short(-100), limits16::min(), limits16::max()}, {0, 1, 0, 1, 0});
  lists_column_wrapper<uint8_t> const int16_expected({{0xee, 0xff}, {0x00, 0x64}, {0xee, 0xff}, {0x80, 0x00}, {0xee, 0xff}}, odd_valids);

  // using limits32 = std::numeric_limits<int32_t>;
  // fixed_width_column_wrapper<int32_t> const int32_col({0, 100, -100, limits32::min(), limits32::max()}, {1, 0, 1, 0, 1});
  // lists_column_wrapper<uint8_t> const int32_expected({{0x00, 0x00, 0x00, 0x00}, {0xcc, 0xdd, 0xee, 0xff}, {0xff, 0xff, 0xff, 0x9c}, {0xcc, 0xdd, 0xee, 0xff}, {0x7f, 0xff, 0xff, 0xff}}, even_valids);

  // using limits64 = std::numeric_limits<int64_t>;
  // fixed_width_column_wrapper<int64_t> const int64_col({long(0), long(100), long(-100), limits64::min(), limits64::max()}, {0, 1, 0, 1, 0});
  // lists_column_wrapper<uint8_t> const int64_expected({{0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff},
  //                                                     {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x64},
  //                                                     {0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff},
  //                                                     {0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
  //                                                     {0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff}}, odd_valids);

  // using limitsfp32 = std::numeric_limits<float>;
  // fixed_width_column_wrapper<float> const fp32_col({float(0.0), float(100.0), float(-100.0), limitsfp32::min(), limitsfp32::max()}, {1, 0, 1, 0, 1});
  // lists_column_wrapper<uint8_t> const fp32_expected({{0x00, 0x00, 0x00, 0x00}, {0xcc, 0xdd, 0xee, 0xff}, {0xc2, 0xc8, 0x00, 0x00}, {0xcc, 0xdd, 0xee, 0xff}, {0x7f, 0x7f, 0xff, 0xff}}, even_valids);

  // using limitsfp64 = std::numeric_limits<double>;
  // fixed_width_column_wrapper<double> const fp64_col({double(0.0), double(100.0), double(-100.0), limitsfp64::min(), limitsfp64::max()}, {0, 1, 0, 1, 0});
  // lists_column_wrapper<uint8_t> const fp64_expected({{0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff},
  //                                                    {0x40, 0x59, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
  //                                                    {0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff},
  //                                                    {0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
  //                                                    {0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff}}, odd_valids);
  
  auto const output_int16 = cudf::byte_cast(int16_col);
  // auto const output_int32 = cudf::byte_cast(int32_col);
  // auto const output_int64 = cudf::byte_cast(int64_col);
  // auto const output_fp32 = cudf::byte_cast(fp32_col);
  // auto const output_fp64 = cudf::byte_cast(fp64_col);

  // CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_int16->view(), int16_expected, true);
  // CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_int32->view(), int32_expected, true);
  // CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_int64->view(), int64_expected, true);
  // CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_fp32->view(), fp32_expected, true);
  // CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_fp64->view(), fp64_expected, true);
}

template <typename T>
class ByteCastTestFloatTyped : public cudf::test::BaseFixture {
};

TYPED_TEST_CASE(ByteCastTestFloatTyped, cudf::test::FloatingPointTypes);

TYPED_TEST(ByteCastTestFloatTyped, TestExtremes)
{
  using T = TypeParam;
  T min   = std::numeric_limits<T>::min();
  T max   = std::numeric_limits<T>::max();
  T nan   = std::numeric_limits<T>::quiet_NaN();
  T inf   = std::numeric_limits<T>::infinity();

  fixed_width_column_wrapper<T> const col1({T(0.0), T(100.0), T(-100.0), min, max, nan, inf, -inf});
  fixed_width_column_wrapper<T> const col2({T(-0.0), T(100.0), T(-100.0), min, max, -nan, inf, -inf});

  auto const output1 = cudf::byte_cast(col1);
  auto const output2 = cudf::byte_cast(col2);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view(), true);
}
