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

#include <cudf/reshape.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/type_lists.hpp>

using namespace cudf::test;

class ByteCastTest : public cudf::test::BaseFixture {
};

TEST_F(ByteCastTest, int16ValuesWithSplit)
{
  using limits = std::numeric_limits<int16_t>;
  fixed_width_column_wrapper<int16_t> const int16_col(
    {short(0), short(100), short(-100), limits::min(), limits::max()});
  lists_column_wrapper<uint8_t> const int16_expected(
    {{0x00, 0x00}, {0x64, 0x00}, {0x9c, 0xff}, {0x00, 0x80}, {0xff, 0x7f}});
  lists_column_wrapper<uint8_t> const int16_expected_slice1(
    {{0x00, 0x00}, {0x00, 0x64}, {0xff, 0x9c}});
  lists_column_wrapper<uint8_t> const int16_expected_slice2({{0x80, 0x00}, {0x7f, 0xff}});

  std::vector<cudf::size_type> splits({3});
  std::vector<cudf::column_view> split_column = cudf::split(int16_col, splits);

  auto const output_int16        = cudf::byte_cast(int16_col, cudf::flip_endianness::NO);
  auto const output_int16_slice1 = cudf::byte_cast(split_column.at(0), cudf::flip_endianness::YES);
  auto const output_int16_slice2 = cudf::byte_cast(split_column.at(1), cudf::flip_endianness::YES);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_int16->view(), int16_expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_int16_slice1->view(), int16_expected_slice1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_int16_slice2->view(), int16_expected_slice2);
}

TEST_F(ByteCastTest, int16ValuesWithNulls)
{
  using limits      = std::numeric_limits<int16_t>;
  auto odd_validity = make_counting_transform_iterator(0, [](auto i) { return i % 2; });

  fixed_width_column_wrapper<int16_t> const int16_col(
    {short(0), short(100), short(-100), limits::min(), limits::max()}, {0, 1, 0, 1, 0});
  /* CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT compares underlying values even when specified as null,
   * resulting in erroneous test failures. The commented out data tests the case where underlying
   * values are different, but are both null. */
  // auto int16_data =
  //   fixed_width_column_wrapper<uint8_t>{0xee, 0xff, 0x00, 0x64, 0xee, 0xff, 0x80, 0x00, 0xee,
  //   0xff};
  auto int16_data =
    fixed_width_column_wrapper<uint8_t>{0x00, 0x00, 0x00, 0x64, 0xff, 0x9c, 0x80, 0x00, 0x7f, 0xff};

  auto int16_expected = cudf::make_lists_column(
    5,
    std::move(fixed_width_column_wrapper<cudf::size_type>{0, 2, 4, 6, 8, 10}.release()),
    std::move(int16_data.release()),
    3,
    detail::make_null_mask(odd_validity, odd_validity + 5));

  auto const output_int16 = cudf::byte_cast(int16_col, cudf::flip_endianness::YES);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(output_int16->view(), int16_expected->view());
}

TEST_F(ByteCastTest, int32Values)
{
  using limits = std::numeric_limits<int32_t>;
  fixed_width_column_wrapper<int32_t> const int32_col({0, 100, -100, limits::min(), limits::max()});
  lists_column_wrapper<uint8_t> const int32_expected_flipped({{0x00, 0x00, 0x00, 0x00},
                                                              {0x00, 0x00, 0x00, 0x64},
                                                              {0xff, 0xff, 0xff, 0x9c},
                                                              {0x80, 0x00, 0x00, 0x00},
                                                              {0x7f, 0xff, 0xff, 0xff}});
  lists_column_wrapper<uint8_t> const int32_expected({{0x00, 0x00, 0x00, 0x00},
                                                      {0x64, 0x00, 0x00, 0x00},
                                                      {0x9c, 0xff, 0xff, 0xff},
                                                      {0x00, 0x00, 0x00, 0x80},
                                                      {0xff, 0xff, 0xff, 0x7f}});

  auto const output_int32_flipped = cudf::byte_cast(int32_col, cudf::flip_endianness::YES);
  auto const output_int32         = cudf::byte_cast(int32_col, cudf::flip_endianness::NO);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_int32_flipped->view(), int32_expected_flipped);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_int32->view(), int32_expected);
}

TEST_F(ByteCastTest, int32ValuesWithNulls)
{
  using limits       = std::numeric_limits<int32_t>;
  auto even_validity = make_counting_transform_iterator(0, [](auto i) { return (i + 1) % 2; });

  fixed_width_column_wrapper<int32_t> const int32_col({0, 100, -100, limits::min(), limits::max()},
                                                      {1, 0, 1, 0, 1});
  /* Data commented out below explained by comment in int16ValuesWithNulls test */
  // auto int32_data =
  //   fixed_width_column_wrapper<uint8_t>{0x00, 0x00, 0x00, 0x00, 0xcc, 0xdd, 0xee, 0xff, 0xff,
  //   0xff,
  //                                       0xff, 0x9c, 0xcc, 0xdd, 0xee, 0xff, 0x7f, 0xff, 0xff,
  //                                       0xff};
  auto int32_data =
    fixed_width_column_wrapper<uint8_t>{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x64, 0xff, 0xff,
                                        0xff, 0x9c, 0x80, 0x00, 0x00, 0x00, 0x7f, 0xff, 0xff, 0xff};
  auto int32_expected = cudf::make_lists_column(
    5,
    std::move(fixed_width_column_wrapper<cudf::size_type>{0, 4, 8, 12, 16, 20}.release()),
    std::move(int32_data.release()),
    3,
    detail::make_null_mask(even_validity, even_validity + 5));

  auto const output_int32 = cudf::byte_cast(int32_col, cudf::flip_endianness::YES);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(output_int32->view(), int32_expected->view());
}

TEST_F(ByteCastTest, int64ValuesWithSplit)
{
  using limits = std::numeric_limits<int64_t>;
  fixed_width_column_wrapper<int64_t> const int64_col(
    {long(0), long(100), long(-100), limits::min(), limits::max()});
  lists_column_wrapper<uint8_t> const int64_expected_flipped(
    {{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
     {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x64},
     {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x9c},
     {0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
     {0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}});
  lists_column_wrapper<uint8_t> const int64_expected_slice1(
    {{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
     {0x64, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
     {0x9c, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff}});
  lists_column_wrapper<uint8_t> const int64_expected_slice2(
    {{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x80},
     {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x7f}});

  std::vector<cudf::size_type> splits({3});
  std::vector<cudf::column_view> split_column = cudf::split(int64_col, splits);

  auto const output_int64_flipped = cudf::byte_cast(int64_col, cudf::flip_endianness::YES);
  auto const output_int64_slice1  = cudf::byte_cast(split_column.at(0), cudf::flip_endianness::NO);
  auto const output_int64_slice2  = cudf::byte_cast(split_column.at(1), cudf::flip_endianness::NO);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_int64_flipped->view(), int64_expected_flipped);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_int64_slice1->view(), int64_expected_slice1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_int64_slice2->view(), int64_expected_slice2);
}

TEST_F(ByteCastTest, int64ValuesWithNulls)
{
  using limits      = std::numeric_limits<int64_t>;
  auto odd_validity = make_counting_transform_iterator(0, [](auto i) { return i % 2; });

  fixed_width_column_wrapper<int64_t> const int64_col(
    {long(0), long(100), long(-100), limits::min(), limits::max()}, {0, 1, 0, 1, 0});
  /* Data commented out below explained by comment in int16ValuesWithNulls test */
  // auto int64_data = fixed_width_column_wrapper<uint8_t>{
  //   0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  //   0x00, 0x64, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x80, 0x00, 0x00, 0x00,
  //   0x00, 0x00, 0x00, 0x00, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff};
  auto int64_data = fixed_width_column_wrapper<uint8_t>{
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x64, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x9c, 0x80, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x7f, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
  auto int64_expected = cudf::make_lists_column(
    5,
    std::move(fixed_width_column_wrapper<cudf::size_type>{0, 8, 16, 24, 32, 40}.release()),
    std::move(int64_data.release()),
    3,
    detail::make_null_mask(odd_validity, odd_validity + 5));

  auto const output_int64 = cudf::byte_cast(int64_col, cudf::flip_endianness::YES);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(output_int64->view(), int64_expected->view());
}

TEST_F(ByteCastTest, fp32ValuesWithSplit)
{
  using limits = std::numeric_limits<float>;
  float nan    = limits::quiet_NaN();
  float inf    = limits::infinity();
  fixed_width_column_wrapper<float> const fp32_col(
    {float(0.0), float(100.0), float(-100.0), limits::min(), limits::max(), nan, -nan, inf, -inf});
  lists_column_wrapper<uint8_t> const fp32_expected({{0x00, 0x00, 0x00, 0x00},
                                                     {0x00, 0x00, 0xc8, 0x42},
                                                     {0x00, 0x00, 0xc8, 0xc2},
                                                     {0x00, 0x00, 0x80, 0x00},
                                                     {0xff, 0xff, 0x7f, 0x7f},
                                                     {0x00, 0x00, 0xc0, 0x7f},
                                                     {0x00, 0x00, 0xc0, 0xff},
                                                     {0x00, 0x00, 0x80, 0x7f},
                                                     {0x00, 0x00, 0x80, 0xff}});
  lists_column_wrapper<uint8_t> const fp32_expected_slice1({{0x00, 0x00, 0x00, 0x00},
                                                            {0x42, 0xc8, 0x00, 0x00},
                                                            {0xc2, 0xc8, 0x00, 0x00},
                                                            {0x00, 0x80, 0x00, 0x00},
                                                            {0x7f, 0x7f, 0xff, 0xff}});
  lists_column_wrapper<uint8_t> const fp32_expected_slice2({{0x7f, 0xc0, 0x00, 0x00},
                                                            {0xff, 0xc0, 0x00, 0x00},
                                                            {0x7f, 0x80, 0x00, 0x00},
                                                            {0xff, 0x80, 0x00, 0x00}});

  std::vector<cudf::size_type> splits({5});
  std::vector<cudf::column_view> split_column = cudf::split(fp32_col, splits);

  auto const output_fp32        = cudf::byte_cast(fp32_col, cudf::flip_endianness::NO);
  auto const output_fp32_slice1 = cudf::byte_cast(split_column.at(0), cudf::flip_endianness::YES);
  auto const output_fp32_slice2 = cudf::byte_cast(split_column.at(1), cudf::flip_endianness::YES);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_fp32->view(), fp32_expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_fp32_slice1->view(), fp32_expected_slice1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_fp32_slice2->view(), fp32_expected_slice2);
}

TEST_F(ByteCastTest, fp32ValuesWithNulls)
{
  using limits       = std::numeric_limits<float>;
  auto even_validity = make_counting_transform_iterator(0, [](auto i) { return (i + 1) % 2; });

  fixed_width_column_wrapper<float> const fp32_col(
    {float(0.0), float(100.0), float(-100.0), limits::min(), limits::max()}, {1, 0, 1, 0, 1});
  /* Data commented out below explained by comment in int16ValuesWithNulls test */
  // auto fp32_data =
  //   fixed_width_column_wrapper<uint8_t>{0x00, 0x00, 0x00, 0x00, 0xcc, 0xdd, 0xee, 0xff, 0xc2,
  //   0xc8,
  //                                       0x00, 0x00, 0xcc, 0xdd, 0xee, 0xff, 0x7f, 0x7f, 0xff,
  //                                       0xff};
  auto fp32_data =
    fixed_width_column_wrapper<uint8_t>{0x00, 0x00, 0x00, 0x00, 0x42, 0xc8, 0x00, 0x00, 0xc2, 0xc8,
                                        0x00, 0x00, 0x00, 0x80, 0x00, 0x00, 0x7f, 0x7f, 0xff, 0xff};
  auto fp32_expected = cudf::make_lists_column(
    5,
    std::move(fixed_width_column_wrapper<cudf::size_type>{0, 4, 8, 12, 16, 20}.release()),
    std::move(fp32_data.release()),
    2,
    detail::make_null_mask(even_validity, even_validity + 5));

  auto const output_fp32 = cudf::byte_cast(fp32_col, cudf::flip_endianness::YES);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(output_fp32->view(), fp32_expected->view());
}

TEST_F(ByteCastTest, fp64ValuesWithSplit)
{
  using limits = std::numeric_limits<double>;
  double nan   = limits::quiet_NaN();
  double inf   = limits::infinity();
  fixed_width_column_wrapper<double> const fp64_col({double(0.0),
                                                     double(100.0),
                                                     double(-100.0),
                                                     limits::min(),
                                                     limits::max(),
                                                     nan,
                                                     -nan,
                                                     inf,
                                                     -inf});
  lists_column_wrapper<uint8_t> const fp64_flipped_expected(
    {{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
     {0x40, 0x59, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
     {0xc0, 0x59, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
     {0x00, 0x10, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
     {0x7f, 0xef, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff},
     {0x7f, 0xf8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
     {0xff, 0xf8, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
     {0x7f, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
     {0xff, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00}});
  lists_column_wrapper<uint8_t> const fp64_expected_slice1(
    {{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
     {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x59, 0x40},
     {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x59, 0xc0},
     {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00},
     {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xef, 0x7f}});
  lists_column_wrapper<uint8_t> const fp64_expected_slice2(
    {{0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0x7f},
     {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf8, 0xff},
     {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x7f},
     {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0xff}});

  std::vector<cudf::size_type> splits({5});
  std::vector<cudf::column_view> split_column = cudf::split(fp64_col, splits);

  auto const output_fp64_flipped = cudf::byte_cast(fp64_col, cudf::flip_endianness::YES);
  auto const output_fp64_slice1  = cudf::byte_cast(split_column.at(0), cudf::flip_endianness::NO);
  auto const output_fp64_slice2  = cudf::byte_cast(split_column.at(1), cudf::flip_endianness::NO);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_fp64_flipped->view(), fp64_flipped_expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_fp64_slice1->view(), fp64_expected_slice1);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_fp64_slice2->view(), fp64_expected_slice2);
}

TEST_F(ByteCastTest, fp64ValuesWithNulls)
{
  using limits      = std::numeric_limits<double>;
  auto odd_validity = make_counting_transform_iterator(0, [](auto i) { return i % 2; });

  fixed_width_column_wrapper<double> const fp64_col(
    {double(0.0), double(100.0), double(-100.0), limits::min(), limits::max()}, {0, 1, 0, 1, 0});
  /* Data commented out below explained by comment in int16ValuesWithNulls test */
  // auto fp64_data = fixed_width_column_wrapper<uint8_t>{
  //   0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x40, 0x59, 0x00, 0x00, 0x00, 0x00,
  //   0x00, 0x00, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff, 0x00, 0x10, 0x00, 0x00,
  //   0x00, 0x00, 0x00, 0x00, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff};
  auto fp64_data = fixed_width_column_wrapper<uint8_t>{
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x59, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0xc0, 0x59, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x7f, 0xef, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
  auto fp64_expected = cudf::make_lists_column(
    5,
    std::move(fixed_width_column_wrapper<cudf::size_type>{0, 8, 16, 24, 32, 40}.release()),
    std::move(fp64_data.release()),
    3,
    detail::make_null_mask(odd_validity, odd_validity + 5));

  auto const output_fp64 = cudf::byte_cast(fp64_col, cudf::flip_endianness::YES);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(output_fp64->view(), fp64_expected->view());
}

TEST_F(ByteCastTest, StringValues)
{
  strings_column_wrapper const strings_col(
    {"", "The quick", " brown fox...", "!\"#$%&\'()*+,-./", "0123456789:;<=>?@", "[\\]^_`{|}~"});
  lists_column_wrapper<int8_t> const strings_expected(
    {{},
     {0x54, 0x68, 0x65, 0x20, 0x71, 0x75, 0x69, 0x63, 0x6b},
     {0x20, 0x62, 0x72, 0x6f, 0x77, 0x6e, 0x20, 0x66, 0x6f, 0x78, 0x2e, 0x2e, 0x2e},
     {0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f},
     {0x30,
      0x31,
      0x32,
      0x33,
      0x34,
      0x35,
      0x36,
      0x37,
      0x38,
      0x39,
      0x3a,
      0x3b,
      0x3c,
      0x3d,
      0x3e,
      0x3f,
      0x40},
     {0x5b, 0x5c, 0x5d, 0x5e, 0x5f, 0x60, 0x7b, 0x7c, 0x7d, 0x7e}});

  auto const output_strings = cudf::byte_cast(strings_col, cudf::flip_endianness::YES);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output_strings->view(), strings_expected);
}
