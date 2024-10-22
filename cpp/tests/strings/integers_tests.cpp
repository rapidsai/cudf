/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <thrust/host_vector.h>
#include <thrust/iterator/transform_iterator.h>

#include <array>
#include <string>
#include <vector>

// Using an alias variable for the null elements
// This will make the code looks cleaner
constexpr auto NULL_VAL = 0;

struct StringsConvertTest : public cudf::test::BaseFixture {};

TEST_F(StringsConvertTest, IsIntegerBasicCheck)
{
  cudf::test::strings_column_wrapper strings1(
    {"+175", "-34", "9.8", "17+2", "+-14", "1234567890", "67de", "", "1e10", "-", "++", ""});
  auto results = cudf::strings::is_integer(cudf::strings_column_view(strings1));
  cudf::test::fixed_width_column_wrapper<bool> expected1({1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected1);

  cudf::test::strings_column_wrapper strings2(
    {"0", "+0", "-0", "1234567890", "-27341132", "+012", "023", "-045"});
  results = cudf::strings::is_integer(cudf::strings_column_view(strings2));
  cudf::test::fixed_width_column_wrapper<bool> expected2({1, 1, 1, 1, 1, 1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected2);
}

TEST_F(StringsConvertTest, ZeroSizeIsIntegerBasicCheck)
{
  cudf::test::strings_column_wrapper strings;
  auto strings_view = cudf::strings_column_view(strings);
  auto results      = cudf::strings::is_integer(strings_view);
  EXPECT_EQ(cudf::type_id::BOOL8, results->view().type().id());
  EXPECT_EQ(0, results->view().size());
}

TEST_F(StringsConvertTest, IsIntegerBoundCheckNoNull)
{
  auto strings = cudf::test::strings_column_wrapper(
    {"+175", "-34", "9.8", "17+2", "+-14", "1234567890", "67de", "", "1e10", "-", "++", ""});
  auto results = cudf::strings::is_integer(cudf::strings_column_view(strings),
                                           cudf::data_type{cudf::type_id::INT32});
  auto expected =
    cudf::test::fixed_width_column_wrapper<bool>({1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  strings = cudf::test::strings_column_wrapper(
    {"0", "+0", "-0", "1234567890", "-27341132", "+012", "023", "-045"});
  results  = cudf::strings::is_integer(cudf::strings_column_view(strings),
                                      cudf::data_type{cudf::type_id::INT32});
  expected = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 1, 1, 1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsConvertTest, IsIntegerBoundCheckWithNulls)
{
  std::vector<char const*> const h_strings{
    "eee", "1234", nullptr, "", "-9832", "93.24", "765é", nullptr};
  auto const strings = cudf::test::strings_column_wrapper(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  auto const results = cudf::strings::is_integer(cudf::strings_column_view(strings),
                                                 cudf::data_type{cudf::type_id::INT32});
  // Input has null elements then the output should have the same null mask
  auto const expected = cudf::test::fixed_width_column_wrapper<bool>(
    std::initializer_list<int8_t>{0, 1, NULL_VAL, 0, 1, 0, 0, NULL_VAL},
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsConvertTest, ZeroSizeIsIntegerBoundCheck)
{
  // Empty input
  auto strings = cudf::test::strings_column_wrapper{};
  auto results = cudf::strings::is_integer(cudf::strings_column_view(strings),
                                           cudf::data_type{cudf::type_id::INT32});
  EXPECT_EQ(cudf::type_id::BOOL8, results->view().type().id());
  EXPECT_EQ(0, results->view().size());
}

TEST_F(StringsConvertTest, IsIntegerBoundCheckSmallNumbers)
{
  auto strings = cudf::test::strings_column_wrapper(
    {"-200", "-129", "-128", "-120", "0", "120", "127", "130", "150", "255", "300", "500"});
  auto results = cudf::strings::is_integer(cudf::strings_column_view(strings),
                                           cudf::data_type{cudf::type_id::INT8});
  auto expected =
    cudf::test::fixed_width_column_wrapper<bool>({0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results  = cudf::strings::is_integer(cudf::strings_column_view(strings),
                                      cudf::data_type{cudf::type_id::UINT8});
  expected = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  strings = cudf::test::strings_column_wrapper(
    {"-40000", "-32769", "-32768", "-32767", "-32766", "32765", "32766", "32767", "32768"});
  results  = cudf::strings::is_integer(cudf::strings_column_view(strings),
                                      cudf::data_type{cudf::type_id::INT16});
  expected = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 1, 1, 1, 1, 1, 1, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results  = cudf::strings::is_integer(cudf::strings_column_view(strings),
                                      cudf::data_type{cudf::type_id::UINT16});
  expected = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 0, 0, 0, 1, 1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results  = cudf::strings::is_integer(cudf::strings_column_view(strings),
                                      cudf::data_type{cudf::type_id::INT32});
  expected = cudf::test::fixed_width_column_wrapper<bool>({1, 1, 1, 1, 1, 1, 1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsConvertTest, IsIntegerBoundCheckLargeNumbers)
{
  auto strings =
    cudf::test::strings_column_wrapper({"-2147483649",   // std::numeric_limits<int32_t>::min() - 1
                                        "-2147483648",   // std::numeric_limits<int32_t>::min()
                                        "-2147483647",   // std::numeric_limits<int32_t>::min() + 1
                                        "2147483646",    // std::numeric_limits<int32_t>::max() - 1
                                        "2147483647",    // std::numeric_limits<int32_t>::max()
                                        "2147483648",    // std::numeric_limits<int32_t>::max() + 1
                                        "4294967294",    // std::numeric_limits<uint32_t>::max() - 1
                                        "4294967295",    // std::numeric_limits<uint32_t>::max()
                                        "4294967296"});  // std::numeric_limits<uint32_t>::max() + 1
  auto results  = cudf::strings::is_integer(cudf::strings_column_view(strings),
                                           cudf::data_type{cudf::type_id::INT32});
  auto expected = cudf::test::fixed_width_column_wrapper<bool>({0, 1, 1, 1, 1, 0, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results  = cudf::strings::is_integer(cudf::strings_column_view(strings),
                                      cudf::data_type{cudf::type_id::UINT32});
  expected = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 0, 1, 1, 1, 1, 1, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  strings = cudf::test::strings_column_wrapper(
    {"-9223372036854775809",    // std::numeric_limits<int64_t>::min() - 1
     "-9223372036854775808",    // std::numeric_limits<int64_t>::min()
     "-9223372036854775807",    // std::numeric_limits<int64_t>::min() + 1
     "9223372036854775806",     // std::numeric_limits<int64_t>::max() - 1
     "9223372036854775807",     // std::numeric_limits<int64_t>::max()
     "9223372036854775808",     // std::numeric_limits<int64_t>::max() + 1
     "18446744073709551614",    // std::numeric_limits<uint64_t>::max() - 1
     "18446744073709551615",    // std::numeric_limits<uint64_t>::max()
     "18446744073709551616"});  // std::numeric_limits<uint64_t>::max() + 1
  results  = cudf::strings::is_integer(cudf::strings_column_view(strings),
                                      cudf::data_type{cudf::type_id::INT64});
  expected = cudf::test::fixed_width_column_wrapper<bool>({0, 1, 1, 1, 1, 0, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results  = cudf::strings::is_integer(cudf::strings_column_view(strings),
                                      cudf::data_type{cudf::type_id::UINT64});
  expected = cudf::test::fixed_width_column_wrapper<bool>({0, 0, 0, 1, 1, 1, 1, 1, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsConvertTest, ToInteger)
{
  std::vector<char const*> h_strings{"eee",
                                     "1234",
                                     nullptr,
                                     "",
                                     "-9832",
                                     "93.24",
                                     "765é",
                                     nullptr,
                                     "-1.78e+5",
                                     "2147483647",
                                     "-2147483648",
                                     "2147483648"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  auto results            = cudf::strings::to_integers(cudf::strings_column_view(strings),
                                            cudf::data_type{cudf::type_id::INT16});
  auto const expected_i16 = cudf::test::fixed_width_column_wrapper<int16_t>(
    std::initializer_list<int16_t>{0, 1234, NULL_VAL, 0, -9832, 93, 765, NULL_VAL, -1, -1, 0, 0},
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_i16);

  results                 = cudf::strings::to_integers(cudf::strings_column_view(strings),
                                       cudf::data_type{cudf::type_id::INT32});
  auto const expected_i32 = cudf::test::fixed_width_column_wrapper<int32_t>(
    std::initializer_list<int32_t>{
      0, 1234, NULL_VAL, 0, -9832, 93, 765, NULL_VAL, -1, 2147483647, -2147483648, -2147483648},
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_i32);

  results                 = cudf::strings::to_integers(cudf::strings_column_view(strings),
                                       cudf::data_type{cudf::type_id::UINT32});
  auto const expected_u32 = cudf::test::fixed_width_column_wrapper<uint32_t>(
    std::initializer_list<uint32_t>{0,
                                    1234,
                                    NULL_VAL,
                                    0,
                                    4294957464,
                                    93,
                                    765,
                                    NULL_VAL,
                                    4294967295,
                                    2147483647,
                                    2147483648,
                                    2147483648},
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected_u32);
}

TEST_F(StringsConvertTest, FromInteger)
{
  int32_t minint = std::numeric_limits<int32_t>::min();
  int32_t maxint = std::numeric_limits<int32_t>::max();
  std::vector<int32_t> h_integers{100, 987654321, 0, 0, -12761, 0, 5, -4, maxint, minint};
  std::vector<char const*> h_expected{
    "100", "987654321", nullptr, "0", "-12761", "0", "5", "-4", "2147483647", "-2147483648"};

  cudf::test::fixed_width_column_wrapper<int32_t> integers(
    h_integers.begin(),
    h_integers.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));

  auto results = cudf::strings::from_integers(integers);

  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsConvertTest, ZeroSizeStringsColumn)
{
  auto const zero_size_column = cudf::make_empty_column(cudf::type_id::INT32)->view();
  auto results                = cudf::strings::from_integers(zero_size_column);
  cudf::test::expect_column_empty(results->view());
}

TEST_F(StringsConvertTest, ZeroSizeIntegersColumn)
{
  auto const zero_size_strings_column = cudf::make_empty_column(cudf::type_id::STRING)->view();
  auto results =
    cudf::strings::to_integers(zero_size_strings_column, cudf::data_type{cudf::type_id::INT32});
  EXPECT_EQ(0, results->size());
}

TEST_F(StringsConvertTest, EmptyStringsColumn)
{
  cudf::test::strings_column_wrapper strings({"", "", ""});
  auto results = cudf::strings::to_integers(cudf::strings_column_view(strings),
                                            cudf::data_type{cudf::type_id::INT64});
  cudf::test::fixed_width_column_wrapper<int64_t> expected{0, 0, 0};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), expected);
}

template <typename T>
class StringsIntegerConvertTest : public StringsConvertTest {};

TYPED_TEST_SUITE(StringsIntegerConvertTest, cudf::test::IntegralTypesNotBool);

TYPED_TEST(StringsIntegerConvertTest, FromToInteger)
{
  thrust::host_vector<TypeParam> h_integers(255);
  std::iota(h_integers.begin(), h_integers.end(), -(TypeParam)(h_integers.size() / 2));
  h_integers.push_back(std::numeric_limits<TypeParam>::min());
  h_integers.push_back(std::numeric_limits<TypeParam>::max());
  auto const d_integers = cudf::detail::make_device_uvector_sync(
    h_integers, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  auto integers      = cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<TypeParam>()},
                                            (cudf::size_type)d_integers.size());
  auto integers_view = integers->mutable_view();
  CUDF_CUDA_TRY(cudaMemcpy(integers_view.data<TypeParam>(),
                           d_integers.data(),
                           d_integers.size() * sizeof(TypeParam),
                           cudaMemcpyDefault));
  integers_view.set_null_count(0);

  // convert to strings
  auto results_strings = cudf::strings::from_integers(integers->view());

  std::vector<std::string> h_strings;
  for (auto itr = h_integers.begin(); itr != h_integers.end(); ++itr)
    h_strings.push_back(std::to_string(*itr));

  cudf::test::strings_column_wrapper expected(h_strings.begin(), h_strings.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results_strings, expected);

  // convert back to integers
  auto strings_view = cudf::strings_column_view(results_strings->view());
  auto results_integers =
    cudf::strings::to_integers(strings_view, cudf::data_type(cudf::type_to_id<TypeParam>()));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results_integers, integers->view());
}

//
template <typename T>
class StringsFloatConvertTest : public StringsConvertTest {};

using FloatTypes = cudf::test::Types<float, double>;
TYPED_TEST_SUITE(StringsFloatConvertTest, FloatTypes);

TYPED_TEST(StringsFloatConvertTest, FromToIntegerError)
{
  auto dtype  = cudf::data_type{cudf::type_to_id<TypeParam>()};
  auto column = cudf::make_numeric_column(dtype, 100);
  EXPECT_THROW(cudf::strings::from_integers(column->view()), cudf::logic_error);

  cudf::test::strings_column_wrapper strings{"this string intentionally left blank"};
  EXPECT_THROW(cudf::strings::to_integers(column->view(), dtype), cudf::logic_error);
}

TEST_F(StringsConvertTest, HexToInteger)
{
  std::vector<char const*> h_strings{
    "1234", nullptr, "98BEEF", "1a5", "CAFE", "2face", "0xAABBCCDD", "112233445566"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  {
    std::vector<int32_t> h_expected;
    for (auto& h_string : h_strings) {
      if (h_string == nullptr)
        h_expected.push_back(0);
      else
        h_expected.push_back(static_cast<int>(std::stol(std::string(h_string), nullptr, 16)));
    }

    auto results = cudf::strings::hex_to_integers(cudf::strings_column_view(strings),
                                                  cudf::data_type{cudf::type_id::INT32});
    cudf::test::fixed_width_column_wrapper<int32_t> expected(
      h_expected.begin(),
      h_expected.end(),
      thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
  {
    std::vector<int64_t> h_expected;
    for (auto& h_string : h_strings) {
      if (h_string == nullptr)
        h_expected.push_back(0);
      else
        h_expected.push_back(std::stol(std::string(h_string), nullptr, 16));
    }

    auto results = cudf::strings::hex_to_integers(cudf::strings_column_view(strings),
                                                  cudf::data_type{cudf::type_id::INT64});
    cudf::test::fixed_width_column_wrapper<int64_t> expected(
      h_expected.begin(),
      h_expected.end(),
      thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  }
}

TEST_F(StringsConvertTest, IsHex)
{
  std::vector<char const*> h_strings{"",
                                     "1234",
                                     nullptr,
                                     "98BEEF",
                                     "1a5",
                                     "2face",
                                     "0xAABBCCDD",
                                     "112233445566",
                                     "XYZ",
                                     "0",
                                     "0x",
                                     "x"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  cudf::test::fixed_width_column_wrapper<bool> expected(
    {0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0},
    {true, true, false, true, true, true, true, true, true, true, true, true});
  auto results = cudf::strings::is_hex(cudf::strings_column_view(strings));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TYPED_TEST(StringsIntegerConvertTest, IntegerToHex)
{
  std::vector<TypeParam> h_integers(255);
  std::generate(h_integers.begin(), h_integers.end(), []() {
    static TypeParam data = 0;
    return data++ << (sizeof(TypeParam) - 1) * 8;
  });

  cudf::test::fixed_width_column_wrapper<TypeParam> integers(h_integers.begin(), h_integers.end());

  std::vector<std::string> h_expected(255);
  std::transform(h_integers.begin(), h_integers.end(), h_expected.begin(), [](auto v) {
    if (v == 0) { return std::string("00"); }
    // special handling for single-byte types
    if constexpr (std::is_same_v<TypeParam, int8_t> || std::is_same_v<TypeParam, uint8_t>) {
      std::array const hex_digits = {
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'};
      std::string str;
      str += hex_digits[(v & 0xF0) >> 4];
      str += hex_digits[(v & 0x0F)];
      return str;
    }
    // all other types work with this
    std::stringstream str;
    str << std::setfill('0') << std::setw(sizeof(TypeParam) * 2) << std::hex << std::uppercase << v;
    return str.str();
  });

  cudf::test::strings_column_wrapper expected(h_expected.begin(), h_expected.end());

  auto results = cudf::strings::integers_to_hex(integers);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsConvertTest, IntegerToHexWithNull)
{
  cudf::test::fixed_width_column_wrapper<int32_t> integers(
    {123456, -1, 0, 0, 12, 12345, 123456789, -123456789},
    {true, true, true, false, true, true, true, true});

  cudf::test::strings_column_wrapper expected(
    {"01E240", "FFFFFFFF", "00", "", "0C", "3039", "075BCD15", "F8A432EB"},
    {true, true, true, false, true, true, true, true});

  auto results = cudf::strings::integers_to_hex(integers);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST_F(StringsConvertTest, IntegerConvertErrors)
{
  cudf::test::fixed_width_column_wrapper<bool> bools(
    {true, true, false, false, true, true, false, true});
  cudf::test::fixed_width_column_wrapper<double> floats(
    {123456.0, -1.0, 0.0, 0.0, 12.0, 12345.0, 123456789.0});
  EXPECT_THROW(cudf::strings::integers_to_hex(bools), cudf::logic_error);
  EXPECT_THROW(cudf::strings::integers_to_hex(floats), cudf::logic_error);
  EXPECT_THROW(cudf::strings::from_integers(bools), cudf::logic_error);
  EXPECT_THROW(cudf::strings::from_integers(floats), cudf::logic_error);

  auto input = cudf::test::strings_column_wrapper({"123456", "-1", "0"});
  auto view  = cudf::strings_column_view(input);
  EXPECT_THROW(cudf::strings::to_integers(view, cudf::data_type(cudf::type_id::BOOL8)),
               cudf::logic_error);
  EXPECT_THROW(cudf::strings::to_integers(view, cudf::data_type(cudf::type_id::FLOAT32)),
               cudf::logic_error);
  EXPECT_THROW(cudf::strings::to_integers(view, cudf::data_type(cudf::type_id::TIMESTAMP_SECONDS)),
               cudf::logic_error);
  EXPECT_THROW(
    cudf::strings::to_integers(view, cudf::data_type(cudf::type_id::DURATION_MILLISECONDS)),
    cudf::logic_error);
  EXPECT_THROW(cudf::strings::to_integers(view, cudf::data_type(cudf::type_id::DECIMAL32)),
               cudf::logic_error);
}
