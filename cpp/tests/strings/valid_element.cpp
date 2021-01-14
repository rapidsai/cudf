/*
 * Copyright (c) 2021, Baidu CORPORATION.
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

#include <cudf/strings/convert/is_valid_element.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

struct ValidStringCharsTest : public cudf::test::BaseFixture {
};

TEST_F(ValidStringCharsTest, ValidFixedPoint)
{
  // allow_decimal = true
  cudf::test::strings_column_wrapper strings1(
    {"+175", "-34", "9.8", "17+2", "+-14", "1234567890", "67de", "", "1e10", "-", "++", "", "21474836482222"});
  auto results = cudf::strings::is_valid_element(cudf::strings_column_view(strings1), true, cudf::data_type{cudf::type_id::INT8});
  cudf::test::fixed_width_column_wrapper<bool> expected1({0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected1);

  cudf::test::strings_column_wrapper strings2(
    {"+175", "-34", "9.8", "17+2", "+-14", "1234567890", "67de", "", "1e10", "-", "++", "", "21474836482222"});
  results = cudf::strings::is_valid_element(cudf::strings_column_view(strings2), true, cudf::data_type{cudf::type_id::INT16});
  cudf::test::fixed_width_column_wrapper<bool> expected2({1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected2);

  cudf::test::strings_column_wrapper strings3(
    {"+175", "-34", "9.8", "17+2", "+-14", "1234567890", "67de", "", "1e10", "-", "++", "", "21474836482222"});
  results = cudf::strings::is_valid_element(cudf::strings_column_view(strings3), true, cudf::data_type{cudf::type_id::INT32});
  cudf::test::fixed_width_column_wrapper<bool> expected3({1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected3);

  cudf::test::strings_column_wrapper strings4(
    {"+175", "-34", "9.8", "17+2", "+-14", "1234567890", "67de", "", "1e10", "-", "++", "", "21474836482222"});
  results = cudf::strings::is_valid_element(cudf::strings_column_view(strings4), true, cudf::data_type{cudf::type_id::INT64});
  cudf::test::fixed_width_column_wrapper<bool> expected4({1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected4);
  
  // allow_decimal = false
  cudf::test::strings_column_wrapper strings5(
    {"+175", "-34", "9.8", "17+2", "+-14", "1234567890", "67de", "", "1e10", "-", "++", "", "21474836482222"});
  results = cudf::strings::is_valid_element(cudf::strings_column_view(strings5), false, cudf::data_type{cudf::type_id::INT8});
  cudf::test::fixed_width_column_wrapper<bool> expected5({0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected5);

  cudf::test::strings_column_wrapper strings6(
    {"+175", "-34", "9.8", "17+2", "+-14", "1234567890", "67de", "", "1e10", "-", "++", "", "21474836482222"});
  results = cudf::strings::is_valid_element(cudf::strings_column_view(strings6), false, cudf::data_type{cudf::type_id::INT16});
  cudf::test::fixed_width_column_wrapper<bool> expected6({1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected6);

  cudf::test::strings_column_wrapper strings7(
    {"+175", "-34", "9.8", "17+2", "+-14", "1234567890", "67de", "", "1e10", "-", "++", "", "21474836482222"});
  results = cudf::strings::is_valid_element(cudf::strings_column_view(strings7), false, cudf::data_type{cudf::type_id::INT32});
  cudf::test::fixed_width_column_wrapper<bool> expected7({1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected7);

  cudf::test::strings_column_wrapper strings8(
    {"+175", "-34", "9.8", "17+2", "+-14", "1234567890", "67de", "", "1e10", "-", "++", "", "21474836482222"});
  results = cudf::strings::is_valid_element(cudf::strings_column_view(strings8), false, cudf::data_type{cudf::type_id::INT64});
  cudf::test::fixed_width_column_wrapper<bool> expected8({1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected8);

  // second test
  cudf::test::strings_column_wrapper strings0(
    {"0", "+0", "-0", "1234567890", "-27341132", "+012", "023", "-045", "-1.1", "+1000.1"});
  results = cudf::strings::is_valid_element(cudf::strings_column_view(strings0), true, cudf::data_type{cudf::type_id::INT64});
  cudf::test::fixed_width_column_wrapper<bool> expected0({1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected0);

}

