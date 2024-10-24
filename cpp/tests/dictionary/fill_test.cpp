/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <cudf/dictionary/encode.hpp>
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>

struct DictionaryFillTest : public cudf::test::BaseFixture {};

TEST_F(DictionaryFillTest, StringsColumn)
{
  cudf::test::strings_column_wrapper strings(
    {"fff", "aaa", "", "bbb", "ccc", "ccc", "ccc", "fff", "aaa", ""});
  auto dictionary = cudf::dictionary::encode(strings);
  cudf::string_scalar fv("___");
  auto results = cudf::fill(dictionary->view(), 1, 4, fv);
  auto decoded = cudf::dictionary::decode(results->view());
  cudf::test::strings_column_wrapper expected(
    {"fff", "___", "___", "___", "ccc", "ccc", "ccc", "fff", "aaa", ""});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(decoded->view(), expected);
}

TEST_F(DictionaryFillTest, WithNulls)
{
  cudf::test::fixed_width_column_wrapper<int64_t> input({9, 8, 7, 6, 4},
                                                        {false, true, true, false, true});
  auto dictionary = cudf::dictionary::encode(input);
  cudf::numeric_scalar<int64_t> fv(-10);
  auto results = cudf::fill(dictionary->view(), 0, 2, fv);
  auto decoded = cudf::dictionary::decode(results->view());
  cudf::test::fixed_width_column_wrapper<int64_t> expected({-10, -10, 7, 6, 4},
                                                           {true, true, true, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(decoded->view(), expected);
}

TEST_F(DictionaryFillTest, FillWithNull)
{
  cudf::test::fixed_width_column_wrapper<double> input({1.2, 8.5, 7.75, 6.25, 4.125},
                                                       {true, true, true, false, true});
  auto dictionary = cudf::dictionary::encode(input);
  cudf::numeric_scalar<double> fv(0, false);
  auto results = cudf::fill(dictionary->view(), 1, 3, fv);
  auto decoded = cudf::dictionary::decode(results->view());
  cudf::test::fixed_width_column_wrapper<double> expected({1.2, 0.0, 0.0, 0.0, 4.125},
                                                          {true, false, false, false, true});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(decoded->view(), expected);
}

TEST_F(DictionaryFillTest, Empty)
{
  auto dictionary = cudf::make_empty_column(cudf::data_type{cudf::type_id::DICTIONARY32});
  cudf::numeric_scalar<int64_t> fv(-10);
  auto results = cudf::fill(dictionary->view(), 0, 0, fv);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results->view(), dictionary->view());
}

TEST_F(DictionaryFillTest, Errors)
{
  cudf::test::strings_column_wrapper input{"this string intentionally left blank"};
  auto dictionary = cudf::dictionary::encode(input);
  cudf::numeric_scalar<int64_t> fv(-10);  // mismatched key
  EXPECT_THROW(cudf::fill(dictionary->view(), 1, 2, fv), cudf::logic_error);
}
