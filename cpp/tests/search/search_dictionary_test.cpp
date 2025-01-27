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

#include <cudf/search.hpp>

struct DictionarySearchTest : public cudf::test::BaseFixture {};

using cudf::numeric_scalar;
using cudf::size_type;
using cudf::string_scalar;
using cudf::test::fixed_width_column_wrapper;

TEST_F(DictionarySearchTest, search_dictionary)
{
  cudf::test::dictionary_column_wrapper<std::string> input(
    {"", "", "10", "10", "20", "20", "30", "40"},
    {false, false, true, true, true, true, true, true});
  cudf::test::dictionary_column_wrapper<std::string> values(
    {"", "08", "10", "11", "30", "32", "90"}, {false, true, true, true, true, true, true});

  auto result = cudf::upper_bound({cudf::table_view{{input}}},
                                  {cudf::table_view{{values}}},
                                  {cudf::order::ASCENDING},
                                  {cudf::null_order::BEFORE});
  fixed_width_column_wrapper<size_type> expect_upper{2, 2, 4, 4, 7, 7, 8};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect_upper);

  result = cudf::lower_bound({cudf::table_view{{input}}},
                             {cudf::table_view{{values}}},
                             {cudf::order::ASCENDING},
                             {cudf::null_order::BEFORE});
  fixed_width_column_wrapper<size_type> expect_lower{0, 2, 2, 4, 6, 7, 8};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect_lower);
}

TEST_F(DictionarySearchTest, search_table_dictionary)
{
  fixed_width_column_wrapper<int32_t> column_0{
    {10, 10, 20, 20, 20, 20, 20, 20, 20, 50, 30},
    {true, true, true, true, true, true, true, true, true, true, false}};
  fixed_width_column_wrapper<float> column_1{
    {5.0, 6.0, .5, .5, .5, .5, .7, .7, .7, .7, .5},
    {true, false, true, true, true, true, true, true, true, true, true}};
  cudf::test::dictionary_column_wrapper<int16_t> column_2{
    {90, 95, 77, 78, 79, 76, 61, 62, 63, 41, 50},
    {true, true, true, true, false, false, true, true, true, true, true}};
  cudf::table_view input({column_0, column_1, column_2});

  fixed_width_column_wrapper<int32_t> values_0{{10, 40, 20}, {true, false, true}};
  fixed_width_column_wrapper<float> values_1{{6., .5, .5}, {false, true, true}};
  cudf::test::dictionary_column_wrapper<int16_t> values_2{{95, 50, 77}, {true, true, false}};
  cudf::table_view values({values_0, values_1, values_2});

  std::vector<cudf::order> order_flags{
    {cudf::order::ASCENDING, cudf::order::ASCENDING, cudf::order::DESCENDING}};
  std::vector<cudf::null_order> null_order_flags{
    {cudf::null_order::AFTER, cudf::null_order::AFTER, cudf::null_order::AFTER}};

  auto result = cudf::lower_bound(input, values, order_flags, null_order_flags);
  fixed_width_column_wrapper<size_type> expect_lower{1, 10, 2};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect_lower);

  result = cudf::upper_bound(input, values, order_flags, null_order_flags);
  fixed_width_column_wrapper<size_type> expect_upper{2, 11, 6};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect_upper);
}

TEST_F(DictionarySearchTest, contains_dictionary)
{
  cudf::test::dictionary_column_wrapper<std::string> column(
    {"00", "00", "17", "17", "23", "23", "29"});
  EXPECT_TRUE(cudf::contains(column, string_scalar{"23"}));
  EXPECT_FALSE(cudf::contains(column, string_scalar{"28"}));

  cudf::test::dictionary_column_wrapper<std::string> needles({"00", "17", "23", "27"});
  fixed_width_column_wrapper<bool> expect{1, 1, 1, 0};
  auto result = cudf::contains(column, needles);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}

TEST_F(DictionarySearchTest, contains_nullable_dictionary)
{
  cudf::test::dictionary_column_wrapper<int64_t> column(
    {0, 0, 17, 17, 23, 23, 29}, {true, false, true, true, true, true, true});
  EXPECT_TRUE(cudf::contains(column, numeric_scalar<int64_t>{23}));
  EXPECT_FALSE(cudf::contains(column, numeric_scalar<int64_t>{28}));

  cudf::test::dictionary_column_wrapper<int64_t> needles({0, 17, 23, 27});
  fixed_width_column_wrapper<bool> expect{1, 1, 1, 0};
  auto result = cudf::contains(column, needles);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*result, expect);
}
