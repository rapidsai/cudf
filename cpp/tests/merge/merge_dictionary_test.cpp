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

#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/merge.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/cudf_gtest.hpp>

#include <vector>

struct MergeDictionaryTest : public cudf::test::BaseFixture {
};

TEST_F(MergeDictionaryTest, Merge1Column)
{
  cudf::test::strings_column_wrapper left_w({"ab", "ab", "cd", "de", "de", "fg", "gh", "gh"});
  auto left = cudf::dictionary::encode(left_w);
  cudf::test::strings_column_wrapper right_w({"ab", "cd", "de", "fg", "gh"});
  auto right = cudf::dictionary::encode(right_w);

  cudf::table_view left_view{{left->view()}};
  cudf::table_view right_view{{right->view()}};

  std::vector<cudf::size_type> key_cols{0};
  std::vector<cudf::order> column_order{cudf::order::ASCENDING};
  std::vector<cudf::null_order> null_precedence{};

  auto result = cudf::merge({left_view, right_view}, key_cols, column_order, null_precedence);

  cudf::test::strings_column_wrapper expected_w(
    {"ab", "ab", "ab", "cd", "cd", "de", "de", "de", "fg", "fg", "gh", "gh", "gh"});
  auto expected = cudf::dictionary::encode(expected_w);

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected->view(), result->get_column(0).view());
}

TEST_F(MergeDictionaryTest, Merge2Columns)
{
  cudf::test::strings_column_wrapper left_w1({"ab", "bc", "cd", "de", "de", "fg", "fg"});
  auto left1 = cudf::dictionary::encode(left_w1);
  cudf::test::strings_column_wrapper left_w2({"zy", "zy", "xw", "xw", "vu", "vu", "ts"});
  auto left2 = cudf::dictionary::encode(left_w2);
  cudf::table_view left_view{{left1->view(), left2->view()}};

  cudf::test::strings_column_wrapper right_w1({"ab", "ab", "bc", "cd", "de", "fg"});
  auto right1 = cudf::dictionary::encode(right_w1);
  cudf::test::strings_column_wrapper right_w2({"zy", "xw", "xw", "vu", "ts", "ts"});
  auto right2 = cudf::dictionary::encode(right_w2);
  cudf::table_view right_view{{right1->view(), right2->view()}};

  std::vector<cudf::size_type> key_cols{0, 1};
  std::vector<cudf::order> column_order{cudf::order::ASCENDING, cudf::order::DESCENDING};
  std::vector<cudf::null_order> null_precedence{};

  auto result   = cudf::merge({left_view, right_view}, key_cols, column_order, null_precedence);
  auto decoded1 = cudf::dictionary::decode(result->get_column(0).view());
  auto decoded2 = cudf::dictionary::decode(result->get_column(1).view());

  cudf::test::strings_column_wrapper expected_1(
    {"ab", "ab", "ab", "bc", "bc", "cd", "cd", "de", "de", "de", "fg", "fg", "fg"});
  cudf::test::strings_column_wrapper expected_2(
    {"zy", "zy", "xw", "zy", "xw", "xw", "vu", "xw", "vu", "ts", "vu", "ts", "ts"});

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_1, decoded1->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_2, decoded2->view());

  left_view  = cudf::table_view{{left1->view(), left_w2}};
  right_view = cudf::table_view{{right1->view(), right_w2}};
  result     = cudf::merge({left_view, right_view}, key_cols, column_order, null_precedence);
  decoded1   = cudf::dictionary::decode(result->get_column(0).view());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_1, decoded1->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_2, result->get_column(1).view());

  left_view  = cudf::table_view{{left_w1, left2->view()}};
  right_view = cudf::table_view{{right_w1, right2->view()}};
  result     = cudf::merge({left_view, right_view}, key_cols, column_order, null_precedence);
  decoded2   = cudf::dictionary::decode(result->get_column(1).view());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_1, result->get_column(0).view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_2, decoded2->view());
}

TEST_F(MergeDictionaryTest, WithNulls)
{
  cudf::test::fixed_width_column_wrapper<int8_t> left_w1({1, 2, 2, 4, 4, 5, 0},
                                                         {1, 1, 1, 1, 1, 1, 0});
  auto left1 = cudf::dictionary::encode(left_w1);
  cudf::test::fixed_width_column_wrapper<int64_t> left_w2({1000, 1000, 800, 500, 500, 100, 0},
                                                          {1, 1, 1, 1, 1, 1, 0});
  auto left2 = cudf::dictionary::encode(left_w2);
  cudf::table_view left_view{{left1->view(), left2->view()}};

  cudf::test::fixed_width_column_wrapper<int8_t> right_w1({1, 1, 2, 4, 5, 0}, {1, 1, 1, 1, 1, 0});
  auto right1 = cudf::dictionary::encode(right_w1);
  cudf::test::fixed_width_column_wrapper<int64_t> right_w2({1000, 800, 800, 400, 100, 0},
                                                           {1, 1, 1, 1, 1, 0});
  auto right2 = cudf::dictionary::encode(right_w2);
  cudf::table_view right_view{{right1->view(), right2->view()}};

  std::vector<cudf::size_type> key_cols{0, 1};
  std::vector<cudf::order> column_order{cudf::order::ASCENDING, cudf::order::DESCENDING};
  std::vector<cudf::null_order> null_precedence{cudf::null_order::AFTER, cudf::null_order::BEFORE};

  auto result   = cudf::merge({left_view, right_view}, key_cols, column_order, null_precedence);
  auto decoded1 = cudf::dictionary::decode(result->get_column(0).view());
  auto decoded2 = cudf::dictionary::decode(result->get_column(1).view());

  cudf::test::fixed_width_column_wrapper<int8_t> expected_1(
    {1, 1, 1, 2, 2, 2, 4, 4, 4, 5, 5, 0, 0}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0});
  cudf::test::fixed_width_column_wrapper<int64_t> expected_2(
    {1000, 1000, 800, 1000, 800, 800, 500, 500, 400, 100, 100, 0, 0},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_1, decoded1->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_2, decoded2->view());

  left_view  = cudf::table_view{{left1->view(), left_w2}};
  right_view = cudf::table_view{{right1->view(), right_w2}};
  result     = cudf::merge({left_view, right_view}, key_cols, column_order, null_precedence);
  decoded1   = cudf::dictionary::decode(result->get_column(0).view());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_1, decoded1->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_2, result->get_column(1).view());

  left_view  = cudf::table_view{{left_w1, left2->view()}};
  right_view = cudf::table_view{{right_w1, right2->view()}};
  result     = cudf::merge({left_view, right_view}, key_cols, column_order, null_precedence);
  decoded2   = cudf::dictionary::decode(result->get_column(1).view());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_1, result->get_column(0).view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected_2, decoded2->view());
}
