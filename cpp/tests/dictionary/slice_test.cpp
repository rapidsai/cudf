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

#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/dictionary/update_keys.hpp>

#include <vector>

struct DictionarySliceTest : public cudf::test::BaseFixture {};

TEST_F(DictionarySliceTest, SliceColumn)
{
  cudf::test::strings_column_wrapper strings{
    {"eee", "aaa", "ddd", "bbb", "ccc", "", "ccc", "eee", "aaa"},
    {true, true, true, true, true, false, true, true, true}};
  auto dictionary = cudf::dictionary::encode(strings);

  std::vector<cudf::size_type> splits{1, 6};
  auto result = cudf::slice(dictionary->view(), splits);

  auto output = cudf::dictionary::decode(cudf::dictionary_column_view(result.front()));
  cudf::test::strings_column_wrapper expected{{"aaa", "ddd", "bbb", "ccc", ""},
                                              {true, true, true, true, false}};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *output);

  {
    auto defragged =
      cudf::dictionary::remove_unused_keys(cudf::dictionary_column_view(result.front()));
    output = cudf::dictionary::decode(cudf::dictionary_column_view(*defragged));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *output);  // should be the same output
  }
  {
    cudf::test::strings_column_wrapper new_keys{"000", "bbb"};
    auto added = cudf::dictionary::add_keys(cudf::dictionary_column_view(result.front()), new_keys);
    output     = cudf::dictionary::decode(cudf::dictionary_column_view(*added));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *output);
  }
  {
    cudf::test::strings_column_wrapper new_keys{"aaa", "bbb", "ccc", "ddd", "000"};
    auto added = cudf::dictionary::set_keys(cudf::dictionary_column_view(result.front()), new_keys);
    output     = cudf::dictionary::decode(cudf::dictionary_column_view(*added));
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected, *output);
  }
  {
    // check new column is created correctly from sliced view (issue 5768)
    cudf::column new_col(result.front());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.front(), new_col.view());
  }
}

TEST_F(DictionarySliceTest, SplitColumn)
{
  cudf::test::fixed_width_column_wrapper<float> input{{4.25, 7.125, 0.5, 0., -11.75, 7.125, 0.5},
                                                      {true, true, true, false, true, true, true}};
  auto dictionary = cudf::dictionary::encode(input);

  std::vector<cudf::size_type> splits{2, 6};
  auto results = cudf::split(dictionary->view(), splits);

  cudf::test::fixed_width_column_wrapper<float> expected1{{4.25, 7.125}, {true, true}};
  auto output1 = cudf::dictionary::decode(cudf::dictionary_column_view(results[0]));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected1, output1->view());

  cudf::test::fixed_width_column_wrapper<float> expected2{{0.5, 0., -11.75, 7.125},
                                                          {true, false, true, true}};
  auto output2 = cudf::dictionary::decode(cudf::dictionary_column_view(results[1]));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected2, output2->view());

  cudf::test::fixed_width_column_wrapper<float> expected3({0.5}, {true});
  auto output3 = cudf::dictionary::decode(cudf::dictionary_column_view(results[2]));
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(expected3, output3->view());
}
