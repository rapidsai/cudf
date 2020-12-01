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

#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/dictionary/update_keys.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <vector>

struct DictionaryRemoveKeysTest : public cudf::test::BaseFixture {
};

TEST_F(DictionaryRemoveKeysTest, StringsColumn)
{
  cudf::test::strings_column_wrapper strings{
    "eee", "aaa", "ddd", "bbb", "ccc", "ccc", "ccc", "eee", "aaa"};
  cudf::test::strings_column_wrapper del_keys{"ddd", "bbb", "fff"};

  auto const dictionary = cudf::dictionary::encode(strings);
  // remove keys
  {
    auto const result =
      cudf::dictionary::remove_keys(cudf::dictionary_column_view(dictionary->view()), del_keys);
    std::vector<const char*> h_expected{
      "eee", "aaa", nullptr, nullptr, "ccc", "ccc", "ccc", "eee", "aaa"};
    cudf::test::strings_column_wrapper expected(
      h_expected.begin(),
      h_expected.end(),
      thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
    auto const decoded = cudf::dictionary::decode(result->view());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(decoded->view(), expected);
  }
  // remove_unused_keys
  {
    cudf::test::fixed_width_column_wrapper<int32_t> gather_map{0, 4, 3, 1};
    auto const table_result =
      cudf::gather(cudf::table_view{{dictionary->view()}}, gather_map)->release();
    auto const result  = cudf::dictionary::remove_unused_keys(table_result.front()->view());
    auto const decoded = cudf::dictionary::decode(result->view());
    cudf::test::strings_column_wrapper expected{"eee", "ccc", "bbb", "aaa"};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(decoded->view(), expected);
  }
}

TEST_F(DictionaryRemoveKeysTest, FloatColumn)
{
  cudf::test::fixed_width_column_wrapper<float> input{4.25, 7.125, 0.5, -11.75, 7.125, 0.5};
  cudf::test::fixed_width_column_wrapper<float> del_keys{4.25, -11.75, 5.0};

  auto const dictionary = cudf::dictionary::encode(input);

  {
    auto const result =
      cudf::dictionary::remove_keys(cudf::dictionary_column_view(dictionary->view()), del_keys);
    auto const decoded = cudf::dictionary::decode(result->view());
    cudf::test::fixed_width_column_wrapper<float> expected{{0., 7.125, 0.5, 0., 7.125, 0.5},
                                                           {0, 1, 1, 0, 1, 1}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(decoded->view(), expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<int32_t> gather_map{0, 2, 3, 1};
    auto const table_result =
      cudf::gather(cudf::table_view{{dictionary->view()}}, gather_map)->release();
    auto const result  = cudf::dictionary::remove_unused_keys(table_result.front()->view());
    auto const decoded = cudf::dictionary::decode(result->view());
    cudf::test::fixed_width_column_wrapper<float> expected{{4.25, 0.5, -11.75, 7.125}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(decoded->view(), expected);
  }
}

TEST_F(DictionaryRemoveKeysTest, WithNull)
{
  cudf::test::fixed_width_column_wrapper<int64_t> input{{444, 0, 333, 111, 222, 222, 222, 444, 0},
                                                        {1, 1, 1, 1, 1, 0, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int64_t> del_keys{0, 111, 777};

  auto const dictionary = cudf::dictionary::encode(input);
  {
    auto const result =
      cudf::dictionary::remove_keys(cudf::dictionary_column_view(dictionary->view()), del_keys);
    auto const decoded = cudf::dictionary::decode(result->view());
    cudf::test::fixed_width_column_wrapper<int64_t> expected{{444, 0, 333, 0, 222, 0, 222, 444, 0},
                                                             {1, 0, 1, 0, 1, 0, 1, 1, 0}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(decoded->view(), expected);
  }
  {
    cudf::test::fixed_width_column_wrapper<int32_t> gather_map{0, 2, 3, 1};
    auto const table_result =
      cudf::gather(cudf::table_view{{dictionary->view()}}, gather_map)->release();
    auto const result  = cudf::dictionary::remove_unused_keys(table_result.front()->view());
    auto const decoded = cudf::dictionary::decode(result->view());
    cudf::test::fixed_width_column_wrapper<int64_t> expected{{444, 333, 111, 0}};
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(decoded->view(), expected);
  }
}

TEST_F(DictionaryRemoveKeysTest, Errors)
{
  cudf::test::fixed_width_column_wrapper<int64_t> input{1, 2, 3};
  auto const dictionary = cudf::dictionary::encode(input);

  cudf::test::fixed_width_column_wrapper<float> del_keys{1.0, 2.0, 3.0};
  EXPECT_THROW(cudf::dictionary::remove_keys(dictionary->view(), del_keys), cudf::logic_error);
  cudf::test::fixed_width_column_wrapper<int64_t> null_keys{{1, 2, 3}, {1, 0, 1}};
  EXPECT_THROW(cudf::dictionary::remove_keys(dictionary->view(), null_keys), cudf::logic_error);
}
