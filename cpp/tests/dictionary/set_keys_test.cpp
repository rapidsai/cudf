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
#include <cudf/dictionary/update_keys.hpp>
#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

#include <vector>

struct DictionarySetKeysTest : public cudf::test::BaseFixture {
};

TEST_F(DictionarySetKeysTest, StringsKeys)
{
  cudf::test::strings_column_wrapper strings{
    "eee", "aaa", "ddd", "bbb", "ccc", "ccc", "ccc", "eee", "aaa"};
  auto dictionary = cudf::dictionary::encode(strings);

  cudf::test::strings_column_wrapper new_keys{"aaa", "ccc", "eee", "fff"};
  auto result = cudf::dictionary::set_keys(dictionary->view(), new_keys);

  std::vector<const char*> h_expected{
    "eee", "aaa", nullptr, nullptr, "ccc", "ccc", "ccc", "eee", "aaa"};
  cudf::test::strings_column_wrapper expected(
    h_expected.begin(),
    h_expected.end(),
    thrust::make_transform_iterator(h_expected.begin(), [](auto str) { return str != nullptr; }));
  auto decoded = cudf::dictionary::decode(result->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*decoded, expected);
}

TEST_F(DictionarySetKeysTest, FloatKeys)
{
  cudf::test::fixed_width_column_wrapper<float> input{4.25, 7.125, 0.5, -11.75, 7.125, 0.5};
  auto dictionary = cudf::dictionary::encode(input);

  cudf::test::fixed_width_column_wrapper<float> new_keys{0.5, 1.0, 4.25, 7.125};
  auto result = cudf::dictionary::set_keys(dictionary->view(), new_keys);

  cudf::test::fixed_width_column_wrapper<float> expected{{4.25, 7.125, 0.5, 0., 7.125, 0.5},
                                                         {1, 1, 1, 0, 1, 1}};
  auto decoded = cudf::dictionary::decode(result->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*decoded, expected);
}

TEST_F(DictionarySetKeysTest, WithNulls)
{
  cudf::test::fixed_width_column_wrapper<int64_t> input{{444, 0, 333, 111, 222, 222, 222, 444, 0},
                                                        {1, 1, 1, 1, 1, 0, 1, 1, 1}};
  auto dictionary = cudf::dictionary::encode(input);

  cudf::test::fixed_width_column_wrapper<int64_t> new_keys{0, 222, 333, 444};
  auto result = cudf::dictionary::set_keys(dictionary->view(), new_keys);

  cudf::test::fixed_width_column_wrapper<int64_t> expected{
    {444, 0, 333, 111, 222, 222, 222, 444, 0}, {1, 1, 1, 0, 1, 0, 1, 1, 1}};
  auto decoded = cudf::dictionary::decode(result->view());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*decoded, expected);
}

TEST_F(DictionarySetKeysTest, Errors)
{
  cudf::test::fixed_width_column_wrapper<int64_t> input{1, 2, 3};
  auto dictionary = cudf::dictionary::encode(input);

  cudf::test::fixed_width_column_wrapper<float> new_keys{1.0, 2.0, 3.0};
  EXPECT_THROW(cudf::dictionary::set_keys(dictionary->view(), new_keys), cudf::logic_error);
  cudf::test::fixed_width_column_wrapper<int64_t> null_keys{{1, 2, 3}, {1, 0, 1}};
  EXPECT_THROW(cudf::dictionary::set_keys(dictionary->view(), null_keys), cudf::logic_error);
}
