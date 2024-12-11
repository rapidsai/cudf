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
#include <cudf_test/testing_main.hpp>

#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/dictionary/update_keys.hpp>
#include <cudf/utilities/error.hpp>

struct DictionaryAddKeysTest : public cudf::test::BaseFixture {};

TEST_F(DictionaryAddKeysTest, StringsColumn)
{
  cudf::test::strings_column_wrapper strings(
    {"fff", "aaa", "ddd", "bbb", "ccc", "ccc", "ccc", "fff", "aaa"});
  cudf::test::strings_column_wrapper new_keys({"ddd", "bbb", "eee"});

  auto dictionary = cudf::dictionary::encode(strings);
  auto result =
    cudf::dictionary::add_keys(cudf::dictionary_column_view(dictionary->view()), new_keys);

  cudf::dictionary_column_view view(result->view());

  cudf::test::strings_column_wrapper keys_expected({"aaa", "bbb", "ccc", "ddd", "eee", "fff"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.keys(), keys_expected);

  cudf::test::fixed_width_column_wrapper<uint8_t> indices_expected({5, 0, 3, 1, 2, 2, 2, 5, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.indices(), indices_expected);
}

TEST_F(DictionaryAddKeysTest, FloatColumn)
{
  cudf::test::fixed_width_column_wrapper<float> input{4.25, 7.125, 0.5, -11.75, 7.125, 0.5};
  cudf::test::fixed_width_column_wrapper<float> new_keys{4.25, -11.75, 5.0};

  auto dictionary = cudf::dictionary::encode(input);
  auto result =
    cudf::dictionary::add_keys(cudf::dictionary_column_view(dictionary->view()), new_keys);
  cudf::dictionary_column_view view(result->view());

  cudf::test::fixed_width_column_wrapper<float> keys_expected{-11.75, 0.5, 4.25, 5.0, 7.125};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.keys(), keys_expected);

  cudf::test::fixed_width_column_wrapper<uint8_t> expected{2, 4, 1, 0, 4, 1};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.indices(), expected);
}

TEST_F(DictionaryAddKeysTest, WithNull)
{
  cudf::test::fixed_width_column_wrapper<int64_t> input{{555, 0, 333, 111, 222, 222, 222, 555, 0},
                                                        {1, 1, 1, 0, 1, 1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int64_t> new_keys{0, 111, 444, 777};

  auto dictionary = cudf::dictionary::encode(input);
  auto result =
    cudf::dictionary::add_keys(cudf::dictionary_column_view(dictionary->view()), new_keys);
  auto decoded = cudf::dictionary::decode(result->view());

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(decoded->view(), input);  // new keys should not change anything
}

TEST_F(DictionaryAddKeysTest, Errors)
{
  cudf::test::fixed_width_column_wrapper<int64_t> input{1, 2, 3};
  auto dictionary = cudf::dictionary::encode(input);

  cudf::test::fixed_width_column_wrapper<float> new_keys{1.0, 2.0, 3.0};
  EXPECT_THROW(cudf::dictionary::add_keys(dictionary->view(), new_keys), cudf::data_type_error);
  cudf::test::fixed_width_column_wrapper<int64_t> null_keys{{1, 2, 3}, {1, 0, 1}};
  EXPECT_THROW(cudf::dictionary::add_keys(dictionary->view(), null_keys), cudf::logic_error);
}

CUDF_TEST_PROGRAM_MAIN()
