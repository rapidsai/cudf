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

#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>

#include <vector>

struct DictionaryDecodeTest : public cudf::test::BaseFixture {};

TEST_F(DictionaryDecodeTest, StringColumn)
{
  std::vector<char const*> h_strings{"eee", "aaa", "ddd", "bbb", "ccc", "ccc", "ccc", "eee", "aaa"};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());

  auto dictionary = cudf::dictionary::encode(strings);
  auto output     = cudf::dictionary::decode(cudf::dictionary_column_view(dictionary->view()));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(strings, *output);
}

TEST_F(DictionaryDecodeTest, FloatColumn)
{
  cudf::test::fixed_width_column_wrapper<float> input{4.25, 7.125, 0.5, -11.75, 7.125, 0.5};

  auto dictionary = cudf::dictionary::encode(input);
  auto output     = cudf::dictionary::decode(cudf::dictionary_column_view(dictionary->view()));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, *output);
}

TEST_F(DictionaryDecodeTest, ColumnWithNull)
{
  cudf::test::fixed_width_column_wrapper<int64_t> input{{444, 0, 333, 111, 222, 222, 222, 444, 000},
                                                        {1, 1, 1, 1, 1, 0, 1, 1, 1}};

  auto dictionary = cudf::dictionary::encode(input);
  auto output     = cudf::dictionary::decode(cudf::dictionary_column_view(dictionary->view()));

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, *output);
}

TEST_F(DictionaryDecodeTest, EmptyColumn)
{
  cudf::test::fixed_width_column_wrapper<int16_t> input;
  auto dictionary = cudf::dictionary::encode(input);
  auto output     = cudf::dictionary::decode(cudf::dictionary_column_view(dictionary->view()));

  // check empty
  EXPECT_EQ(output->size(), 0);
  EXPECT_EQ(output->type().id(), cudf::type_id::EMPTY);
}
