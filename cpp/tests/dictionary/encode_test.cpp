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

struct DictionaryEncodeTest : public cudf::test::BaseFixture {};

TEST_F(DictionaryEncodeTest, EncodeStringColumn)
{
  cudf::test::strings_column_wrapper strings(
    {"eee", "aaa", "ddd", "bbb", "ccc", "ccc", "ccc", "eee", "aaa"});

  auto dictionary = cudf::dictionary::encode(strings);
  cudf::dictionary_column_view view(dictionary->view());

  cudf::test::strings_column_wrapper keys_expected({"aaa", "bbb", "ccc", "ddd", "eee"});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.keys(), keys_expected);

  cudf::test::fixed_width_column_wrapper<int32_t> indices_expected({4, 0, 3, 1, 2, 2, 2, 4, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.indices(), indices_expected);
}

TEST_F(DictionaryEncodeTest, EncodeFloat)
{
  cudf::test::fixed_width_column_wrapper<float> input{4.25, 7.125, 0.5, -11.75, 7.125, 0.5};

  auto dictionary = cudf::dictionary::encode(input);
  cudf::dictionary_column_view view(dictionary->view());

  cudf::test::fixed_width_column_wrapper<float> keys_expected{-11.75, 0.5, 4.25, 7.125};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.keys(), keys_expected);

  cudf::test::fixed_width_column_wrapper<int32_t> expected{2, 3, 1, 0, 3, 1};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.indices(), expected);
}

TEST_F(DictionaryEncodeTest, EncodeWithNull)
{
  cudf::test::fixed_width_column_wrapper<int64_t> input{
    {444, 0, 333, 111, 222, 222, 222, 444, 000},
    {true, true, true, true, true, false, true, true, true}};

  auto dictionary = cudf::dictionary::encode(input);
  cudf::dictionary_column_view view(dictionary->view());

  cudf::test::fixed_width_column_wrapper<int64_t> keys_expected{0, 111, 222, 333, 444};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.keys(), keys_expected);

  cudf::test::fixed_width_column_wrapper<int32_t> expected{4, 0, 3, 1, 2, 5, 2, 4, 0};
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(view.indices(), expected);
}

TEST_F(DictionaryEncodeTest, InvalidEncode)
{
  cudf::test::fixed_width_column_wrapper<int16_t> input{0, 1, 2, 3, -1, -2, -3};

  EXPECT_THROW(cudf::dictionary::encode(input, cudf::data_type{cudf::type_id::UINT16}),
               cudf::logic_error);
}
