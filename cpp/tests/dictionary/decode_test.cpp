/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
  cudf::test::fixed_width_column_wrapper<int64_t> input{
    {444, 0, 333, 111, 222, 222, 222, 444, 000},
    {true, true, true, true, true, false, true, true, true}};

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
