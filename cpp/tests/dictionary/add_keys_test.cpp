/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/dictionary/update_keys.hpp>
#include <cudf/utilities/error.hpp>

struct DictionaryAddKeysTest : public cudf::test::BaseFixture {};

TEST_F(DictionaryAddKeysTest, Strings)
{
  auto input = cudf::test::strings_column_wrapper(
    {"fff", "aaa", "ddd", "bbb", "ccc", "ccc", "ccc", "fff", "aaa"});
  cudf::test::strings_column_wrapper new_keys({"ddd", "bbb", "eee"});

  auto dictionary = cudf::dictionary::encode(input);
  auto result =
    cudf::dictionary::add_keys(cudf::dictionary_column_view(dictionary->view()), new_keys);

  cudf::dictionary_column_view view(result->view());
  auto decode = cudf::dictionary::decode(view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, decode->view());
}

TEST_F(DictionaryAddKeysTest, Floats)
{
  cudf::test::fixed_width_column_wrapper<float> input{4.25, 7.125, 0.5, -11.75, 7.125, 0.5};
  cudf::test::fixed_width_column_wrapper<float> new_keys{4.25, -11.75, 5.0};

  auto dictionary = cudf::dictionary::encode(input);
  auto result =
    cudf::dictionary::add_keys(cudf::dictionary_column_view(dictionary->view()), new_keys);
  cudf::dictionary_column_view view(result->view());
  auto decode = cudf::dictionary::decode(view);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(input, decode->view());
}

TEST_F(DictionaryAddKeysTest, WithNull)
{
  cudf::test::fixed_width_column_wrapper<int64_t> input{{555, 0, 333, 111, 222, 222, 222, 555, 0},
                                                        {1, 1, 1, 0, 1, 1, 1, 1, 1}};
  cudf::test::fixed_width_column_wrapper<int64_t> new_keys{0, 111, 444, 777};

  auto dictionary = cudf::dictionary::encode(input);
  auto result =
    cudf::dictionary::add_keys(cudf::dictionary_column_view(dictionary->view()), new_keys);
  cudf::dictionary_column_view view(result->view());
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
  EXPECT_THROW(cudf::dictionary::add_keys(dictionary->view(), null_keys), std::invalid_argument);
}

CUDF_TEST_PROGRAM_MAIN()
