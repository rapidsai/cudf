/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/dictionary/detail/search.hpp>
#include <cudf/dictionary/search.hpp>
#include <cudf/utilities/memory_resource.hpp>

struct DictionarySearchTest : public cudf::test::BaseFixture {};

TEST_F(DictionarySearchTest, StringsColumn)
{
  cudf::test::dictionary_column_wrapper<std::string> dictionary(
    {"fff", "aaa", "ddd", "bbb", "ccc", "ccc", "ccc", "", ""},
    {true, true, true, true, true, true, true, true, false});

  auto result = cudf::dictionary::get_index(dictionary, cudf::string_scalar("ccc"));
  EXPECT_TRUE(result->is_valid());

  result = cudf::dictionary::get_index(dictionary, cudf::string_scalar("eee"));
  EXPECT_FALSE(result->is_valid());

  result        = cudf::dictionary::detail::get_insert_index(dictionary,
                                                      cudf::string_scalar("eee"),
                                                      cudf::get_default_stream(),
                                                      cudf::get_current_device_resource_ref());
  auto n_result = dynamic_cast<cudf::numeric_scalar<cudf::size_type>*>(result.get());

  auto view = cudf::dictionary_column_view(dictionary);
  EXPECT_EQ(view.keys().size(), n_result->value());
}

TEST_F(DictionarySearchTest, WithNulls)
{
  cudf::test::dictionary_column_wrapper<int64_t> dictionary({9, 8, 7, 6, 4},
                                                            {false, true, true, false, true});

  auto result = cudf::dictionary::get_index(dictionary, cudf::numeric_scalar<int64_t>(4));
  EXPECT_TRUE(result->is_valid());

  result = cudf::dictionary::get_index(dictionary, cudf::numeric_scalar<int64_t>(5));
  EXPECT_FALSE(result->is_valid());
  result = cudf::dictionary::detail::get_insert_index(dictionary,
                                                      cudf::numeric_scalar<int64_t>(5),
                                                      cudf::get_default_stream(),
                                                      cudf::get_current_device_resource_ref());
  EXPECT_TRUE(result->is_valid());
}

TEST_F(DictionarySearchTest, EmptyColumn)
{
  cudf::test::dictionary_column_wrapper<int64_t> dictionary{};
  cudf::numeric_scalar<int64_t> key(7);
  auto result = cudf::dictionary::get_index(dictionary, key);
  EXPECT_FALSE(result->is_valid());
  result = cudf::dictionary::detail::get_insert_index(
    dictionary, key, cudf::get_default_stream(), cudf::get_current_device_resource_ref());
  EXPECT_FALSE(result->is_valid());
}

TEST_F(DictionarySearchTest, Errors)
{
  cudf::test::dictionary_column_wrapper<int64_t> dictionary({1, 2, 3});
  cudf::numeric_scalar<double> key(7);
  EXPECT_THROW(cudf::dictionary::get_index(dictionary, key), cudf::data_type_error);
  EXPECT_THROW(
    cudf::dictionary::detail::get_insert_index(
      dictionary, key, cudf::get_default_stream(), cudf::get_current_device_resource_ref()),
    cudf::data_type_error);
}
