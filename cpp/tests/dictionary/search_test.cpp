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
  auto n_result = dynamic_cast<cudf::numeric_scalar<uint32_t>*>(result.get());
  EXPECT_EQ(uint32_t{3}, n_result->value());

  result = cudf::dictionary::get_index(dictionary, cudf::string_scalar("eee"));
  EXPECT_FALSE(result->is_valid());
  result   = cudf::dictionary::detail::get_insert_index(dictionary,
                                                      cudf::string_scalar("eee"),
                                                      cudf::get_default_stream(),
                                                      cudf::get_current_device_resource_ref());
  n_result = dynamic_cast<cudf::numeric_scalar<uint32_t>*>(result.get());
  EXPECT_EQ(uint32_t{5}, n_result->value());
}

TEST_F(DictionarySearchTest, WithNulls)
{
  cudf::test::dictionary_column_wrapper<int64_t> dictionary({9, 8, 7, 6, 4},
                                                            {false, true, true, false, true});

  auto result = cudf::dictionary::get_index(dictionary, cudf::numeric_scalar<int64_t>(4));
  EXPECT_TRUE(result->is_valid());
  auto n_result = dynamic_cast<cudf::numeric_scalar<uint32_t>*>(result.get());
  EXPECT_EQ(uint32_t{0}, n_result->value());

  result = cudf::dictionary::get_index(dictionary, cudf::numeric_scalar<int64_t>(5));
  EXPECT_FALSE(result->is_valid());
  result   = cudf::dictionary::detail::get_insert_index(dictionary,
                                                      cudf::numeric_scalar<int64_t>(5),
                                                      cudf::get_default_stream(),
                                                      cudf::get_current_device_resource_ref());
  n_result = dynamic_cast<cudf::numeric_scalar<uint32_t>*>(result.get());
  EXPECT_EQ(uint32_t{1}, n_result->value());
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
