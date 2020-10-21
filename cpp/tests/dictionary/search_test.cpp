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

#include <cudf/dictionary/detail/search.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/dictionary/search.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <vector>

struct DictionarySearchTest : public cudf::test::BaseFixture {
};

TEST_F(DictionarySearchTest, StringsColumn)
{
  std::vector<const char*> h_strings{"fff", "aaa", "ddd", "bbb", "ccc", "ccc", "ccc", "", nullptr};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));

  auto dictionary = cudf::dictionary::encode(strings);

  cudf::string_scalar key("ccc");
  auto result = cudf::dictionary::get_index(cudf::dictionary_column_view(dictionary->view()), key);
  EXPECT_TRUE(result->is_valid());
  auto n_result = dynamic_cast<cudf::numeric_scalar<uint32_t>*>(result.get());
  EXPECT_EQ(3, n_result->value());

  cudf::string_scalar no_key("eee");
  result = cudf::dictionary::get_index(cudf::dictionary_column_view(dictionary->view()), no_key);
  EXPECT_FALSE(result->is_valid());
  result = cudf::dictionary::detail::get_insert_index(
    cudf::dictionary_column_view(dictionary->view()), no_key);
  n_result = dynamic_cast<cudf::numeric_scalar<uint32_t>*>(result.get());
  EXPECT_EQ(5, n_result->value());
}

TEST_F(DictionarySearchTest, WithNulls)
{
  cudf::test::fixed_width_column_wrapper<int64_t> input({9, 8, 7, 6, 4}, {0, 1, 1, 0, 1});
  auto dictionary = cudf::dictionary::encode(input);

  cudf::numeric_scalar<int64_t> key(4);
  auto result = cudf::dictionary::get_index(cudf::dictionary_column_view(dictionary->view()), key);
  EXPECT_TRUE(result->is_valid());
  auto n_result = dynamic_cast<cudf::numeric_scalar<uint32_t>*>(result.get());
  EXPECT_EQ(0, n_result->value());

  cudf::numeric_scalar<int64_t> no_key(5);
  result = cudf::dictionary::get_index(cudf::dictionary_column_view(dictionary->view()), no_key);
  EXPECT_FALSE(result->is_valid());
  result = cudf::dictionary::detail::get_insert_index(
    cudf::dictionary_column_view(dictionary->view()), no_key);
  n_result = dynamic_cast<cudf::numeric_scalar<uint32_t>*>(result.get());
  EXPECT_EQ(1, n_result->value());
}

TEST_F(DictionarySearchTest, EmptyColumn)
{
  cudf::test::fixed_width_column_wrapper<int64_t> input{};
  auto dictionary = cudf::dictionary::encode(input);
  cudf::numeric_scalar<int64_t> key(7);
  auto result = cudf::dictionary::get_index(cudf::dictionary_column_view(dictionary->view()), key);
  EXPECT_FALSE(result->is_valid());
  result = cudf::dictionary::detail::get_insert_index(
    cudf::dictionary_column_view(dictionary->view()), key);
  EXPECT_FALSE(result->is_valid());
}

TEST_F(DictionarySearchTest, Errors)
{
  cudf::test::fixed_width_column_wrapper<int64_t> input({1, 2, 3});
  auto dictionary = cudf::dictionary::encode(input);
  cudf::numeric_scalar<double> key(7);
  EXPECT_THROW(cudf::dictionary::get_index(cudf::dictionary_column_view(dictionary->view()), key),
               cudf::logic_error);
  EXPECT_THROW(cudf::dictionary::detail::get_insert_index(
                 cudf::dictionary_column_view(dictionary->view()), key),
               cudf::logic_error);
}
