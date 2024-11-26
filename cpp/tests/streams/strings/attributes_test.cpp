/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <cudf_test/default_stream.hpp>

#include <cudf/strings/attributes.hpp>
#include <cudf/strings/strings_column_view.hpp>

struct StringsAttributesTest : public cudf::test::BaseFixture {};

TEST_F(StringsAttributesTest, CodePoints)
{
  std::vector<char const*> h_strings{"eee", "bb", nullptr, "", "aa", "bbb", "ééé"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  auto strings_view = cudf::strings_column_view(strings);

  auto results = cudf::strings::code_points(strings_view, cudf::test::get_default_stream());
}

TEST_F(StringsAttributesTest, CountCharacters)
{
  std::vector<std::string> h_strings(
    40000, "something a bit longer than 32 bytes ééé ééé ééé ééé ééé ééé ééé");
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  auto strings_view = cudf::strings_column_view(strings);

  auto results = cudf::strings::count_characters(strings_view, cudf::test::get_default_stream());
}

TEST_F(StringsAttributesTest, CountBytes)
{
  std::vector<char const*> h_strings{
    "eee", "bb", nullptr, "", "aa", "ééé", "something a bit longer than 32 bytes"};
  cudf::test::strings_column_wrapper strings(
    h_strings.begin(),
    h_strings.end(),
    thrust::make_transform_iterator(h_strings.begin(), [](auto str) { return str != nullptr; }));
  auto strings_view = cudf::strings_column_view(strings);

  auto results = cudf::strings::count_bytes(strings_view, cudf::test::get_default_stream());
}
