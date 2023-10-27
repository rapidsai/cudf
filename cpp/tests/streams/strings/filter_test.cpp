/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cudf/strings/char_types/char_types.hpp>
#include <cudf/strings/translate.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>

#include <string>
#include <vector>

class StringsFilterTest : public cudf::test::BaseFixture {};

static std::pair<cudf::char_utf8, cudf::char_utf8> make_entry(char const* from, char const* to)
{
  cudf::char_utf8 in  = 0;
  cudf::char_utf8 out = 0;
  cudf::strings::detail::to_char_utf8(from, in);
  if (to) cudf::strings::detail::to_char_utf8(to, out);
  return std::pair(in, out);
}

TEST_F(StringsFilterTest, Translate)
{
  auto input = cudf::test::strings_column_wrapper({"  aBc  ", "   ", "aaaa ", "\tb"});
  auto view  = cudf::strings_column_view(input);

  std::vector<std::pair<cudf::char_utf8, cudf::char_utf8>> translate_table{
    make_entry("b", 0), make_entry("a", "A"), make_entry(" ", "_")};
  cudf::strings::translate(view, translate_table, cudf::test::get_default_stream());
}

TEST_F(StringsFilterTest, Filter)
{
  auto input = cudf::test::strings_column_wrapper({"  aBc  ", "   ", "aaaa ", "\tb"});
  auto view  = cudf::strings_column_view(input);

  std::vector<std::pair<cudf::char_utf8, cudf::char_utf8>> filter_table{
    make_entry("b", 0), make_entry("a", "A"), make_entry(" ", "_")};

  auto const repl = cudf::string_scalar("X", true, cudf::test::get_default_stream());
  auto const keep = cudf::strings::filter_type::KEEP;
  cudf::strings::filter_characters(
    view, filter_table, keep, repl, cudf::test::get_default_stream());
}

TEST_F(StringsFilterTest, FilterTypes)
{
  auto input = cudf::test::strings_column_wrapper({"  aBc  ", "   ", "aaaa ", "\tb"});
  auto view  = cudf::strings_column_view(input);

  auto const verify_types =
    cudf::strings::string_character_types::LOWER | cudf::strings::string_character_types::UPPER;
  auto const all_types = cudf::strings::string_character_types::ALL_TYPES;
  cudf::strings::all_characters_of_type(
    view, verify_types, all_types, cudf::test::get_default_stream());

  auto const repl        = cudf::string_scalar("X", true, cudf::test::get_default_stream());
  auto const space_types = cudf::strings::string_character_types::SPACE;
  cudf::strings::filter_characters_of_type(
    view, all_types, repl, space_types, cudf::test::get_default_stream());
}
