/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>

#include <nvtext/normalize.hpp>
#include <nvtext/replace.hpp>

class TextReplaceTest : public cudf::test::BaseFixture {};

TEST_F(TextReplaceTest, Replace)
{
  auto const input     = cudf::test::strings_column_wrapper({"the fox jumped over the dog"});
  auto const targets   = cudf::test::strings_column_wrapper({"the", "dog"});
  auto const repls     = cudf::test::strings_column_wrapper({"_", ""});
  auto const delimiter = cudf::string_scalar{" ", true, cudf::test::get_default_stream()};
  nvtext::replace_tokens(cudf::strings_column_view(input),
                         cudf::strings_column_view(targets),
                         cudf::strings_column_view(repls),
                         delimiter,
                         cudf::test::get_default_stream());
}

TEST_F(TextReplaceTest, Filter)
{
  auto const input     = cudf::test::strings_column_wrapper({"one two three", "four five six"});
  auto const delimiter = cudf::string_scalar{" ", true, cudf::test::get_default_stream()};
  auto const repl      = cudf::string_scalar{"_", true, cudf::test::get_default_stream()};
  nvtext::filter_tokens(
    cudf::strings_column_view(input), 1, delimiter, repl, cudf::test::get_default_stream());
}

TEST_F(TextReplaceTest, NormalizeSpaces)
{
  auto input =
    cudf::test::strings_column_wrapper({"the\tquick brown\nfox", "jumped\rover the lazy\r\t\n"});
  nvtext::normalize_spaces(cudf::strings_column_view(input), cudf::test::get_default_stream());
}

TEST_F(TextReplaceTest, NormalizeCharacters)
{
  auto input = cudf::test::strings_column_wrapper({"abc£def", "éè â îô\taeio", "\tĂĆĖÑ  Ü"});
  auto sv    = cudf::strings_column_view(input);
  auto cn    = nvtext::create_character_normalizer(false, sv, cudf::test::get_default_stream());
  nvtext::normalize_characters(sv, *cn, cudf::test::get_default_stream());
}
