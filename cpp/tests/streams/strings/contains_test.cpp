/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>

#include <cudf/strings/contains.hpp>
#include <cudf/strings/regex/regex_program.hpp>

#include <string>

class StringsContainsTest : public cudf::test::BaseFixture {};

TEST_F(StringsContainsTest, Contains)
{
  auto input = cudf::test::strings_column_wrapper({"Héllo", "thesé", "tést strings", ""});
  auto view  = cudf::strings_column_view(input);

  auto const pattern = std::string("[a-z]");
  auto const prog    = cudf::strings::regex_program::create(pattern);
  cudf::strings::contains_re(view, *prog, cudf::test::get_default_stream());
  cudf::strings::matches_re(view, *prog, cudf::test::get_default_stream());
  cudf::strings::count_re(view, *prog, cudf::test::get_default_stream());
}

TEST_F(StringsContainsTest, Like)
{
  auto input = cudf::test::strings_column_wrapper({"Héllo", "thesés", "tést", ""});
  auto view  = cudf::strings_column_view(input);

  auto const pattern = std::string_view("%és");
  auto const escape  = std::string_view("%");
  cudf::strings::like(view, pattern, escape, cudf::test::get_default_stream());

  auto const s_escape = cudf::string_scalar(escape, true, cudf::test::get_default_stream());
  auto const patterns = cudf::test::strings_column_wrapper({"H%", "t%s", "t", ""});
  cudf::strings::like(
    view, cudf::strings_column_view(patterns), s_escape, cudf::test::get_default_stream());
}
