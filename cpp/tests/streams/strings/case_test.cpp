/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>

#include <cudf/strings/capitalize.hpp>
#include <cudf/strings/case.hpp>

class StringsCaseTest : public cudf::test::BaseFixture {};

TEST_F(StringsCaseTest, LowerUpper)
{
  auto const input =
    cudf::test::strings_column_wrapper({"",
                                        "The quick brown fox",
                                        "jumps over the lazy dog.",
                                        "all work and no play makes Jack a dull boy",
                                        R"(!"#$%&'()*+,-./0123456789:;<=>?@[\]^_`{|}~)"});
  auto view = cudf::strings_column_view(input);

  cudf::strings::to_lower(view, cudf::test::get_default_stream());
  cudf::strings::to_upper(view, cudf::test::get_default_stream());
  cudf::strings::swapcase(view, cudf::test::get_default_stream());
}

TEST_F(StringsCaseTest, Capitalize)
{
  auto const input =
    cudf::test::strings_column_wrapper({"",
                                        "The Quick Brown Fox",
                                        "jumps over the lazy dog",
                                        "all work and no play makes Jack a dull boy"});
  auto view = cudf::strings_column_view(input);

  auto const delimiter = cudf::string_scalar(" ", true, cudf::test::get_default_stream());
  cudf::strings::capitalize(view, delimiter, cudf::test::get_default_stream());
  cudf::strings::is_title(view, cudf::test::get_default_stream());
  cudf::strings::title(
    view, cudf::strings::string_character_types::ALPHA, cudf::test::get_default_stream());
}
