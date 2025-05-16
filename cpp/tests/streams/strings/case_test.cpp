/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
