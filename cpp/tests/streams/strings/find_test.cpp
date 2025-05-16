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

#include <cudf/strings/find.hpp>
#include <cudf/strings/find_multiple.hpp>
#include <cudf/strings/findall.hpp>
#include <cudf/strings/regex/regex_program.hpp>

#include <string>

class StringsFindTest : public cudf::test::BaseFixture {};

TEST_F(StringsFindTest, Find)
{
  auto input = cudf::test::strings_column_wrapper({"Héllo", "thesé", "tést strings", ""});
  auto view  = cudf::strings_column_view(input);

  auto const target = cudf::string_scalar("é", true, cudf::test::get_default_stream());
  cudf::strings::find(view, target, 0, -1, cudf::test::get_default_stream());
  cudf::strings::rfind(view, target, 0, -1, cudf::test::get_default_stream());
  cudf::strings::find(view, view, 0, cudf::test::get_default_stream());
  cudf::strings::find_multiple(view, view, cudf::test::get_default_stream());
  cudf::strings::contains(view, target, cudf::test::get_default_stream());
  cudf::strings::starts_with(view, target, cudf::test::get_default_stream());
  cudf::strings::starts_with(view, view, cudf::test::get_default_stream());
  cudf::strings::ends_with(view, target, cudf::test::get_default_stream());
  cudf::strings::ends_with(view, view, cudf::test::get_default_stream());

  auto const pattern = std::string("[a-z]");
  auto const prog    = cudf::strings::regex_program::create(pattern);
  cudf::strings::findall(view, *prog, cudf::test::get_default_stream());
  cudf::strings::find_re(view, *prog, cudf::test::get_default_stream());
}
