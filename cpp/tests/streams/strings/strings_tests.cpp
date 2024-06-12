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

#include <cudf/strings/padding.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/strip.hpp>
#include <cudf/strings/wrap.hpp>

#include <string>

class StringsTest : public cudf::test::BaseFixture {};

TEST_F(StringsTest, Strip)
{
  auto input = cudf::test::strings_column_wrapper({"  aBc  ", "   ", "aaaa ", "\tb"});
  auto view  = cudf::strings_column_view(input);

  auto const strip = cudf::string_scalar(" ", true, cudf::test::get_default_stream());
  auto const side  = cudf::strings::side_type::BOTH;
  cudf::strings::strip(view, side, strip, cudf::test::get_default_stream());
}

TEST_F(StringsTest, Pad)
{
  auto input = cudf::test::strings_column_wrapper({"333", "", "4444", "1"});
  auto view  = cudf::strings_column_view(input);

  auto const side = cudf::strings::side_type::BOTH;
  cudf::strings::pad(view, 6, side, " ", cudf::test::get_default_stream());
  cudf::strings::zfill(view, 6, cudf::test::get_default_stream());
}

TEST_F(StringsTest, Wrap)
{
  auto input = cudf::test::strings_column_wrapper({"the quick brown fox jumped"});
  auto view  = cudf::strings_column_view(input);

  cudf::strings::wrap(view, 6, cudf::test::get_default_stream());
}

TEST_F(StringsTest, Slice)
{
  auto input = cudf::test::strings_column_wrapper({"hello", "these", "are test strings"});
  auto view  = cudf::strings_column_view(input);

  auto start = cudf::numeric_scalar(2, true, cudf::test::get_default_stream());
  auto stop  = cudf::numeric_scalar(5, true, cudf::test::get_default_stream());
  auto step  = cudf::numeric_scalar(1, true, cudf::test::get_default_stream());
  cudf::strings::slice_strings(view, start, stop, step, cudf::test::get_default_stream());

  auto starts = cudf::test::fixed_width_column_wrapper<cudf::size_type>({1, 2, 3});
  auto stops  = cudf::test::fixed_width_column_wrapper<cudf::size_type>({4, 5, 6});
  cudf::strings::slice_strings(view, starts, stops, cudf::test::get_default_stream());
}
