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

#include <nvtext/tokenize.hpp>

class TextTokenizeTest : public cudf::test::BaseFixture {};

TEST_F(TextTokenizeTest, Tokenize)
{
  auto const input     = cudf::test::strings_column_wrapper({"the fox jumped", "over thé dog"});
  auto const view      = cudf::strings_column_view(input);
  auto const delimiter = cudf::string_scalar{" ", true, cudf::test::get_default_stream()};
  nvtext::tokenize(view, delimiter, cudf::test::get_default_stream());
  nvtext::count_tokens(view, delimiter, cudf::test::get_default_stream());
  auto const delimiters = cudf::test::strings_column_wrapper({" ", "o", "é"});
  nvtext::tokenize(view, cudf::strings_column_view(delimiters), cudf::test::get_default_stream());
  nvtext::count_tokens(
    view, cudf::strings_column_view(delimiters), cudf::test::get_default_stream());
}

TEST_F(TextTokenizeTest, CharacterTokenize)
{
  auto const input =
    cudf::test::strings_column_wrapper({"the", "fox", "jumped", "over", "thé", "dog"});
  nvtext::character_tokenize(cudf::strings_column_view(input), cudf::test::get_default_stream());
}

TEST_F(TextTokenizeTest, Detokenize)
{
  auto const input =
    cudf::test::strings_column_wrapper({"the", "fox", "jumped", "over", "thé", "dog"});
  auto const view      = cudf::strings_column_view(input);
  auto const indices   = cudf::test::fixed_width_column_wrapper<int32_t>({0, 0, 0, 1, 1, 1});
  auto const separator = cudf::string_scalar{" ", true, cudf::test::get_default_stream()};
  nvtext::detokenize(view, indices, separator, cudf::test::get_default_stream());
}
