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

#include <nvtext/stemmer.hpp>

class TextStemmerTest : public cudf::test::BaseFixture {};

TEST_F(TextStemmerTest, IsLetter)
{
  auto const input =
    cudf::test::strings_column_wrapper({"abbey", "normal", "creates", "yearly", "trouble"});
  auto const view      = cudf::strings_column_view(input);
  auto const delimiter = cudf::string_scalar{" ", true, cudf::test::get_default_stream()};
  nvtext::is_letter(view, nvtext::letter_type::VOWEL, 0, cudf::test::get_default_stream());
  auto const indices = cudf::test::fixed_width_column_wrapper<int32_t>({0, 1, 3, 5, 4});
  nvtext::is_letter(view, nvtext::letter_type::VOWEL, indices, cudf::test::get_default_stream());
}

TEST_F(TextStemmerTest, Porter)
{
  auto const input =
    cudf::test::strings_column_wrapper({"abbey", "normal", "creates", "yearly", "trouble"});
  auto const view = cudf::strings_column_view(input);
  nvtext::porter_stemmer_measure(view, cudf::test::get_default_stream());
}
