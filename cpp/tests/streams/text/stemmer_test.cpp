/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
