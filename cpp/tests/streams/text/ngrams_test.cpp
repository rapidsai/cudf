/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/default_stream.hpp>

#include <nvtext/generate_ngrams.hpp>
#include <nvtext/ngrams_tokenize.hpp>

class TextNGramsTest : public cudf::test::BaseFixture {};

TEST_F(TextNGramsTest, GenerateNgrams)
{
  auto const input =
    cudf::test::strings_column_wrapper({"the", "fox", "jumped", "over", "thé", "dog"});
  auto const separator = cudf::string_scalar{"_", true, cudf::test::get_default_stream()};
  nvtext::generate_ngrams(
    cudf::strings_column_view(input), 3, separator, cudf::test::get_default_stream());
}

TEST_F(TextNGramsTest, GenerateCharacterNgrams)
{
  auto const input =
    cudf::test::strings_column_wrapper({"the", "fox", "jumped", "over", "thé", "dog"});
  nvtext::generate_character_ngrams(
    cudf::strings_column_view(input), 3, cudf::test::get_default_stream());
}

TEST_F(TextNGramsTest, HashCharacterNgrams)
{
  auto input =
    cudf::test::strings_column_wrapper({"the quick brown fox", "jumped over the lazy dog."});
  nvtext::hash_character_ngrams(
    cudf::strings_column_view(input), 5, 5, cudf::test::get_default_stream());
}

TEST_F(TextNGramsTest, NgramsTokenize)
{
  auto input =
    cudf::test::strings_column_wrapper({"the quick brown fox", "jumped over the lazy dog."});
  auto const delimiter = cudf::string_scalar{" ", true, cudf::test::get_default_stream()};
  auto const separator = cudf::string_scalar{"_", true, cudf::test::get_default_stream()};
  nvtext::ngrams_tokenize(
    cudf::strings_column_view(input), 2, delimiter, separator, cudf::test::get_default_stream());
}
