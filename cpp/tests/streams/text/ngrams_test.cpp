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
    cudf::strings_column_view(input), 5, cudf::test::get_default_stream());
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
