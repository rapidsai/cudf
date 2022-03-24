/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <nvtext/bpe_tokenize.hpp>

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>

struct TextBPETokenize : public cudf::test::BaseFixture {
};

TEST_F(TextBPETokenize, BytePairEncoding)
{
  // partial table based on values from https://huggingface.co/gpt2/raw/main/merges.txt
  auto mpt = cudf::test::strings_column_wrapper({
    "e n",    // 12
    "i t",    // 14
    "i s",    // 15
    "e s",    // 18
    "en t",   // 42
    "c e",    // 88
    "es t",   // 139
    "en ce",  // 338
    "T h",    // 561
    "Th is",  // 956
    "t est",  // 9032
    "s ent",  // 33830
  });

  nvtext::bpe_merge_pairs merge_pairs{cudf::strings_column_view(mpt)};

  auto validity = cudf::test::iterators::null_at(4);
  cudf::test::strings_column_wrapper input({" This\tis  it\n",
                                            "This is test-sentence-1",
                                            "This is test sentence-2",
                                            "This-is test sentence 3",
                                            "",
                                            ""},
                                           validity);
  auto sv = cudf::strings_column_view(input);

  auto results = nvtext::byte_pair_encoding(sv, merge_pairs);

  auto expected = cudf::test::strings_column_wrapper({" This is it",
                                                      "This is test - sent ence - 1",
                                                      "This is test sent ence - 2",
                                                      "This - is test sent ence 3",
                                                      "",
                                                      ""},
                                                     validity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  auto sliced          = cudf::slice(input, {1, 4}).front();
  auto sliced_expected = cudf::slice(expected, {1, 4}).front();

  results = nvtext::byte_pair_encoding(cudf::strings_column_view(sliced), merge_pairs);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), sliced_expected);
}

TEST_F(TextBPETokenize, BytePairEncodingSeparator)
{
  auto mpt = cudf::test::strings_column_wrapper(
    {"e n", "i t", "e s", "en t", "c e", "es t", "en ce", "t est", "s ent"});
  nvtext::bpe_merge_pairs merge_pairs{cudf::strings_column_view(mpt)};

  cudf::test::strings_column_wrapper input(
    {"test-sentence-1", "test sentence-2", "test sentence 3", " test sentence 4 "});
  auto sv = cudf::strings_column_view(input);

  auto results = nvtext::byte_pair_encoding(sv, merge_pairs, std::string(" Ġ"));

  auto expected = cudf::test::strings_column_wrapper(
    {"test - sent ence - 1", "test Ġsent ence - 2", "test Ġsent ence Ġ3", " Ġtest Ġsent ence Ġ4"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TEST_F(TextBPETokenize, BPE_Empty)
{
  auto mpt = cudf::test::strings_column_wrapper({"i s", "i t"});
  nvtext::bpe_merge_pairs merge_pairs{mpt.release()};
  auto empty   = cudf::make_empty_column(cudf::type_id::STRING);
  auto results = nvtext::byte_pair_encoding(cudf::strings_column_view(empty->view()), merge_pairs);
  EXPECT_EQ(0, results->size());
}

TEST_F(TextBPETokenize, BPE_Error)
{
  auto empty = cudf::make_empty_column(cudf::type_id::STRING);
  nvtext::bpe_merge_pairs merge_pairs{std::move(empty)};
  cudf::test::strings_column_wrapper input({"isit"});
  EXPECT_THROW(nvtext::byte_pair_encoding(cudf::strings_column_view(input), merge_pairs),
               cudf::logic_error);
}
