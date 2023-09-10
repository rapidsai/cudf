/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

struct TextBPETokenize : public cudf::test::BaseFixture {};

TEST_F(TextBPETokenize, BytePairEncoding)
{
  // partial table based on values from https://huggingface.co/gpt2/raw/main/merges.txt
  auto mpt = cudf::test::strings_column_wrapper(
    {"e n", "i t", "i s", "e s", "en t", "c e", "es t", "en ce", "T h", "Th is", "t est", "s ent"});

  auto merge_pairs = nvtext::load_merge_pairs(cudf::strings_column_view(mpt));

  auto validity = cudf::test::iterators::null_at(4);
  cudf::test::strings_column_wrapper input(
    {"Thisisit", "Thisis test-sentence-1", "Thisistestsentence-2", "This-istestsentence 3", "", ""},
    validity);
  auto sv = cudf::strings_column_view(input);

  auto results = nvtext::byte_pair_encoding(sv, *merge_pairs);

  auto expected = cudf::test::strings_column_wrapper({"This is it",
                                                      "This is   test - sent ence - 1",
                                                      "This is test sent ence - 2",
                                                      "This - is test sent ence   3",
                                                      "",
                                                      ""},
                                                     validity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  auto sliced          = cudf::slice(input, {1, 4}).front();
  auto sliced_expected = cudf::slice(expected, {1, 4}).front();
  sv                   = cudf::strings_column_view(sliced);

  results = nvtext::byte_pair_encoding(sv, *merge_pairs);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), sliced_expected);
}

TEST_F(TextBPETokenize, BytePairEncodingSeparator)
{
  auto mpt = cudf::test::strings_column_wrapper(
    {"Ġ t", "Ġt he", "h e", "e n", "i t", "e s", "en t", "c e", "es t", "en ce", "t est", "s ent"});
  auto merge_pairs = nvtext::load_merge_pairs(cudf::strings_column_view(mpt));

  cudf::test::strings_column_wrapper input(
    {"Ġthe test sentence", "test Ġthe sentence", "Ġthetest sentence", "testĠthesentence"});
  auto sv = cudf::strings_column_view(input);

  auto results = nvtext::byte_pair_encoding(sv, *merge_pairs, std::string("$"));

  auto expected = cudf::test::strings_column_wrapper({"Ġthe$ $test$ $sent$ence",
                                                      "test$ $Ġthe$ $sent$ence",
                                                      "Ġthe$test$ $sent$ence",
                                                      "test$Ġthe$sent$ence"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TEST_F(TextBPETokenize, DISABLED_BPEAdjacentPairs)
{
  auto mpt         = cudf::test::strings_column_wrapper({
    "▁ H",    //    157
    "m m",    //  10742
    "? !",    //  50675
    "▁H mm",  // 174381
    "mm m",   // 262776
    "?! !",   // 352313
    "? !?",   // 352314
    "mm mm",  // 387733
    "▁H m",   // 471269
    "?! ?!",  // 506981
    "?!? !",  // 506982
  });
  auto merge_pairs = nvtext::load_merge_pairs(cudf::strings_column_view(mpt));

  cudf::test::strings_column_wrapper input({"▁Hmmmmm", "?!?!?!"});

  auto results  = nvtext::byte_pair_encoding(cudf::strings_column_view(input), *merge_pairs);
  auto expected = cudf::test::strings_column_wrapper({"▁Hmm mmm", "?!?! ?!"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TEST_F(TextBPETokenize, BPE_Empty)
{
  auto mpt         = cudf::test::strings_column_wrapper({"i s", "i t"});
  auto merge_pairs = nvtext::load_merge_pairs(cudf::strings_column_view(mpt));
  auto empty       = cudf::make_empty_column(cudf::type_id::STRING);
  auto results = nvtext::byte_pair_encoding(cudf::strings_column_view(empty->view()), *merge_pairs);
  EXPECT_EQ(0, results->size());
}

TEST_F(TextBPETokenize, BPE_Error)
{
  auto empty = cudf::make_empty_column(cudf::type_id::STRING);
  EXPECT_THROW(nvtext::load_merge_pairs(cudf::strings_column_view(*empty)), cudf::logic_error);
  auto null_pairs = cudf::test::strings_column_wrapper({"", ""}, {1, 0});
  EXPECT_THROW(nvtext::load_merge_pairs(cudf::strings_column_view(null_pairs)), cudf::logic_error);
}
