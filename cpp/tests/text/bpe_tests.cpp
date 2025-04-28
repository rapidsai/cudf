/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/byte_pair_encoding.hpp>

struct TextBytePairEncoding : public cudf::test::BaseFixture {};

TEST_F(TextBytePairEncoding, BytePairEncoding)
{
  // partial table based on values from https://huggingface.co/gpt2/raw/main/merges.txt
  auto mpt = cudf::test::strings_column_wrapper({
    "e n",    // 14
    "i t",    // 16
    "i s",    // 17
    "e s",    // 20
    "en t",   // 44
    "c e",    // 90
    "es t",   // 141
    "en ce",  // 340
    "t h",    // 146
    "h i",    // 5049
    "th is",  // 5407
    "t est",  // 9034
    "s i",    // 13142
    "s ent"   // 33832
  });

  auto merge_pairs = nvtext::load_merge_pairs(cudf::strings_column_view(mpt));

  auto validity = cudf::test::iterators::null_at(4);
  cudf::test::strings_column_wrapper input(
    {"thisisit", "thisis test-sentence-1", "thisistestsentence-2", "this-istestsentence 3", "", ""},
    validity);
  auto sv = cudf::strings_column_view(input);

  auto results  = nvtext::byte_pair_encoding(sv, *merge_pairs);
  auto expected = cudf::test::strings_column_wrapper({"this is it",
                                                      "this is   test - sent ence - 1",
                                                      "this is test sent ence - 2",
                                                      "this - is test sent ence   3",
                                                      "",
                                                      ""},
                                                     validity);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);

  auto sliced          = cudf::slice(input, {1, 4}).front();
  auto sliced_expected = cudf::slice(expected, {1, 4}).front();

  sv      = cudf::strings_column_view(sliced);
  results = nvtext::byte_pair_encoding(sv, *merge_pairs);
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), sliced_expected);
}

TEST_F(TextBytePairEncoding, BytePairEncodingSeparator)
{
  auto mpt = cudf::test::strings_column_wrapper(
    {"Ġ t", "Ġt he", "h e", "e n", "i t", "e s", "en t", "c e", "es t", "en ce", "t est", "s ent"});

  auto merge_pairs = nvtext::load_merge_pairs(cudf::strings_column_view(mpt));

  cudf::test::strings_column_wrapper input(
    {"Ġthe test sentence", "test Ġthe sentence", "Ġthetest sentence", "testĠthesentence"});
  auto sv = cudf::strings_column_view(input);

  auto results = nvtext::byte_pair_encoding(sv, *merge_pairs, std::string_view("$"));

  auto expected = cudf::test::strings_column_wrapper({"Ġthe$ $test$ $sent$ence",
                                                      "test$ $Ġthe$ $sent$ence",
                                                      "Ġthe$test$ $sent$ence",
                                                      "test$Ġthe$sent$ence"});
  CUDF_TEST_EXPECT_COLUMNS_EQUIVALENT(results->view(), expected);
}

TEST_F(TextBytePairEncoding, BPEAdjacentPairs)
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

TEST_F(TextBytePairEncoding, BPE_Empty)
{
  auto mpt         = cudf::test::strings_column_wrapper({"i s", "i t"});
  auto merge_pairs = nvtext::load_merge_pairs(cudf::strings_column_view(mpt));
  auto empty       = cudf::make_empty_column(cudf::type_id::STRING);
  auto results = nvtext::byte_pair_encoding(cudf::strings_column_view(empty->view()), *merge_pairs);
  EXPECT_EQ(0, results->size());
}

TEST_F(TextBytePairEncoding, BPE_Error)
{
  auto empty = cudf::make_empty_column(cudf::type_id::STRING);
  EXPECT_THROW(nvtext::load_merge_pairs(cudf::strings_column_view(*empty)), cudf::logic_error);
  auto null_pairs = cudf::test::strings_column_wrapper({"", ""}, {true, false});
  EXPECT_THROW(nvtext::load_merge_pairs(cudf::strings_column_view(null_pairs)), cudf::logic_error);
}
