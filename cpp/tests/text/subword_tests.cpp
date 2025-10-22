/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/subword_tokenize.hpp>
#include <nvtext/wordpiece_tokenize.hpp>

struct TextSubwordTest : public cudf::test::BaseFixture {};

TEST(TextSubwordTest, TokenizedToTensor)
{
  using LCW = cudf::test::lists_column_wrapper<cudf::size_type>;
  // clang-format off
  auto input = LCW({LCW{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20 },
                    LCW{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
                    LCW{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 },
                    LCW{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 }});
  // clang-format on

  auto const all_tokens  = 20;
  auto const new_size    = 15;
  auto const stride      = 10;
  auto const do_truncate = true;

  // all tokens with padding
  auto results = nvtext::tokenized_to_tensor(input, all_tokens, stride, do_truncate);
  EXPECT_EQ(results.nrows_tensor, 4);
  EXPECT_EQ(results.sequence_length, all_tokens);
  // clang-format off
  auto expected_tokens = cudf::test::fixed_width_column_wrapper<uint32_t>(
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
     1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  0,  0,  0,  0,  0,
     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,  0,  0,  0}
  );
  auto expected_attn = cudf::test::fixed_width_column_wrapper<uint32_t>(
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}
  );
  auto expected_metadata = cudf::test::fixed_width_column_wrapper<uint32_t>(
    { 0,0,19,
      1,0,9,
      2,0,14,
      3,0,16 }
  );
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results.tensor_token_ids->view(), expected_tokens);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results.tensor_attention_mask->view(), expected_attn);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results.tensor_metadata->view(), expected_metadata);

  // without truncate should be the same result since all tokens fit within all_tokens
  results = nvtext::tokenized_to_tensor(input, all_tokens, stride, !do_truncate);
  EXPECT_EQ(results.nrows_tensor, 4);
  EXPECT_EQ(results.sequence_length, all_tokens);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results.tensor_token_ids->view(), expected_tokens);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results.tensor_attention_mask->view(), expected_attn);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results.tensor_metadata->view(), expected_metadata);

  // truncate with padding
  results = nvtext::tokenized_to_tensor(input, new_size, stride, do_truncate);
  EXPECT_EQ(results.nrows_tensor, 4);
  EXPECT_EQ(results.sequence_length, new_size);
  // clang-format off
  expected_tokens = cudf::test::fixed_width_column_wrapper<uint32_t>(
    {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  0,  0,  0,  0,  0,
     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
  );
  expected_attn = cudf::test::fixed_width_column_wrapper<uint32_t>(
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
  );
  expected_metadata = cudf::test::fixed_width_column_wrapper<uint32_t>(
    { 0,0,14,
      1,0,9,
      2,0,14,
      3,0,14 }
  );
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results.tensor_token_ids->view(), expected_tokens);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results.tensor_attention_mask->view(), expected_attn);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results.tensor_metadata->view(), expected_metadata);

  // no truncate with stride
  results = nvtext::tokenized_to_tensor(input, new_size, stride, !do_truncate);
  EXPECT_EQ(results.nrows_tensor, 6);
  EXPECT_EQ(results.sequence_length, new_size);
  // clang-format off
  expected_tokens = cudf::test::fixed_width_column_wrapper<uint32_t>(
    { 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
     11, 12, 13, 14, 15, 16, 17, 18, 19, 20,  0,  0,  0,  0,  0,
      1,  2,  3,  4,  5,  6,  7,  8,  9, 10,  0,  0,  0,  0,  0,
      1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
      1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
     11, 12, 13, 14, 15, 16, 17,  0,  0,  0,  0,  0,  0,  0,  0}
  );
  expected_attn = cudf::test::fixed_width_column_wrapper<uint32_t>(
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0}
  );
  expected_metadata = cudf::test::fixed_width_column_wrapper<uint32_t>(
    { 0,0,12, 0,2,9,
      1,0,9,
      2,0,14,
      3,0,12, 3,2,6}
  );
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results.tensor_token_ids->view(), expected_tokens);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results.tensor_attention_mask->view(), expected_attn);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results.tensor_metadata->view(), expected_metadata);

  // max_sequence_length == stride
  results = nvtext::tokenized_to_tensor(input, new_size, new_size, !do_truncate);
  EXPECT_EQ(results.nrows_tensor, 6);
  EXPECT_EQ(results.sequence_length, new_size);
  // clang-format off
  expected_tokens = cudf::test::fixed_width_column_wrapper<uint32_t>(
    { 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
      1,  2,  3,  4,  5,  6,  7,  8,  9, 10,  0,  0,  0,  0,  0,
      1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
      1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
     16, 17,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,}
  );
  expected_attn = cudf::test::fixed_width_column_wrapper<uint32_t>(
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
  );
  expected_metadata = cudf::test::fixed_width_column_wrapper<uint32_t>(
    { 0,0,14, 0,0,4,
      1,0,9,
      2,0,14,
      3,0,14, 3,0,1}
  );
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results.tensor_token_ids->view(), expected_tokens);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results.tensor_attention_mask->view(), expected_attn);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results.tensor_metadata->view(), expected_metadata);
}

TEST(TextSubwordTest, EmptyTokens)
{
  using LCW = cudf::test::lists_column_wrapper<cudf::size_type>;

  auto const seq_length = 10;
  auto results          = nvtext::tokenized_to_tensor(LCW{}, seq_length, seq_length, true);
  EXPECT_EQ(results.nrows_tensor, 0);
  EXPECT_EQ(results.sequence_length, seq_length);
  EXPECT_EQ(results.tensor_token_ids->size(), 0);

  results = nvtext::tokenized_to_tensor(LCW({LCW{}, LCW{}}), seq_length, seq_length, true);
  EXPECT_EQ(results.nrows_tensor, 2);
  EXPECT_EQ(results.sequence_length, seq_length);
  auto expected = cudf::test::fixed_width_column_wrapper<uint32_t>(
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
  auto metadata = cudf::test::fixed_width_column_wrapper<uint32_t>({0, 0, 0, 1, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results.tensor_token_ids->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results.tensor_attention_mask->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(results.tensor_metadata->view(), metadata);

  auto input = LCW({LCW{}, LCW{}}, cudf::test::iterators::all_nulls());
  results    = nvtext::tokenized_to_tensor(input, seq_length, seq_length, true);
  EXPECT_EQ(results.nrows_tensor, 0);
  EXPECT_EQ(results.sequence_length, seq_length);
  EXPECT_EQ(results.tensor_token_ids->size(), 0);
}

TEST(TextSubwordTest, ErrorChecks)
{
  using LCW  = cudf::test::lists_column_wrapper<cudf::size_type>;
  auto input = LCW({LCW{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, LCW{1, 2, 3, 4, 5}});

  EXPECT_THROW(nvtext::tokenized_to_tensor(input, 10, 20, true), std::invalid_argument);
  EXPECT_THROW(
    nvtext::tokenized_to_tensor(input, std::numeric_limits<cudf::size_type>::max(), 20, true),
    std::overflow_error);
}

TEST(TextSubwordTest, WordPiece)
{
  auto vocabulary = cudf::test::strings_column_wrapper(
    {"ate", "brown", "cheese", "dog", "fox", "jumped", "lazy", "quick", "over", "the", "[UNK]"});
  auto vocab = nvtext::load_wordpiece_vocabulary(cudf::strings_column_view(vocabulary));

  auto input = cudf::test::strings_column_wrapper(
    {"the quick brown fox jumped over",
     "the  lazy  brown  dog",
     " ate brown cheese dog fox jumped lazy quick over the [UNK] "});
  auto sv      = cudf::strings_column_view(input);
  auto results = nvtext::wordpiece_tokenize(sv, *vocab);

  using LCW = cudf::test::lists_column_wrapper<cudf::size_type>;
  // clang-format off
  auto expected = LCW({LCW{ 9, 7, 1, 4, 5, 8},
                       LCW{ 9, 6, 1, 3},
                       LCW{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}});
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = nvtext::wordpiece_tokenize(sv, *vocab, 5);
  // clang-format off
  expected = LCW({LCW{ 9, 7, 1, 4, 5},
                  LCW{ 9, 6, 1, 3},
                  LCW{ 0, 1, 2, 3, 4}});
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST(TextSubwordTest, WordPieceWithSubwords)
{
  auto vocabulary =
    cudf::test::strings_column_wrapper({"", "[UNK]", "!", "a", "I", "G", "have", "##P", "##U"});
  auto vocab = nvtext::load_wordpiece_vocabulary(cudf::strings_column_view(vocabulary));

  auto input =
    cudf::test::strings_column_wrapper({"I have a GPU ! ", "do not have a gpu", "no gpu"});
  auto sv      = cudf::strings_column_view(input);
  auto results = nvtext::wordpiece_tokenize(sv, *vocab);

  using LCW = cudf::test::lists_column_wrapper<cudf::size_type>;
  // clang-format off
  auto expected = LCW({LCW{4, 6, 3, 5, 7, 8, 2},
                       LCW{1, 1, 6, 3, 1},
                       LCW{1, 1}});
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  // max is applied to input words and not output tokens
  results = nvtext::wordpiece_tokenize(sv, *vocab, 6);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = nvtext::wordpiece_tokenize(sv, *vocab, 4);
  // clang-format off
  expected = LCW({LCW{4, 6, 3, 5, 7, 8},
                  LCW{1, 1, 6, 3},
                  LCW{1, 1}});
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST(TextSubwordTest, WordPieceSliced)
{
  auto vocabulary = cudf::test::strings_column_wrapper(
    {"ate", "brown", "cheese", "dog", "fox", "jumped", "lazy", "quick", "over", "the", "[UNK]"});
  auto vocab = nvtext::load_wordpiece_vocabulary(cudf::strings_column_view(vocabulary));

  auto input = cudf::test::strings_column_wrapper(
    {" ate the cheese dog quick over  lazy day ",
     "the quick brown fox jumped over",
     "the  lazy  brown  dog",
     " ate brown cheese dog fox jumped lazy quick over the [UNK] ",
     " ate the cheese dog quick over  lazy day "});

  auto sliced  = cudf::slice(input, {1, 4});
  auto sv      = cudf::strings_column_view(sliced.front());
  auto results = nvtext::wordpiece_tokenize(sv, *vocab);

  using LCW = cudf::test::lists_column_wrapper<cudf::size_type>;
  // clang-format off
  auto expected = LCW({LCW{ 9, 7, 1, 4, 5, 8},
                       LCW{ 9, 6, 1, 3},
                       LCW{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}});
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);

  results = nvtext::wordpiece_tokenize(sv, *vocab, 5);
  // clang-format off
  expected = LCW({LCW{ 9, 7, 1, 4, 5},
                  LCW{ 9, 6, 1, 3},
                  LCW{ 0, 1, 2, 3, 4}});
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST(TextSubwordTest, WordPieceEmpty)
{
  auto vocabulary = cudf::test::strings_column_wrapper({""});
  auto vocab      = nvtext::load_wordpiece_vocabulary(cudf::strings_column_view(vocabulary));
  auto input      = cudf::test::strings_column_wrapper();
  auto sv         = cudf::strings_column_view(input);
  auto results    = nvtext::wordpiece_tokenize(sv, *vocab);
  EXPECT_EQ(results->size(), 0);
  results = nvtext::wordpiece_tokenize(sv, *vocab, 10);
  EXPECT_EQ(results->size(), 0);
}

TEST(TextSubwordTest, WordPieceAllNulls)
{
  auto vocabulary = cudf::test::strings_column_wrapper({""});
  auto vocab      = nvtext::load_wordpiece_vocabulary(cudf::strings_column_view(vocabulary));
  auto input      = cudf::test::strings_column_wrapper({"", "", ""}, {false, false, false});
  auto sv         = cudf::strings_column_view(input);
  auto results    = nvtext::wordpiece_tokenize(sv, *vocab);
  using LCW       = cudf::test::lists_column_wrapper<cudf::size_type>;
  auto expected   = LCW({LCW{}, LCW{}, LCW{}}, cudf::test::iterators::all_nulls());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  results = nvtext::wordpiece_tokenize(sv, *vocab, 10);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST(TextSubwordTest, WordPieceNoTokens)
{
  auto vocabulary = cudf::test::strings_column_wrapper({"x"});
  auto vocab      = nvtext::load_wordpiece_vocabulary(cudf::strings_column_view(vocabulary));
  auto input      = cudf::test::strings_column_wrapper({"  ", " www ", "xxxx"});
  auto sv         = cudf::strings_column_view(input);
  auto results    = nvtext::wordpiece_tokenize(sv, *vocab);
  using LCW       = cudf::test::lists_column_wrapper<cudf::size_type>;
  LCW expected({LCW{}, LCW{-1}, LCW{-1}});  // -1 indicates [unk] not found in vocabulary
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
  results = nvtext::wordpiece_tokenize(sv, *vocab, 10);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(*results, expected);
}

TEST(TextSubwordTest, WordPieceErrors)
{
  auto empty = cudf::test::strings_column_wrapper();
  EXPECT_THROW(nvtext::load_wordpiece_vocabulary(cudf::strings_column_view(empty)),
               std::invalid_argument);
  auto nulls = cudf::test::strings_column_wrapper({"", "", ""}, {false, false, false});
  EXPECT_THROW(nvtext::load_wordpiece_vocabulary(cudf::strings_column_view(nulls)),
               std::invalid_argument);

  auto vocabulary = cudf::test::strings_column_wrapper({"x"});
  auto vocab      = nvtext::load_wordpiece_vocabulary(cudf::strings_column_view(vocabulary));
  auto input      = cudf::test::strings_column_wrapper({"  "});
  EXPECT_THROW(nvtext::wordpiece_tokenize(cudf::strings_column_view(input), *vocab, -1),
               std::invalid_argument);
}
