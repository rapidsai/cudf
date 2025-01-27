/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/subword_tokenize.hpp>

#include <fstream>
#include <vector>

// Global environment for temporary files
auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

struct TextSubwordTest : public cudf::test::BaseFixture {};

// Create a fake hashed vocab text file for the tests in this source file.
// The vocab only includes the following words:
//  'this', 'is', 'a', 'test', 'tést'
// The period '.' character also has a token id.
void create_hashed_vocab(std::string const& hash_file)
{
  std::vector<std::pair<int, int>> coefficients(23, {65559, 0});
  std::ofstream outfile(hash_file, std::ofstream::out);
  outfile << "1\n0\n" << coefficients.size() << "\n";
  for (auto c : coefficients)
    outfile << c.first << " " << c.second << "\n";
  std::vector<uint64_t> hash_table(23, 0);
  outfile << hash_table.size() << "\n";
  hash_table[0]  = 3015668L;              // based on values
  hash_table[1]  = 6205475701751155871L;  // from the
  hash_table[5]  = 6358029;               // bert_hash_table.txt
  hash_table[16] = 451412625363L;         // file for the test
  hash_table[20] = 6206321707968235495L;  // words above
  for (auto h : hash_table)
    outfile << h << "\n";
  outfile << "100\n101\n102\n\n";
}

TEST(TextSubwordTest, Tokenize)
{
  uint32_t nrows = 100;
  std::vector<char const*> h_strings(nrows, "This is a test. A test this is.");
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  std::string hash_file = temp_env->get_temp_filepath("hashed_vocab.txt");
  create_hashed_vocab(hash_file);
  auto vocab = nvtext::load_vocabulary_file(hash_file);

  uint32_t max_sequence_length = 16;
  uint32_t stride              = 16;

  auto result = nvtext::subword_tokenize(cudf::strings_column_view{strings},
                                         *vocab,
                                         max_sequence_length,
                                         stride,
                                         true,    // do_lower_case
                                         false);  // do_truncate

  EXPECT_EQ(nrows, result.nrows_tensor);

  {
    std::vector<uint32_t> base_data(
      {2023, 2003, 1037, 3231, 1012, 1037, 3231, 2023, 2003, 1012, 0, 0, 0, 0, 0, 0});
    std::vector<uint32_t> h_expected;
    for (uint32_t idx = 0; idx < nrows; ++idx)
      h_expected.insert(h_expected.end(), base_data.begin(), base_data.end());
    cudf::test::fixed_width_column_wrapper<uint32_t> expected(h_expected.begin(), h_expected.end());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tensor_token_ids->view(), expected);
  }

  {
    std::vector<uint32_t> base_data({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0});
    std::vector<uint32_t> h_expected;
    for (uint32_t idx = 0; idx < nrows; ++idx)
      h_expected.insert(h_expected.end(), base_data.begin(), base_data.end());
    cudf::test::fixed_width_column_wrapper<uint32_t> expected(h_expected.begin(), h_expected.end());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tensor_attention_mask->view(), expected);
  }

  {
    std::vector<uint32_t> h_expected;
    for (uint32_t idx = 0; idx < nrows; ++idx) {
      // 0,0,9,1,0,9,2,0,9,3,0,9,4,0,9,5,0,9,6,0,9,7,0,9,8,0,9,9,0,9,...
      h_expected.push_back(idx);
      h_expected.push_back(0);
      h_expected.push_back(9);
    }
    cudf::test::fixed_width_column_wrapper<uint32_t> expected(h_expected.begin(), h_expected.end());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tensor_metadata->view(), expected);
  }
}

TEST(TextSubwordTest, TokenizeMultiRow)
{
  std::vector<char const*> h_strings{"This is a test.", "This is a test. This is a tést."};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  std::string hash_file = temp_env->get_temp_filepath("hashed_vocab.txt");
  create_hashed_vocab(hash_file);
  auto vocab = nvtext::load_vocabulary_file(hash_file);

  uint32_t max_sequence_length = 8;
  uint32_t stride              = 6;

  auto result = nvtext::subword_tokenize(cudf::strings_column_view{strings},
                                         *vocab,
                                         max_sequence_length,
                                         stride,
                                         true,    // do_lower_case
                                         false);  // do_truncate

  EXPECT_EQ(uint32_t{3}, result.nrows_tensor);
  cudf::test::fixed_width_column_wrapper<uint32_t> expected_tokens(
    {2023, 2003, 1037, 3231, 1012, 0,    0,    0,    2023, 2003, 1037, 3231,
     1012, 2023, 2003, 1037, 2003, 1037, 3231, 1012, 0,    0,    0,    0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tensor_token_ids->view(), expected_tokens);
  cudf::test::fixed_width_column_wrapper<uint32_t> expected_attn(
    {1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tensor_attention_mask->view(), expected_attn);
  cudf::test::fixed_width_column_wrapper<uint32_t> expected_metadata({0, 0, 4, 1, 0, 6, 1, 1, 3});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tensor_metadata->view(), expected_metadata);
}

TEST(TextSubwordTest, TokenizeWithEmptyRow)
{
  std::string hash_file = temp_env->get_temp_filepath("hashed_vocab.txt");
  create_hashed_vocab(hash_file);
  auto vocab = nvtext::load_vocabulary_file(hash_file);

  cudf::test::strings_column_wrapper strings{
    "This is a test.", "", "This is a test. This is a tést."};
  auto input = cudf::strings_column_view{strings};

  uint32_t const max_seq = 8;
  uint32_t const stride  = 6;
  bool const lower       = true;
  bool const truncate    = false;

  auto result = nvtext::subword_tokenize(input, *vocab, max_seq, stride, lower, truncate);

  EXPECT_EQ(uint32_t{4}, result.nrows_tensor);

  // clang-format off
  auto expected_tokens = cudf::test::fixed_width_column_wrapper<uint32_t>(
    {2023, 2003, 1037, 3231, 1012,   0,    0,    0,
        0,    0,    0,    0,    0,   0,    0,    0,
     2023, 2003, 1037, 3231, 1012, 2023, 2003, 1037,   // this one
     2003, 1037, 3231, 1012,    0,    0,    0,    0}); // continues here
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tensor_token_ids->view(), expected_tokens);
  // clang-format off
  auto expected_attn = cudf::test::fixed_width_column_wrapper<uint32_t>(
     {1, 1, 1, 1, 1, 0, 0, 0,
      0, 0, 0, 0, 0, 0, 0, 0,
      1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 0, 0, 0, 0});
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tensor_attention_mask->view(), expected_attn);
  // clang-format off
  auto expected_metadata = cudf::test::fixed_width_column_wrapper<uint32_t>(
    {0,0,4, 1,0,0, 2,0,6, 2,1,3}); // note that the 3rd element has 2 tensors
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tensor_metadata->view(), expected_metadata);
}

TEST(TextSubwordTest, TokenizeMaxEqualsTokens)
{
  cudf::test::strings_column_wrapper strings({"This is a test."});
  std::string hash_file = temp_env->get_temp_filepath("hashed_vocab.txt");
  create_hashed_vocab(hash_file);
  auto vocab = nvtext::load_vocabulary_file(hash_file);

  uint32_t max_sequence_length = 5;  // five tokens in strings;
  uint32_t stride              = 5;  // this should not effect the result

  auto result = nvtext::subword_tokenize(cudf::strings_column_view{strings},
                                         *vocab,
                                         max_sequence_length,
                                         stride,
                                         true,    // do_lower_case
                                         false);  // do_truncate

  EXPECT_EQ(uint32_t{1}, result.nrows_tensor);
  cudf::test::fixed_width_column_wrapper<uint32_t> expected_tokens({2023, 2003, 1037, 3231, 1012});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tensor_token_ids->view(), expected_tokens);
  cudf::test::fixed_width_column_wrapper<uint32_t> expected_attn({1, 1, 1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tensor_attention_mask->view(), expected_attn);
  cudf::test::fixed_width_column_wrapper<uint32_t> expected_metadata({0, 0, 4});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tensor_metadata->view(), expected_metadata);
}

TEST(TextSubwordTest, ParameterErrors)
{
  std::vector<char const*> h_strings{"This is a test.", "This is a test. This is a tést.", "", ""};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  std::string hash_file = temp_env->get_temp_filepath("hashed_vocab.txt");
  create_hashed_vocab(hash_file);
  auto vocab = nvtext::load_vocabulary_file(hash_file);

  EXPECT_THROW(nvtext::subword_tokenize(cudf::strings_column_view{strings},
                                        *vocab,
                                        12,     // max_sequence_length
                                        13,     // stride <= max_sequence_length
                                        true,   // do_lower_case
                                        true),  // do_truncate
               cudf::logic_error);

  EXPECT_THROW(nvtext::subword_tokenize(cudf::strings_column_view{strings},
                                        *vocab,
                                        858993459,
                                        5,
                                        true,   // do_lower_case
                                        true),  // do_truncate
               std::overflow_error);
}

TEST(TextSubwordTest, EmptyStrings)
{
  cudf::test::strings_column_wrapper strings;
  std::string hash_file = temp_env->get_temp_filepath("hashed_vocab.txt");
  create_hashed_vocab(hash_file);
  auto vocab  = nvtext::load_vocabulary_file(hash_file);
  auto result = nvtext::subword_tokenize(cudf::strings_column_view{strings},
                                         *vocab,
                                         16,
                                         16,
                                         true,    // do_lower_case
                                         false);  // do_truncate
  EXPECT_EQ(uint32_t{0}, result.nrows_tensor);
  EXPECT_EQ(0, result.tensor_token_ids->size());
  EXPECT_EQ(0, result.tensor_attention_mask->size());
  EXPECT_EQ(0, result.tensor_metadata->size());
}

TEST(TextSubwordTest, AllNullStrings)
{
  cudf::test::strings_column_wrapper strings({"", "", ""}, {false, false, false});
  std::string hash_file = temp_env->get_temp_filepath("hashed_vocab.txt");
  create_hashed_vocab(hash_file);
  auto vocab  = nvtext::load_vocabulary_file(hash_file);
  auto result = nvtext::subword_tokenize(cudf::strings_column_view{strings},
                                         *vocab,
                                         16,
                                         16,
                                         true,    // do_lower_case
                                         false);  // do_truncate
  EXPECT_EQ(uint32_t{0}, result.nrows_tensor);
  EXPECT_EQ(0, result.tensor_token_ids->size());
  EXPECT_EQ(0, result.tensor_attention_mask->size());
  EXPECT_EQ(0, result.tensor_metadata->size());
}

TEST(TextSubwordTest, NoTokens)
{
  std::string hash_file = temp_env->get_temp_filepath("hashed_vocab.txt");
  create_hashed_vocab(hash_file);
  auto vocab = nvtext::load_vocabulary_file(hash_file);

  cudf::test::strings_column_wrapper strings({"  ", "\n\r", "\t"});
  auto input = cudf::strings_column_view{strings};

  uint32_t const max_seq = 16;
  uint32_t const stride  = 16;
  bool const lower       = true;
  bool const truncate    = true;

  auto result = nvtext::subword_tokenize(input, *vocab, max_seq, stride, lower, truncate);

  std::vector<uint32_t> zeros(max_seq * input.size(), 0);

  EXPECT_EQ(static_cast<uint32_t>(input.size()), result.nrows_tensor);

  auto expected = cudf::test::fixed_width_column_wrapper<uint32_t>(zeros.begin(), zeros.end());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tensor_token_ids->view(), expected);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tensor_attention_mask->view(), expected);
  auto expected_metadata =
    cudf::test::fixed_width_column_wrapper<uint32_t>({0, 0, 0, 1, 0, 0, 2, 0, 0});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tensor_metadata->view(), expected_metadata);
}

TEST(TextSubwordTest, TokenizeFromVocabStruct)
{
  std::string hash_file = temp_env->get_temp_filepath("hashed_vocab.txt");
  create_hashed_vocab(hash_file);

  std::vector<char const*> h_strings{"This is a test.", "This is a test. This is a tést."};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  auto vocab  = nvtext::load_vocabulary_file(hash_file);
  auto result = nvtext::subword_tokenize(cudf::strings_column_view{strings},
                                         *vocab,
                                         8,
                                         6,
                                         true,   // do_lower_case
                                         true);  // do_truncate

  EXPECT_EQ(uint32_t{2}, result.nrows_tensor);
  cudf::test::fixed_width_column_wrapper<uint32_t> expected_tokens(
    {2023, 2003, 1037, 3231, 1012, 0, 0, 0, 2023, 2003, 1037, 3231, 1012, 2023, 2003, 1037});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tensor_token_ids->view(), expected_tokens);
  cudf::test::fixed_width_column_wrapper<uint32_t> expected_attn(
    {1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tensor_attention_mask->view(), expected_attn);
  cudf::test::fixed_width_column_wrapper<uint32_t> expected_metadata({0, 0, 4, 1, 0, 7});
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tensor_metadata->view(), expected_metadata);
}

TEST(TextSubwordTest, LoadVocabFileErrors)
{
  std::vector<char const*> h_strings{"This is a test.", "This is a test. This is a tést."};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  std::string hash_file = temp_env->get_temp_filepath("nothing.txt");
  EXPECT_THROW(nvtext::load_vocabulary_file(hash_file), cudf::logic_error);
}

// This includes the words above and 7 special tokens:
//  [BOS] [EOS] [UNK] [SEP] [PAD] [CLS] [MASK]
// The data here was generated by the utility:
//   cudf.utils.hash_vocab_utils.hash_vocab()
void create_special_tokens_hashed_vocab(std::string const& hash_file)
{
  std::ofstream outfile(hash_file, std::ofstream::out);
  outfile << "26899\n27424\n3\n";
  outfile << "1416131940466419714 0\n";
  outfile << "313740585393291779 2\n";
  outfile << "17006415773850330120 5\n";
  outfile << "13\n";
  outfile << "5903884228619468800\n";
  outfile << "6205475701751152650\n";
  outfile << "16285378285009240068\n";
  outfile << "5162333542489915397\n";
  outfile << "6064762127302393859\n";
  outfile << "6173800107753209857\n";
  outfile << "5322083323972878342\n";
  outfile << "6242701866907861003\n";
  outfile << "451412623368\n";
  outfile << "3014668\n";
  outfile << "5214737420442796034\n";
  outfile << "6206321707968233479\n";
  outfile << "6357001\n";
  outfile << "1\n2\n3\n\n";
}

TEST(TextSubwordTest, TokenizeWithSpecialTokens)
{
  std::string hash_file = temp_env->get_temp_filepath("hashed_vocab.txt");
  create_special_tokens_hashed_vocab(hash_file);

  // clang-format off
  std::vector<const char*> h_strings{
    "[BOS]This is a tést.[eos]",
    "[CLS]A test[SEP]this is.",
    "[PAD] [A][MASK]",
    "test this [CL",
    "S] is a ."};
  // clang-format on
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  auto vocab  = nvtext::load_vocabulary_file(hash_file);
  auto result = nvtext::subword_tokenize(cudf::strings_column_view{strings},
                                         *vocab,
                                         8,
                                         6,
                                         true,   // do_lower_case
                                         true);  // do_truncate

  EXPECT_EQ(static_cast<uint32_t>(h_strings.size()), result.nrows_tensor);
  // clang-format off
  cudf::test::fixed_width_column_wrapper<uint32_t> expected_tokens(
    { 5, 7,  8, 9, 10, 12,  6, 0,
      2, 9, 10, 3,  7,  8, 12, 0,
      0, 1,  9, 1,  4,  0,  0, 0,
     10, 7,  1, 1,  0,  0,  0, 0,
      1, 1,  8, 9, 12,  0,  0, 0});
  cudf::test::fixed_width_column_wrapper<uint32_t> expected_attn(
    {1, 1, 1, 1, 1, 1, 1, 0,
     1, 1, 1, 1, 1, 1, 1, 0,
     1, 1, 1, 1, 1, 0, 0, 0,
     1, 1, 1, 1, 0, 0, 0, 0,
     1, 1, 1, 1, 1, 0, 0, 0});
  cudf::test::fixed_width_column_wrapper<uint32_t> expected_metadata(
    {0, 0, 6,
     1, 0, 6,
     2, 0, 4,
     3, 0, 3,
     4, 0, 4});
  // clang-format on
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tensor_token_ids->view(), expected_tokens);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tensor_attention_mask->view(), expected_attn);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tensor_metadata->view(), expected_metadata);
}

TEST(TextSubwordTest, ZeroHashBinCoefficient)
{
  std::string hash_file = temp_env->get_temp_filepath("hashed_vocab.txt");
  {
    std::ofstream outfile(hash_file, std::ofstream::out);
    outfile << "26899\n27424\n2\n";
    outfile << "6321733446031528966 0\n0 0\n9\n";  // zeroes are here
    outfile << "6206321707968233475\n3014663\n6205475701751152646\n";
    outfile << "451412623364\n5214737420442796033\n6173800107753209856\n";
    outfile << "0\n6356997\n6064762127302393858\n";
    outfile << "0\n1\n2\n";
  }

  std::vector<char const*> h_strings{".zzzz"};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  auto vocab  = nvtext::load_vocabulary_file(hash_file);
  auto result = nvtext::subword_tokenize(cudf::strings_column_view{strings},
                                         *vocab,
                                         8,
                                         8,
                                         true,   // do_lower_case
                                         true);  // do_truncate

  // clang-format off
  cudf::test::fixed_width_column_wrapper<uint32_t> expected_tokens({7, 0, 0, 0, 0, 0, 0, 0});
  cudf::test::fixed_width_column_wrapper<uint32_t> expected_attn(  {1, 1, 0, 0, 0, 0, 0, 0});
  cudf::test::fixed_width_column_wrapper<uint32_t> expected_metadata({0, 0, 1});
  // clang-format on

  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tensor_token_ids->view(), expected_tokens);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tensor_attention_mask->view(), expected_attn);
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tensor_metadata->view(), expected_metadata);
}
