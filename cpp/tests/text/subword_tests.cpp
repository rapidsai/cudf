/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/strings/strings_column_view.hpp>

#include <tests/utilities/base_fixture.hpp>
#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

#include <nvtext/subword_tokenize.hpp>

#include <fstream>
#include <iostream>
#include <vector>

#define MAX_ROWS_TENSOR 300

// Global environment for temporary files
auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

struct TextSubwordTest : public cudf::test::BaseFixture {
};

// Create a fake hashed vocab text file for the tests in this source file.
// The vocab only includes the following words:
//  'this', 'is', 'a', 'test', 'tést'
// The period '.' character is also supported.
void create_hashed_vocab(std::string const& hash_file)
{
  std::vector<std::pair<int, int>> coefficients(23, {65559, 0});
  std::ofstream outfile(hash_file, std::ofstream::out);
  outfile << "1\n0\n" << coefficients.size() << "\n";
  for (auto c : coefficients) outfile << c.first << " " << c.second << "\n";
  std::vector<uint64_t> hash_table(23, 0);
  outfile << hash_table.size() << "\n";
  hash_table[0]  = 3015668L;              // based on values
  hash_table[1]  = 6205475701751155871L;  // from the
  hash_table[5]  = 6358029;               // bert_hash_table.txt
  hash_table[16] = 451412625363L;         // file for the test
  hash_table[20] = 6206321707968235495L;  // words above
  for (auto h : hash_table) outfile << h << "\n";
  outfile << "100\n101\n102\n\n";
}

TEST(TextSubwordTest, Tokenize)
{
  uint32_t nrows = 100;
  std::vector<const char*> h_strings(nrows, "This is a test. A test this is.");
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  std::string hash_file = temp_env->get_temp_filepath("hashed_vocab.txt");
  create_hashed_vocab(hash_file);

  uint32_t max_sequence_length = 16;
  uint32_t stride              = 16;

  auto result = nvtext::subword_tokenize(cudf::strings_column_view{strings},
                                         hash_file,
                                         max_sequence_length,
                                         stride,
                                         true,   // do_lower_case
                                         false,  // do_truncate
                                         MAX_ROWS_TENSOR);

  EXPECT_EQ(nrows, result.nrows_tensor);

  {
    std::vector<uint32_t> base_data(
      {2023, 2003, 1037, 3231, 1012, 1037, 3231, 2023, 2003, 1012, 0, 0, 0, 0, 0, 0});
    std::vector<uint32_t> h_expected;
    for (auto idx = 0; idx < nrows; ++idx)
      h_expected.insert(h_expected.end(), base_data.begin(), base_data.end());
    cudf::test::fixed_width_column_wrapper<uint32_t> expected(h_expected.begin(), h_expected.end());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tensor_token_ids->view(), expected);
  }

  {
    std::vector<uint32_t> base_data({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0});
    std::vector<uint32_t> h_expected;
    for (auto idx = 0; idx < nrows; ++idx)
      h_expected.insert(h_expected.end(), base_data.begin(), base_data.end());
    cudf::test::fixed_width_column_wrapper<uint32_t> expected(h_expected.begin(), h_expected.end());
    CUDF_TEST_EXPECT_COLUMNS_EQUAL(result.tensor_attention_mask->view(), expected);
  }

  {
    std::vector<uint32_t> h_expected;
    for (auto idx = 0; idx < nrows; ++idx) {
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
  std::vector<const char*> h_strings{"This is a test.", "This is a test. This is a tést."};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  std::string hash_file = temp_env->get_temp_filepath("hashed_vocab.txt");
  create_hashed_vocab(hash_file);

  uint32_t max_sequence_length = 8;
  uint32_t stride              = 6;

  auto result = nvtext::subword_tokenize(cudf::strings_column_view{strings},
                                         hash_file,
                                         max_sequence_length,
                                         stride,
                                         true,   // do_lower_case
                                         false,  // do_truncate
                                         MAX_ROWS_TENSOR);

  EXPECT_EQ(3, result.nrows_tensor);
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

TEST(TextSubwordTest, ParameterErrors)
{
  std::vector<const char*> h_strings{"This is a test.", "This is a test. This is a tést."};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  std::string hash_file = temp_env->get_temp_filepath("hashed_vocab.txt");
  create_hashed_vocab(hash_file);
  EXPECT_THROW(nvtext::subword_tokenize(cudf::strings_column_view{strings},
                                        hash_file,
                                        12,    // max_sequence_length
                                        13,    // stride <= max_sequence_length
                                        true,  // do_lower_case
                                        true,  // do_truncate
                                        MAX_ROWS_TENSOR),
               cudf::logic_error);

  EXPECT_THROW(nvtext::subword_tokenize(cudf::strings_column_view{strings},
                                        hash_file,
                                        5,
                                        5,
                                        true,  // do_lower_case
                                        true,  // do_truncate
                                        858993459),
               cudf::logic_error);
}

TEST(TextSubwordTest, EmptyStrings)
{
  cudf::test::strings_column_wrapper strings;
  std::string hash_file = temp_env->get_temp_filepath("hashed_vocab.txt");
  create_hashed_vocab(hash_file);
  auto result = nvtext::subword_tokenize(cudf::strings_column_view{strings},
                                         hash_file,
                                         16,
                                         16,
                                         true,   // do_lower_case
                                         false,  // do_truncate
                                         MAX_ROWS_TENSOR);
  EXPECT_EQ(0, result.nrows_tensor);
  EXPECT_EQ(0, result.tensor_token_ids->size());
  EXPECT_EQ(0, result.tensor_attention_mask->size());
  EXPECT_EQ(0, result.tensor_metadata->size());
}

TEST(TextSubwordTest, AllNullStrings)
{
  cudf::test::strings_column_wrapper strings({"", "", ""}, {0, 0, 0});
  std::string hash_file = temp_env->get_temp_filepath("hashed_vocab.txt");
  create_hashed_vocab(hash_file);
  auto result = nvtext::subword_tokenize(cudf::strings_column_view{strings},
                                         hash_file,
                                         16,
                                         16,
                                         true,   // do_lower_case
                                         false,  // do_truncate
                                         MAX_ROWS_TENSOR);
  EXPECT_EQ(0, result.nrows_tensor);
  EXPECT_EQ(0, result.tensor_token_ids->size());
  EXPECT_EQ(0, result.tensor_attention_mask->size());
  EXPECT_EQ(0, result.tensor_metadata->size());
}

TEST(TextSubwordTest, TokenizeFromVocabStruct)
{
  std::string hash_file = temp_env->get_temp_filepath("hashed_vocab.txt");
  create_hashed_vocab(hash_file);

  std::vector<const char*> h_strings{"This is a test.", "This is a test. This is a tést."};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  auto vocab  = nvtext::load_vocabulary_file(hash_file);
  auto result = nvtext::subword_tokenize(cudf::strings_column_view{strings},
                                         vocab,
                                         8,
                                         6,
                                         true,  // do_lower_case
                                         true,  // do_truncate
                                         MAX_ROWS_TENSOR);

  EXPECT_EQ(2, result.nrows_tensor);
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
  std::vector<const char*> h_strings{"This is a test.", "This is a test. This is a tést."};
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  std::string hash_file = temp_env->get_temp_filepath("nothing.txt");
  EXPECT_THROW(nvtext::load_vocabulary_file(hash_file), cudf::logic_error);
}
