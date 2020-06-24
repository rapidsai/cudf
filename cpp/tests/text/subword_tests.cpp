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

#define MAX_NUM_SENTENCES 101
#define MAX_NUM_CHARS 150000
#define MAX_ROWS_TENSOR 300

// Global environment for temporary files
auto const temp_env = static_cast<cudf::test::TempDirTestEnvironment*>(
  ::testing::AddGlobalTestEnvironment(new cudf::test::TempDirTestEnvironment));

struct TextSubwordTest : public cudf::test::BaseFixture {
};

TEST(TextSubwordTest, Tokenize)
{
  uint32_t nrows = MAX_NUM_SENTENCES - 1;
  std::vector<const char*> h_strings(nrows, "This is a test. A test this is.");
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  // create a fake hashed vocab text file for this test
  // this only works with words in the strings above
  std::string hash_file = temp_env->get_temp_filepath("hashed_vocab.txt");
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

  uint32_t max_sequence_length = 16;
  uint32_t stride              = 16;  // no repeated tokens
  uint32_t do_truncate         = 0;
  uint32_t do_lower            = 1;

  auto result = nvtext::subword_tokenize(cudf::strings_column_view{strings},
                                         hash_file,
                                         max_sequence_length,
                                         stride,
                                         do_lower,
                                         do_truncate,
                                         MAX_NUM_SENTENCES,
                                         MAX_NUM_CHARS,
                                         MAX_ROWS_TENSOR);

  EXPECT_EQ(nrows, result.nrows_tensor);

  {
    std::vector<uint32_t> base_data(
      {2023, 2003, 1037, 3231, 1012, 1037, 3231, 2023, 2003, 1012, 0, 0, 0, 0, 0, 0});
    std::vector<uint32_t> h_expected;
    for (auto idx = 0; idx < nrows; ++idx)
      h_expected.insert(h_expected.end(), base_data.begin(), base_data.end());
    cudf::test::fixed_width_column_wrapper<uint32_t> expected(h_expected.begin(), h_expected.end());
    cudf::test::expect_columns_equal(result.tensor_token_ids->view(), expected);
  }

  {
    std::vector<uint32_t> base_data({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0});
    std::vector<uint32_t> h_expected;
    for (auto idx = 0; idx < nrows; ++idx)
      h_expected.insert(h_expected.end(), base_data.begin(), base_data.end());
    cudf::test::fixed_width_column_wrapper<uint32_t> expected(h_expected.begin(), h_expected.end());
    cudf::test::expect_columns_equal(result.tensor_attention_mask->view(), expected);
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
    cudf::test::expect_columns_equal(result.tensor_metadata->view(), expected);
  }
}
