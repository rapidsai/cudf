/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <cudf_test/default_stream.hpp>

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
  auto vocab = nvtext::load_vocabulary_file(hash_file, cudf::test::get_default_stream());

  uint32_t max_sequence_length = 16;
  uint32_t stride              = 16;

  auto result = nvtext::subword_tokenize(cudf::strings_column_view{strings},
                                         *vocab,
                                         max_sequence_length,
                                         stride,
                                         true,   // do_lower_case
                                         false,  // do_truncate
                                         cudf::test::get_default_stream());

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
