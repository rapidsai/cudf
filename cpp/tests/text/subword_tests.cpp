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
#include <cudf/scalar/scalar.hpp>
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
  cudf::test::strings_column_wrapper sentences{"This is a test."};
  // create a fake hashed vocab text file for this test
  // this only works with words in the sentences above
  std::string hash_file = temp_env->get_temp_filepath("fake_hashed_vocab.txt");
  {
    std::vector<std::pair<int, int>> coefficients(23, {65559, 0});
    std::ofstream outfile(hash_file, std::ofstream::out);
    outfile << "1\n0\n" << coefficients.size() << "\n";
    for (auto c : coefficients) outfile << c.first << " " << c.second << "\n";
    std::vector<uint64_t> hash_table(23, 0);
    outfile << hash_table.size() << "\n";
    hash_table[0]  = 3015668L;
    hash_table[1]  = 6205475701751155871L;
    hash_table[5]  = 6358029;
    hash_table[16] = 451412625363L;
    hash_table[20] = 6206321707968235495L;
    for (auto h : hash_table) outfile << h << "\n";
    outfile << "100\n101\n102\n\n";
  }

  uint32_t max_sequence_length = 64;
  uint32_t stride              = 48;
  uint32_t do_truncate         = 0;
  uint32_t do_lower            = 1;

  auto result = nvtext::subword_tokenize(cudf::strings_column_view{sentences},
                                         hash_file,
                                         max_sequence_length,
                                         stride,
                                         do_lower,
                                         do_truncate,
                                         MAX_NUM_SENTENCES,
                                         MAX_NUM_CHARS,
                                         MAX_ROWS_TENSOR);

  std::vector<uint32_t> host_final_tensor;
  std::vector<uint32_t> host_attn_mask;
  std::vector<uint32_t> host_metadata;
  host_final_tensor.resize(result->nrows_tensor * max_sequence_length);
  host_attn_mask.resize(result->nrows_tensor * max_sequence_length);
  host_metadata.resize(result->nrows_tensor * 3);

  cudaMemcpy(host_final_tensor.data(),
             result->device_tensor_tokenIDS,
             result->nrows_tensor * max_sequence_length * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(host_attn_mask.data(),
             result->device_attention_mask,
             result->nrows_tensor * max_sequence_length * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(host_metadata.data(),
             result->device_tensor_metadata,
             result->nrows_tensor * 3 * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);

  std::vector<uint32_t> expected_tensor;
  std::vector<uint32_t> expected_attn_mask;
  std::vector<uint32_t> expected_metadata;
  expected_tensor = {2023, 2003, 1037, 3231, 1012, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0,    0,    0,    0,    0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0,    0,    0,    0,    0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                     0,    0,    0,    0,    0,    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  EXPECT_EQ(expected_tensor, host_final_tensor);
  expected_attn_mask = {1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  EXPECT_EQ(expected_attn_mask, host_attn_mask);
  expected_metadata = {0, 0, 4};
  EXPECT_EQ(expected_metadata, host_metadata);

  // not sure how these are freed by the caller
  cudaFree(result->device_attention_mask);
  cudaFree(result->device_tensor_metadata);
  cudaFree(result->device_tensor_tokenIDS);
}
