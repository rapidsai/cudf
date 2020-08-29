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

#include <benchmark/benchmark.h>
#include <cudf/strings/strings_column_view.hpp>
#include <nvtext/subword_tokenize.hpp>

#include <tests/utilities/column_utilities.hpp>
#include <tests/utilities/column_wrapper.hpp>

#include <fstream>
#include <iostream>
#include <vector>

#define MAX_ROWS_TENSOR 300

static std::string create_hash_vocab_file()
{
  std::string dir_template("/tmp");
  if (const char* env_p = std::getenv("WORKSPACE")) dir_template = env_p;
  std::string hash_file = dir_template + "/hash_vocab.txt";
  // create a fake hashed vocab text file for this test
  // this only works with words in the strings in the benchmark code below
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
  return hash_file;
}

static void BM_cuda_tokenizer_cudf(benchmark::State& state)
{
  uint32_t nrows = 1000;
  std::vector<const char*> h_strings(nrows, "This is a test ");
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  std::string hash_file = create_hash_vocab_file();
  std::vector<uint32_t> offsets{14};
  uint32_t max_sequence_length = 64;
  uint32_t stride              = 48;
  uint32_t do_truncate         = 0;
  uint32_t do_lower            = 1;
  //
  auto vocab = nvtext::load_vocabulary_file(hash_file);
  for (auto _ : state) {
    auto result = nvtext::subword_tokenize(cudf::strings_column_view{strings},
                                           vocab,
                                           max_sequence_length,
                                           stride,
                                           do_lower,
                                           do_truncate,
                                           MAX_ROWS_TENSOR);
  }
}
BENCHMARK(BM_cuda_tokenizer_cudf);

BENCHMARK_MAIN();
