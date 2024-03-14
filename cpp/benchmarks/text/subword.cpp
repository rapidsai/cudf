/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <benchmarks/fixture/benchmark_fixture.hpp>
#include <benchmarks/synchronization/synchronization.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/file_utilities.hpp>

#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/subword_tokenize.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

static std::string create_hash_vocab_file()
{
  static temp_directory const subword_tmpdir{"cudf_gbench"};
  auto dir_template     = subword_tmpdir.path();
  std::string hash_file = dir_template + "/hash_vocab.txt";
  // create a fake hashed vocab text file for this test
  // this only works with words in the strings in the benchmark code below
  std::vector<std::pair<int, int>> coefficients(23, {65559, 0});
  std::ofstream outfile(hash_file, std::ofstream::out);
  outfile << "1\n0\n" << coefficients.size() << "\n";
  for (auto c : coefficients)
    outfile << c.first << " " << c.second << "\n";
  std::vector<uint64_t> hash_table(23, 0);
  outfile << hash_table.size() << "\n";
  hash_table[0]  = 3015668L;
  hash_table[1]  = 6205475701751155871L;
  hash_table[5]  = 6358029;
  hash_table[16] = 451412625363L;
  hash_table[20] = 6206321707968235495L;
  for (auto h : hash_table)
    outfile << h << "\n";
  outfile << "100\n101\n102\n\n";
  return hash_file;
}

static void BM_subword_tokenizer(benchmark::State& state)
{
  auto const nrows = static_cast<cudf::size_type>(state.range(0));
  std::vector<char const*> h_strings(nrows, "This is a test ");
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  static std::string hash_file = create_hash_vocab_file();
  std::vector<uint32_t> offsets{14};
  uint32_t max_sequence_length = 64;
  uint32_t stride              = 48;
  uint32_t do_truncate         = 0;
  uint32_t do_lower            = 1;
  //
  auto vocab = nvtext::load_vocabulary_file(hash_file);
  for (auto _ : state) {
    cuda_event_timer raii(state, true);
    auto result = nvtext::subword_tokenize(cudf::strings_column_view{strings},
                                           *vocab,
                                           max_sequence_length,
                                           stride,
                                           do_lower,
                                           do_truncate);
  }
}

class Subword : public cudf::benchmark {};

#define SUBWORD_BM_BENCHMARK_DEFINE(name)                                                        \
  BENCHMARK_DEFINE_F(Subword, name)(::benchmark::State & state) { BM_subword_tokenizer(state); } \
  BENCHMARK_REGISTER_F(Subword, name)                                                            \
    ->RangeMultiplier(2)                                                                         \
    ->Range(1 << 10, 1 << 17)                                                                    \
    ->UseManualTime()                                                                            \
    ->Unit(benchmark::kMillisecond);

SUBWORD_BM_BENCHMARK_DEFINE(BM_subword_tokenizer);

// BENCHMARK_MAIN();
