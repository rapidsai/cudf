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

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/file_utilities.hpp>

#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/subword_tokenize.hpp>

#include <nvbench/nvbench.cuh>

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

static void bench_subword_tokenizer(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));

  std::vector<char const*> h_strings(num_rows, "This is a test ");
  cudf::test::strings_column_wrapper strings(h_strings.begin(), h_strings.end());
  static std::string hash_file = create_hash_vocab_file();
  std::vector<uint32_t> offsets{14};
  uint32_t max_sequence = 64;
  uint32_t stride       = 48;
  uint32_t do_truncate  = 0;
  uint32_t do_lower     = 1;

  auto input = cudf::strings_column_view{strings};

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  auto chars_size = input.chars_size(cudf::get_default_stream());
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);
  state.add_global_memory_writes<nvbench::int32_t>(num_rows * max_sequence);

  auto vocab = nvtext::load_vocabulary_file(hash_file);
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto result =
      nvtext::subword_tokenize(input, *vocab, max_sequence, stride, do_lower, do_truncate);
  });
}

NVBENCH_BENCH(bench_subword_tokenizer)
  .set_name("subword_tokenize")
  .add_int64_axis("num_rows", {32768, 262144, 2097152});
