/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>

#include <cudf/io/csv.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvtext/bpe_tokenize.hpp>
#include <nvtext/subword_tokenize.hpp>

#include <nvbench/nvbench.cuh>

static cudf::io::table_with_metadata read_csv(std::string const& file_path)
{
  auto source_info = cudf::io::source_info(file_path);
  auto builder     = cudf::io::csv_reader_options::builder(source_info);
  auto options     = builder.build();
  return cudf::io::read_csv(options);
}

static void bench_tokenize(nvbench::state& state)
{
  auto csv_metadata = read_csv("input_strings.csv");
  cudf::strings_column_view input(csv_metadata.tbl->view().column(0));

  auto mps        = nvtext::load_merge_pairs_file("merges.txt");
  auto vocab      = nvtext::load_vocabulary_file("hashed_vocab.txt");
  auto seq_len    = 64;
  auto stride     = 48;
  auto lower_case = true;
  auto truncate   = false;

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));

  auto chars_size = input.chars_size();
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);
  state.add_global_memory_writes<nvbench::int8_t>(chars_size);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto bpe    = nvtext::byte_pair_encoding(input, *mps);
    auto result = nvtext::subword_tokenize(input, *vocab, seq_len, stride, lower_case, truncate);
  });
}

NVBENCH_BENCH(bench_tokenize).set_name("bpe_tokenize");
