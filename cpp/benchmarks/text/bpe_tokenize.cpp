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

#include <cudf/filling.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <nvtext/bpe_tokenize.hpp>
#include <nvtext/subword_tokenize.hpp>

#include <nvbench/nvbench.cuh>

static void bench_bpe(nvbench::state& state)
{
  auto const n_rows    = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const row_width = static_cast<cudf::size_type>(state.get_int64("row_width"));
  if (static_cast<std::size_t>(n_rows) * static_cast<std::size_t>(row_width) >=
      static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max())) {
    state.skip("Skip benchmarks greater than size_type limit");
  }

  auto raw_data = cudf::test::strings_column_wrapper({"test sentence",
                                                      "thisis it",
                                                      "this is sentence",
                                                      "this isit",
                                                      "thisisit",
                                                      "sentenceisit",
                                                      "this sentence is test",
                                                      "isitthis",
                                                      "this this it this",
                                                      "sentence"})
                    .release();

  if (row_width / 20 > 1) {
    std::vector<cudf::column_view> columns;
    for (int i = 0; i < row_width / 20; ++i) {
      columns.push_back(raw_data->view());
    }
    raw_data = cudf::strings::concatenate(cudf::table_view(columns));
  }
  auto data_view = raw_data->view();

  // Create a randomized gather-map to build a column out of the raw strings in data.
  data_profile gather_profile =
    data_profile_builder().cardinality(0).null_probability(0.0).distribution(
      cudf::type_id::INT32, distribution_id::UNIFORM, 1, data_view.size() - 1);
  auto gather_table =
    create_random_table({cudf::type_id::INT32}, row_count{n_rows}, gather_profile);
  gather_table->get_column(0).set_null_mask(rmm::device_buffer{}, 0);
  auto gather_map  = gather_table->view().column(0);
  auto table_input = cudf::gather(cudf::table_view({data_view}), gather_map);
  auto input       = cudf::strings_column_view(table_input->view().column(0));

  cudf::test::strings_column_wrapper merge_pairs(
    {"e n", "i t", "i s", "e s", "en t", "c e", "es t", "en ce", "t est", "s ent"});
  auto mps = nvtext::load_merge_pairs(cudf::strings_column_view(merge_pairs));

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));

  auto chars_size = input.chars_size();
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);
  state.add_global_memory_writes<nvbench::int8_t>(chars_size);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto result = nvtext::byte_pair_encoding(input, *mps);
  });
}

NVBENCH_BENCH(bench_bpe)
  .set_name("byte_pair_encoding")
  .add_int64_axis("row_width", {32, 64, 128, 256, 512})
  .add_int64_axis("num_rows", {32768, 262144, 2097152, 16777216});
