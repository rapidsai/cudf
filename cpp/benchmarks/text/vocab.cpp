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

#include <cudf_test/column_wrapper.hpp>

#include <cudf/reduction.hpp>
#include <nvtext/tokenize.hpp>

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/char_types/char_types.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

static void bench_vocab_tokenize(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const row_width = static_cast<cudf::size_type>(state.get_int64("row_width"));

  if (static_cast<std::size_t>(num_rows) * static_cast<std::size_t>(row_width) >=
      static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max())) {
    state.skip("Skip benchmarks greater than size_type limit");
  }

  auto const column = [num_rows, row_width] {
    data_profile const profile = data_profile_builder().no_validity().distribution(
      cudf::type_id::STRING, distribution_id::NORMAL, 0, row_width);
    auto const col = create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile);
    return cudf::strings::filter_characters_of_type(
      cudf::strings_column_view(col->view()),
      cudf::strings::string_character_types::ALL_TYPES,
      cudf::string_scalar(" "),
      cudf::strings::string_character_types::ALPHANUM);
  }();
  cudf::strings_column_view input(column->view());

  auto const vocab_col = [] {
    data_profile const profile = data_profile_builder().no_validity().distribution(
      cudf::type_id::STRING, distribution_id::NORMAL, 0, 15);
    auto const col = create_random_column(cudf::type_id::STRING, row_count{100}, profile);
    return cudf::strings::filter_characters_of_type(
      cudf::strings_column_view(col->view()),
      cudf::strings::string_character_types::ALL_TYPES,
      cudf::string_scalar(""),
      cudf::strings::string_character_types::ALPHANUM);
  }();
  auto const vocab = nvtext::load_vocabulary(cudf::strings_column_view(vocab_col->view()));

  auto token_count = [input] {
    auto const counts = nvtext::count_tokens(input);
    auto const agg    = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
    auto const count  = cudf::reduce(counts->view(), *agg, counts->type());
    return static_cast<cudf::scalar_type_t<cudf::size_type>*>(count.get())
      ->value(cudf::get_default_stream());
  }();

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  auto chars_size = input.chars_size() + cudf::strings_column_view(vocab_col->view()).chars_size();
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);
  state.add_global_memory_writes<nvbench::int32_t>(token_count);

  auto const delimiter = cudf::string_scalar("");
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto result = nvtext::tokenize_with_vocabulary(input, *vocab, delimiter);
  });
}

NVBENCH_BENCH(bench_vocab_tokenize)
  .set_name("vocab_tokenize")
  .add_int64_axis("row_width", {32, 64, 128, 256, 512, 1024})
  .add_int64_axis("num_rows", {262144, 524288, 1048576, 2097152, 4194304, 16777216});
