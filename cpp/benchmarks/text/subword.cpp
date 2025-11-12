/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/column_wrapper.hpp>

#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/wordpiece_tokenize.hpp>

#include <nvbench/nvbench.cuh>

#include <vector>

static void bench_wordpiece_tokenizer(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const max_words = static_cast<cudf::size_type>(state.get_int64("max_words"));

  auto const h_strings = std::vector<char const*>(
    num_rows,
    "This is a test This is a test This is a test This is a test This is a test This is a test "
    "This is a test This is a test ");
  auto const num_words = 32;  // "This is a test" * 8
  auto const d_strings = cudf::test::strings_column_wrapper(h_strings.begin(), h_strings.end());
  auto const input     = cudf::strings_column_view{d_strings};

  auto const vocabulary =
    cudf::test::strings_column_wrapper({"", "[UNK]", "This", "is", "a", "test"});
  auto const vocab = nvtext::load_wordpiece_vocabulary(cudf::strings_column_view(vocabulary));

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  auto chars_size = input.chars_size(cudf::get_default_stream());
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);
  auto out_size = num_rows * (max_words > 0 ? std::min(max_words, num_words) : num_words);
  state.add_global_memory_writes<nvbench::int32_t>(out_size);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto result = nvtext::wordpiece_tokenize(input, *vocab, max_words);
  });
}

NVBENCH_BENCH(bench_wordpiece_tokenizer)
  .set_name("wordpiece_tokenize")
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_int64_axis("max_words", {0, 20, 40});
