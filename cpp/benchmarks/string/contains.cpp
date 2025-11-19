/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/fixture/benchmark_fixture.hpp>

#include <cudf/strings/contains.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

// longer pattern lengths demand more working memory per string
std::string patterns[] = {"^\\d+ [a-z]+", "[A-Z ]+\\d+ +\\d+[A-Z]+\\d+$", "5W43"};

static void bench_contains(nvbench::state& state)
{
  auto const num_rows      = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const row_width     = static_cast<cudf::size_type>(state.get_int64("row_width"));
  auto const pattern_index = static_cast<cudf::size_type>(state.get_int64("pattern"));
  auto const hit_rate      = static_cast<cudf::size_type>(state.get_int64("hit_rate"));

  auto col   = create_string_column(num_rows, row_width, hit_rate);
  auto input = cudf::strings_column_view(col->view());

  auto pattern = patterns[pattern_index];
  auto program = cudf::strings::regex_program::create(pattern);

  state.add_global_memory_reads<nvbench::int8_t>(col->alloc_size());
  state.add_global_memory_writes<nvbench::int32_t>(input.size());

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { cudf::strings::contains_re(input, *program); });
}

NVBENCH_BENCH(bench_contains)
  .set_name("contains")
  .add_int64_axis("row_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_int64_axis("hit_rate", {50, 100})  // percentage
  .add_int64_axis("pattern", {0, 1, 2});
