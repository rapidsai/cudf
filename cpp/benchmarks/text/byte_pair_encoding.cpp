/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/strings/strings_column_view.hpp>

#include <nvtext/byte_pair_encoding.hpp>

#include <rmm/device_buffer.hpp>

#include <nvbench/nvbench.cuh>

static void bench_byte_pair_encoding(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));

  auto mpt         = cudf::test::strings_column_wrapper({
    "e n",    // 14
    "i t",    // 16
    "i s",    // 17
    "e s",    // 20
    "en t",   // 44
    "c e",    // 90
    "es t",   // 141
    "en ce",  // 340
    "t h",    // 146
    "h i",    // 5049
    "th is",  // 5407
    "t est",  // 9034
    "s i",    // 13142
    "s ent"   // 33832
  });
  auto merge_pairs = nvtext::load_merge_pairs(cudf::strings_column_view(mpt));

  data_profile const strings_profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width);
  auto const strings_table =
    create_random_table({cudf::type_id::STRING}, row_count{num_rows}, strings_profile);
  cudf::strings_column_view input(strings_table->view().column(0));

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));

  auto chars_size = input.chars_size(cudf::get_default_stream());
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);
  // output are integers (one per row)
  state.add_global_memory_writes<nvbench::int32_t>(num_rows);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto result = nvtext::byte_pair_encoding(input, *merge_pairs);
  });
}

NVBENCH_BENCH(bench_byte_pair_encoding)
  .set_name("byte_pair_encoding")
  .add_int64_axis("min_width", {0})
  .add_int64_axis("max_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144});
