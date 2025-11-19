/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_uvector.hpp>

#include <nvbench/nvbench.cuh>

#include <limits>

static void bench_factory(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));

  data_profile const profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width);
  auto const column = create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile);
  auto const sv     = cudf::strings_column_view(column->view());

  auto stream    = cudf::get_default_stream();
  auto mr        = cudf::get_current_device_resource_ref();
  auto d_strings = cudf::strings::detail::create_string_vector_from_column(sv, stream, mr);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto const data_size = column->alloc_size();
  state.add_global_memory_reads<nvbench::int8_t>(data_size);
  state.add_global_memory_writes<nvbench::int8_t>(data_size);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::make_strings_column(d_strings, cudf::string_view{nullptr, 0});
  });
}

NVBENCH_BENCH(bench_factory)
  .set_name("factory")
  .add_int64_axis("min_width", {0})
  .add_int64_axis("max_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152});
