/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

#include <cudf/copying.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

static void bench_shift(nvbench::state& state)
{
  auto const num_rows     = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const shift_factor = static_cast<int>(state.get_int64("shift_factor"));
  auto const offset       = num_rows * (static_cast<double>(shift_factor) / 100.0);
  auto const use_validity = static_cast<bool>(state.get_int64("use_validity"));

  auto const input_table =
    create_sequence_table({cudf::type_id::INT32},
                          row_count{num_rows},
                          use_validity ? std::optional<double>{1.0} : std::nullopt);
  cudf::column_view input{input_table->get_column(0)};

  auto fill = use_validity ? cudf::numeric_scalar<int32_t>(0) : cudf::numeric_scalar<int32_t>(777);

  auto const elems_read = (num_rows - offset);
  auto const bytes_read = elems_read * sizeof(int32_t);

  auto const elems_written = use_validity ? (num_rows - offset) : num_rows;
  auto const bytes_written = elems_written * sizeof(int32_t);
  auto const null_bytes    = use_validity ? 2 * cudf::bitmask_allocation_size_bytes(num_rows) : 0;

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.add_global_memory_reads<int8_t>(bytes_read);
  state.add_global_memory_writes<int8_t>(bytes_written + null_bytes);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) { cudf::shift(input, offset, fill); });
}

NVBENCH_BENCH(bench_shift)
  .set_name("shift")
  .add_int64_axis("num_rows", {1024, 32768, 1048576, 33554432, 1073741824})
  .add_int64_axis("shift_factor", {0, 10, 50, 100})
  .add_int64_axis("use_validity", {false, true});
