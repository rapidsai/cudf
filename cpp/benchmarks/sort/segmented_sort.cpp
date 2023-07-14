/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

void nvbench_segmented_sort(nvbench::state& state)
{
  auto const stable     = static_cast<bool>(state.get_int64("stable"));
  auto const dtype      = cudf::type_to_id<int32_t>();
  auto const size_bytes = static_cast<size_t>(state.get_int64("size_bytes"));
  auto const null_freq  = state.get_float64("null_frequency");
  auto const row_width  = static_cast<cudf::size_type>(state.get_int64("row_width"));

  data_profile const table_profile =
    data_profile_builder().null_probability(null_freq).distribution(
      dtype, distribution_id::UNIFORM, 0, 10);
  auto const input =
    create_random_table({cudf::type_id::INT32}, table_size_bytes{size_bytes}, table_profile);
  auto const rows = input->num_rows();

  auto const segments = cudf::sequence((rows / row_width) + 1,
                                       cudf::numeric_scalar<int32_t>(0),
                                       cudf::numeric_scalar<int32_t>(row_width));

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.add_element_count(size_bytes, "bytes");
  state.add_global_memory_reads<nvbench::int32_t>(rows * row_width);
  state.add_global_memory_writes<nvbench::int32_t>(rows);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    if (stable)
      cudf::stable_segmented_sorted_order(*input, *segments);
    else
      cudf::segmented_sorted_order(*input, *segments);
  });
}

NVBENCH_BENCH(nvbench_segmented_sort)
  .set_name("segmented_sort")
  .add_int64_axis("stable", {0, 1})
  .add_int64_power_of_two_axis("size_bytes", {16, 18, 20, 22, 24, 28})
  .add_float64_axis("null_frequency", {0, 0.1})
  .add_int64_axis("row_width", {16, 128, 1024});
