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

#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

template <typename DataType>
static void bench_sort(nvbench::state& state, nvbench::type_list<DataType>)
{
  auto const ordered   = static_cast<bool>(state.get_int64("ordered"));
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_cols  = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const nulls     = state.get_float64("nulls");
  auto const data_type = cudf::type_to_id<DataType>();

  data_profile const profile =
    data_profile_builder().cardinality(0).null_probability(nulls).distribution(
      data_type, distribution_id::UNIFORM, 100, 10'000);
  auto input_table =
    create_random_table(cycle_dtypes({data_type}, num_cols), row_count{num_rows}, profile);
  cudf::table_view input{*input_table};

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.add_global_memory_reads<nvbench::int8_t>(input_table->alloc_size());
  state.add_global_memory_writes<nvbench::int32_t>(num_rows);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    if (ordered) {
      cudf::sorted_order(input);
    } else {
      cudf::sort(input);
    }
  });
}

NVBENCH_DECLARE_TYPE_STRINGS(cudf::timestamp_ms, "time_ms", "time_ms");

using Types = nvbench::type_list<int32_t, float, cudf::timestamp_ms>;

NVBENCH_BENCH_TYPES(bench_sort, NVBENCH_TYPE_AXES(Types))
  .set_name("sort")
  .add_float64_axis("nulls", {0, 0.1})
  .add_int64_axis("num_rows", {262144, 2097152, 16777216, 67108864})
  .add_int64_axis("num_cols", {1, 8})
  .add_int64_axis("ordered", {0, 1});
