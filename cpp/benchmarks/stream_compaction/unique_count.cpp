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

#include <cudf/column/column_view.hpp>
#include <cudf/sorting.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/types.hpp>

#include <nvbench/nvbench.cuh>

template <typename Type>
void nvbench_unique_count(nvbench::state& state, nvbench::type_list<Type>)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("NumRows"));
  auto const nulls    = state.get_float64("NullProbability");

  data_profile profile = data_profile_builder().cardinality(0).null_probability(nulls).distribution(
    cudf::type_to_id<Type>(), distribution_id::UNIFORM, 0, num_rows / 100);

  auto source_column = create_random_column(cudf::type_to_id<Type>(), row_count{num_rows}, profile);
  auto sorted_table  = cudf::sort(cudf::table_view({source_column->view()}));

  auto input = sorted_table->view();

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::unique_count(input, cudf::null_equality::EQUAL);
  });
}

using data_type = nvbench::type_list<bool, int8_t, int32_t, int64_t, float, cudf::timestamp_ms>;

NVBENCH_BENCH_TYPES(nvbench_unique_count, NVBENCH_TYPE_AXES(data_type))
  .set_name("unique_count")
  .set_type_axes_names({"Type"})
  .add_int64_axis("NumRows", {10'000, 100'000, 1'000'000, 10'000'000})
  .add_float64_axis("NullProbability", {0.0, 0.1});
