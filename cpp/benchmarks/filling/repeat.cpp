/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.
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
#include <benchmarks/common/nvbench_utilities.hpp>

#include <cudf/filling.hpp>

#include <nvbench/nvbench.cuh>

namespace {
template <typename TypeParam>
void nvbench_repeat(nvbench::state& state, nvbench::type_list<TypeParam>)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_cols = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const nulls    = state.get_int64("nulls");

  auto const input_table =
    create_sequence_table(cycle_dtypes({cudf::type_to_id<TypeParam>()}, num_cols),
                          row_count{num_rows},
                          nulls ? std::optional<double>{0.1} : std::nullopt);
  // Create table view
  auto const input = input_table->view();

  // repeat counts
  data_profile const profile = data_profile_builder().cardinality(0).no_validity().distribution(
    cudf::type_to_id<cudf::size_type>(), distribution_id::UNIFORM, 0, 3);
  auto counts =
    create_random_column(cudf::type_to_id<cudf::size_type>(), row_count{num_rows}, profile);

  auto output = cudf::repeat(input, counts->view());

  state.add_global_memory_reads(input_table->alloc_size());
  state.add_global_memory_writes(output->alloc_size());
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { auto result = cudf::repeat(input, counts->view()); });
}
}  // namespace

using Types = nvbench::type_list<double, int32_t>;

NVBENCH_BENCH_TYPES(nvbench_repeat, NVBENCH_TYPE_AXES(Types))
  .set_name("repeat")
  .set_type_axes_names({"DataType"})
  .add_int64_power_of_two_axis("num_rows", {10, 14, 18, 22, 26})
  .add_int64_axis("num_cols", {1, 2, 4, 8})
  .add_int64_axis("nulls", {0, 1});
