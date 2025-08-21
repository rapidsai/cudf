/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

template <typename DataType>
void bench_segmented_top_k(nvbench::state& state, nvbench::type_list<DataType>)
{
  auto const ordered   = static_cast<bool>(state.get_int64("ordered"));
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const segment   = static_cast<cudf::size_type>(state.get_int64("segment"));
  auto const k         = static_cast<cudf::size_type>(state.get_int64("k"));
  auto const data_type = cudf::type_to_id<DataType>();

  data_profile const profile = data_profile_builder().no_validity().distribution(
    data_type, distribution_id::UNIFORM, 0, segment);
  auto const input = create_random_column(data_type, row_count{num_rows}, profile);

  auto const segments = cudf::sequence((num_rows / segment) + 1,
                                       cudf::numeric_scalar<int32_t>(0),
                                       cudf::numeric_scalar<int32_t>(segment));

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.add_global_memory_reads<nvbench::int32_t>(num_rows);
  state.add_global_memory_writes<nvbench::int32_t>(segments->size() * k);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    if (ordered) {
      cudf::segmented_top_k_order(input->view(), segments->view(), k);
    } else {
      cudf::segmented_top_k(input->view(), segments->view(), k);
    }
  });
}

NVBENCH_DECLARE_TYPE_STRINGS(cudf::timestamp_s, "time_s", "time_s");

using Types = nvbench::type_list<int32_t, float, cudf::timestamp_s>;

NVBENCH_BENCH_TYPES(bench_segmented_top_k, NVBENCH_TYPE_AXES(Types))
  .set_name("segmented_top_k")
  .add_int64_axis("num_rows", {262144, 2097152, 16777216, 67108864})
  .add_int64_axis("segment", {1024, 2048})
  .add_int64_axis("k", {100, 1000})
  .add_int64_axis("ordered", {0, 1});
