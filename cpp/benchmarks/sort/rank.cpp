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

#include <cudf/column/column_view.hpp>
#include <cudf/sorting.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

static void bench_rank(nvbench::state& state)
{
  auto const n_rows = static_cast<cudf::size_type>(state.get_int64("n_rows"));
  auto const nulls  = state.get_float64("nulls");

  // Create columns with values in the range [0,100)
  data_profile const profile =
    data_profile_builder().cardinality(0).null_probability(nulls).distribution(
      cudf::type_id::INT32, distribution_id::UNIFORM, 0, 10);

  auto input = create_random_column(cudf::type_id::INT32, row_count{n_rows}, profile);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.add_element_count(n_rows, "n_rows");
  state.add_global_memory_reads<nvbench::int32_t>(n_rows);
  state.add_global_memory_writes<nvbench::int32_t>(n_rows);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto result = cudf::rank(input->view(),
                             cudf::rank_method::FIRST,
                             cudf::order::ASCENDING,
                             nulls ? cudf::null_policy::INCLUDE : cudf::null_policy::EXCLUDE,
                             cudf::null_order::AFTER,
                             false);
  });
}

NVBENCH_BENCH(bench_rank)
  .set_name("rank")
  .add_float64_axis("nulls", {0, 0.1})
  .add_int64_axis("n_rows", {32768, 262144, 2097152, 16777216, 67108864});
