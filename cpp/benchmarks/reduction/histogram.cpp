/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include "cudf/aggregation.hpp"
#include "cudf/detail/aggregation/aggregation.hpp"

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/nvbench_utilities.hpp>
#include <benchmarks/common/table_utilities.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/reduction.hpp>
#include <cudf/reduction/detail/histogram.hpp>
#include <cudf/types.hpp>

#include <nvbench/nvbench.cuh>

template <typename type>
static void nvbench_reduction_histogram(nvbench::state& state, nvbench::type_list<type>)
{
  auto const dtype = cudf::type_to_id<type>();

  auto const cardinality      = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const num_rows         = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const null_probability = state.get_float64("null_probability");

  if (cardinality > num_rows) {
    state.skip("cardinality > num_rows");
    return;
  }

  data_profile const profile = data_profile_builder()
                                 .null_probability(null_probability)
                                 .cardinality(cardinality)
                                 .distribution(dtype, distribution_id::UNIFORM, 0, num_rows);

  auto const input = create_random_column(dtype, row_count{num_rows}, profile);
  auto agg         = cudf::make_histogram_aggregation<cudf::reduce_aggregation>();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    rmm::cuda_stream_view stream_view{launch.get_stream()};
    auto result = cudf::reduce(*input, *agg, input->type(), stream_view);
  });

  state.add_element_count(input->size());
}

using data_type = nvbench::type_list<int32_t, int64_t>;

NVBENCH_BENCH_TYPES(nvbench_reduction_histogram, NVBENCH_TYPE_AXES(data_type))
  .set_name("histogram")
  .add_float64_axis("null_probability", {0.1})
  .add_int64_axis("cardinality",
                  {0, 100, 1'000, 10'000, 100'000, 1'000'000, 10'000'000, 50'000'000})
  .add_int64_axis("num_rows", {10'000, 100'000, 1'000'000, 10'000'000, 100'000'000});
