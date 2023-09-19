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
#include <benchmarks/fixture/benchmark_fixture.hpp>

#include <cudf/detail/search.hpp>
#include <cudf/lists/list_view.hpp>
#include <cudf/types.hpp>

#include <rmm/mr/device/per_device_resource.hpp>

#include <nvbench/nvbench.cuh>

auto constexpr num_unique_elements = 1000;

template <typename Type>
static void nvbench_contains_table(nvbench::state& state, nvbench::type_list<Type>)
{
  auto const size               = state.get_int64("table_size");
  auto const dtype              = cudf::type_to_id<Type>();
  double const null_probability = state.get_float64("null_probability");

  auto builder = data_profile_builder().null_probability(null_probability);
  if (dtype == cudf::type_id::LIST) {
    builder.distribution(dtype, distribution_id::UNIFORM, 0, num_unique_elements)
      .distribution(cudf::type_id::INT32, distribution_id::UNIFORM, 0, num_unique_elements)
      .list_depth(1);
  } else {
    builder.distribution(dtype, distribution_id::UNIFORM, 0, num_unique_elements);
  }

  auto const haystack = create_random_table(
    {dtype}, table_size_bytes{static_cast<size_t>(size)}, data_profile{builder}, 0);
  auto const needles = create_random_table(
    {dtype}, table_size_bytes{static_cast<size_t>(size)}, data_profile{builder}, 1);

  auto mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto const stream_view = rmm::cuda_stream_view{launch.get_stream()};
    [[maybe_unused]] auto const result =
      cudf::detail::contains(haystack->view(),
                             needles->view(),
                             cudf::null_equality::EQUAL,
                             cudf::nan_equality::ALL_EQUAL,
                             stream_view,
                             rmm::mr::get_current_device_resource());
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH_TYPES(nvbench_contains_table,
                    NVBENCH_TYPE_AXES(nvbench::type_list<int32_t, cudf::list_view>))
  .set_name("contains_table")
  .set_type_axes_names({"type"})
  .add_float64_axis("null_probability", {0.0, 0.1})
  .add_int64_axis("table_size", {10'000, 100'000, 1'000'000, 10'000'000});
