/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <benchmarks/fixture/rmm_pool_raii.hpp>
#include <benchmarks/join/join_benchmark_common.hpp>

template <typename key_type, typename payload_type, bool Nullable>
void nvbench_join(nvbench::state& state,
                  nvbench::type_list<key_type, payload_type, nvbench::enum_type<Nullable>>)
{
  // TODO: to be replaced by nvbench fixture once it's ready
  cudf::rmm_pool_raii pool_raii;

  auto join = [](cudf::table_view const& left_input,
                 cudf::table_view const& right_input,
                 std::vector<cudf::size_type> const& left_on,
                 std::vector<cudf::size_type> const& right_on,
                 cudf::null_equality compare_nulls,
                 rmm::cuda_stream_view stream) {
    cudf::hash_join hj_obj(left_input.select(left_on), compare_nulls, stream);
    auto result =
      hj_obj.inner_join(right_input.select(right_on), compare_nulls, std::nullopt, stream);
    auto join_indices = std::make_pair(std::move(result.second), std::move(result.first));
  };

  BM_join<key_type, payload_type, Nullable>(state, join);
}

// join -----------------------------------------------------------------------
NVBENCH_BENCH_TYPES(nvbench_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int32_t>,
                                      nvbench::type_list<nvbench::int32_t>,
                                      nvbench::enum_type_list<false>))
  .set_name("join_32bit")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {100'000, 10'000'000, 80'000'000, 100'000'000})
  .add_int64_axis("Probe Table Size", {400'000, 40'000'000, 100'000'000, 240'000'000});

NVBENCH_BENCH_TYPES(nvbench_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int64_t>,
                                      nvbench::type_list<nvbench::int64_t>,
                                      nvbench::enum_type_list<false>))
  .set_name("join_64bit")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {40'000'000, 50'000'000})
  .add_int64_axis("Probe Table Size", {50'000'000, 120'000'000});

NVBENCH_BENCH_TYPES(nvbench_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int32_t>,
                                      nvbench::type_list<nvbench::int32_t>,
                                      nvbench::enum_type_list<true>))
  .set_name("join_32bit_nulls")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {100'000, 10'000'000, 80'000'000, 100'000'000})
  .add_int64_axis("Probe Table Size", {400'000, 40'000'000, 100'000'000, 240'000'000});

NVBENCH_BENCH_TYPES(nvbench_join,
                    NVBENCH_TYPE_AXES(nvbench::type_list<nvbench::int64_t>,
                                      nvbench::type_list<nvbench::int64_t>,
                                      nvbench::enum_type_list<true>))
  .set_name("join_64bit_nulls")
  .set_type_axes_names({"Key Type", "Payload Type", "Nullable"})
  .add_int64_axis("Build Table Size", {40'000'000, 50'000'000})
  .add_int64_axis("Probe Table Size", {50'000'000, 120'000'000});
