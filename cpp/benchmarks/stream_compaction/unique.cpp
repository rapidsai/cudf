/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

#include <cudf/column/column_view.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/types.hpp>
#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <fixture/rmm_pool_raii.hpp>

#include <nvbench/nvbench.cuh>

#include <memory>
#include <random>

// necessary for custom enum types
// see: https://github.com/NVIDIA/nvbench/blob/main/examples/enums.cu
NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  // Enum type:
  cudf::duplicate_keep_option,
  // Callable to generate input strings:
  [](cudf::duplicate_keep_option option) {
    switch (option) {
      case cudf::duplicate_keep_option::KEEP_FIRST: return "KEEP_FIRST";
      case cudf::duplicate_keep_option::KEEP_LAST: return "KEEP_LAST";
      case cudf::duplicate_keep_option::KEEP_NONE: return "KEEP_NONE";
      default: return "ERROR";
    }
  },
  // Callable to generate descriptions:
  [](auto) { return std::string{}; })

NVBENCH_DECLARE_TYPE_STRINGS(cudf::timestamp_ms, "cudf::timestamp_ms", "cudf::timestamp_ms");

template <typename Type, cudf::duplicate_keep_option Keep>
void nvbench_unique(nvbench::state& state, nvbench::type_list<Type, nvbench::enum_type<Keep>>)
{
  if constexpr (not std::is_same_v<Type, int32_t> and
                Keep != cudf::duplicate_keep_option::KEEP_FIRST) {
    state.skip("Skip unwanted benchmarks.");
  }

  cudf::rmm_pool_raii pool_raii;

  auto const num_rows = state.get_int64("NumRows");

  cudf::test::UniformRandomGenerator<long> rand_gen(0, 100);
  auto elements = cudf::detail::make_counting_transform_iterator(
    0, [&rand_gen](auto row) { return rand_gen.generate(); });
  auto valids = cudf::detail::make_counting_transform_iterator(
    0, [](auto i) { return i % 100 == 0 ? false : true; });
  cudf::test::fixed_width_column_wrapper<Type, long> values(elements, elements + num_rows, valids);

  auto input_column = cudf::column_view(values);
  auto input_table  = cudf::table_view({input_column, input_column, input_column, input_column});

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    rmm::cuda_stream_view stream_view{launch.get_stream()};
    auto result =
      cudf::detail::unique(input_table, {0}, Keep, cudf::null_equality::EQUAL, stream_view);
  });
}

using data_type   = nvbench::type_list<bool, int8_t, int32_t, int64_t, float, cudf::timestamp_ms>;
using keep_option = nvbench::enum_type_list<cudf::duplicate_keep_option::KEEP_FIRST,
                                            cudf::duplicate_keep_option::KEEP_LAST,
                                            cudf::duplicate_keep_option::KEEP_NONE>;

NVBENCH_BENCH_TYPES(nvbench_unique, NVBENCH_TYPE_AXES(data_type, keep_option))
  .set_name("unique")
  .set_type_axes_names({"Type", "KeepOption"})
  .add_int64_axis("NumRows", {10'000, 100'000, 1'000'000, 10'000'000});
