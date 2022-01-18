/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

enum class algorithm { SORT_BASED, HASH_BASED };

// mandatory for enum types
NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  // Enum type:
  algorithm,
  // Callable to generate input strings:
  // Short identifier used for tables, command-line args, etc.
  // Used when context is available to figure out the enum type.
  [](algorithm algo) {
    switch (algo) {
      case algorithm::SORT_BASED: return "SORT_BASED";
      case algorithm::HASH_BASED: return "HASH_BASED";
      default: return "ERROR";
    }
  },
  // Callable to generate descriptions:
  // If non-empty, these are used in `--list` to describe values.
  // Used when context may not be available to figure out the type from the
  // input string.
  // Just use `[](auto) { return std::string{}; }` if you don't want these.
  [](auto) { return std::string{}; })

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  // Enum type:
  cudf::duplicate_keep_option,
  // Callable to generate input strings:
  // Short identifier used for tables, command-line args, etc.
  // Used when context is available to figure out the enum type.
  [](cudf::duplicate_keep_option option) {
    switch (option) {
      case cudf::duplicate_keep_option::KEEP_FIRST: return "KEEP_FIRST";
      case cudf::duplicate_keep_option::KEEP_LAST: return "KEEP_LAST";
      case cudf::duplicate_keep_option::KEEP_NONE: return "KEEP_NONE";
      default: return "ERROR";
    }
  },
  // Callable to generate descriptions:
  // If non-empty, these are used in `--list` to describe values.
  // Used when context may not be available to figure out the type from the
  // input string.
  // Just use `[](auto) { return std::string{}; }` if you don't want these.
  [](auto) { return std::string{}; })

NVBENCH_DECLARE_TYPE_STRINGS(cudf::timestamp_ms, "cudf::timestamp_ms", "cudf::timestamp_ms");

template <typename Type, algorithm Algo, cudf::duplicate_keep_option Keep>
void nvbench_compaction(
  nvbench::state& state,
  nvbench::type_list<Type, nvbench::enum_type<Algo>, nvbench::enum_type<Keep>>)
{
  if constexpr ((not std::is_same_v<Type, int32_t> and
                 Keep != cudf::duplicate_keep_option::KEEP_FIRST and
                 Algo == algorithm::SORT_BASED) or
                (Algo == algorithm::HASH_BASED and
                 Keep != cudf::duplicate_keep_option::KEEP_FIRST)) {
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
    auto const result = [&]() {
      if constexpr (Algo == algorithm::HASH_BASED) {
        return cudf::detail::unordered_drop_duplicates(
          input_table, {0}, cudf::null_equality::EQUAL, stream_view);
      } else {
        return cudf::detail::sort_and_drop_duplicates(input_table,
                                                      {0},
                                                      Keep,
                                                      cudf::null_equality::EQUAL,
                                                      cudf::null_order::BEFORE,
                                                      stream_view);
      }
    }();
  });
}

using data_type   = nvbench::type_list<bool,
                                     nvbench::int8_t,
                                     nvbench::int32_t,
                                     nvbench::int64_t,
                                     nvbench::float32_t,
                                     cudf::timestamp_ms>;
using algo        = nvbench::enum_type_list<algorithm::SORT_BASED, algorithm::HASH_BASED>;
using keep_option = nvbench::enum_type_list<cudf::duplicate_keep_option::KEEP_FIRST,
                                            cudf::duplicate_keep_option::KEEP_LAST,
                                            cudf::duplicate_keep_option::KEEP_NONE>;

NVBENCH_BENCH_TYPES(nvbench_compaction, NVBENCH_TYPE_AXES(data_type, algo, keep_option))
  .set_name("drop_duplicates")
  .set_type_axes_names({"Type", "Algorithm", "KeepOption"})
  .add_int64_axis("NumRows", {10'000, 100'000, 1'000'000, 10'000'000});
