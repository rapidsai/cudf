/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cudf/interop.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <vector>

template <cudf::type_id data_type>
void BM_to_arrow_device(nvbench::state& state, nvbench::type_list<nvbench::enum_type<data_type>>)
{
  auto const table_size     = static_cast<cudf::size_type>(state.get_int64("table_size"));
  auto const column_count   = static_cast<cudf::size_type>(state.get_int64("column_count"));
  auto const elements_count = static_cast<int64_t>(table_size) * column_count;

  std::vector<cudf::type_id> types;

  std::fill_n(std::back_inserter(types), column_count, data_type);

  auto const source_table = create_random_table(types, row_count{table_size});

  state.add_element_count(elements_count, "elements_count");
  state.add_global_memory_reads<cudf::id_to_type<data_type>>(elements_count);
  state.add_global_memory_writes<cudf::id_to_type<data_type>>(elements_count);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::to_arrow_device(source_table->view(), rmm::cuda_stream_view{launch.get_stream()});
  });
}

template <cudf::type_id data_type>
void BM_to_arrow_host(nvbench::state& state, nvbench::type_list<nvbench::enum_type<data_type>>)
{
  auto const table_size     = static_cast<cudf::size_type>(state.get_int64("table_size"));
  auto const column_count   = static_cast<cudf::size_type>(state.get_int64("column_count"));
  auto const elements_count = static_cast<int64_t>(table_size) * column_count;

  std::vector<cudf::type_id> types;

  std::fill_n(std::back_inserter(types), column_count, data_type);

  auto const source_table = create_random_table(types, row_count{table_size});

  state.add_element_count(elements_count, "elements_count");
  state.add_global_memory_reads<cudf::id_to_type<data_type>>(elements_count);
  state.add_global_memory_writes<cudf::id_to_type<data_type>>(elements_count);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::to_arrow_host(source_table->view(), rmm::cuda_stream_view{launch.get_stream()});
  });
}

using data_types =
  nvbench::enum_type_list<cudf::type_id::INT64, cudf::type_id::BOOL8, cudf::type_id::DECIMAL64>;

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  cudf::type_id,
  [](cudf::type_id value) {
    switch (value) {
      case cudf::type_id::INT64: return "cudf::type_id::INT64";
      case cudf::type_id::BOOL8: return "cudf::type_id::BOOL8";
      case cudf::type_id::DECIMAL64: return "cudf::type_id::DECIMAL64";
      default: return "unknown";
    }
  },
  [](cudf::type_id value) {
    switch (value) {
      case cudf::type_id::INT64: return "cudf::type_id::INT64";
      case cudf::type_id::BOOL8: return "cudf::type_id::BOOL8";
      case cudf::type_id::DECIMAL64: return "cudf::type_id::DECIMAL64";
      default: return "unknown";
    }
  })

NVBENCH_BENCH_TYPES(BM_to_arrow_device, NVBENCH_TYPE_AXES(data_types))
  .set_type_axes_names({"data_type"})
  .set_name("to_arrow_device")
  .add_int64_axis("table_size", {10'000, 100'000, 1'000'000, 10'000'000, 100'000'000})
  .add_int64_axis("column_count", {1, 10, 100});

NVBENCH_BENCH_TYPES(BM_to_arrow_host, NVBENCH_TYPE_AXES(data_types))
  .set_type_axes_names({"data_type"})
  .set_name("to_arrow_host")
  .add_int64_axis("table_size", {10'000, 100'000, 1'000'000, 10'000'000, 100'000'000})
  .add_int64_axis("column_count", {1, 10, 100});

void BM_from_arrow_device();
void BM_from_arrow_host();
