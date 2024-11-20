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
#include <benchmarks/common/table_utilities.hpp>

#include <cudf/interop.hpp>

#include <thrust/iterator/counting_iterator.h>

#include <nanoarrow/nanoarrow.hpp>
#include <nanoarrow/nanoarrow_device.h>
#include <nanoarrow_utils.hpp>
#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <iterator>
#include <vector>

template <cudf::type_id data_type>
void BM_to_arrow_device(nvbench::state& state, nvbench::type_list<nvbench::enum_type<data_type>>)
{
  auto const num_rows     = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_columns  = static_cast<cudf::size_type>(state.get_int64("num_columns"));
  auto const num_elements = static_cast<int64_t>(num_rows) * num_columns;

  std::vector<cudf::type_id> types(num_columns, data_type);

  auto const table         = create_random_table(types, row_count{num_rows});
  int64_t const size_bytes = estimate_size(table->view());

  state.add_element_count(num_elements, "num_elements");
  state.add_global_memory_reads(size_bytes);
  state.add_global_memory_writes(size_bytes);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::to_arrow_device(table->view(), rmm::cuda_stream_view{launch.get_stream()});
  });
}

template <cudf::type_id data_type>
void BM_to_arrow_host(nvbench::state& state, nvbench::type_list<nvbench::enum_type<data_type>>)
{
  auto const num_rows     = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_columns  = static_cast<cudf::size_type>(state.get_int64("num_columns"));
  auto const num_elements = static_cast<int64_t>(num_rows) * num_columns;

  std::vector<cudf::type_id> types(num_columns, data_type);

  auto const table         = create_random_table(types, row_count{num_rows});
  int64_t const size_bytes = estimate_size(table->view());

  state.add_element_count(num_elements, "num_elements");
  state.add_global_memory_reads(size_bytes);
  state.add_global_memory_writes(size_bytes);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::to_arrow_host(table->view(), rmm::cuda_stream_view{launch.get_stream()});
  });
}

template <cudf::type_id data_type>
void BM_from_arrow_device(nvbench::state& state, nvbench::type_list<nvbench::enum_type<data_type>>)
{
  auto const num_rows     = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_columns  = static_cast<cudf::size_type>(state.get_int64("num_columns"));
  auto const num_elements = static_cast<int64_t>(num_rows) * num_columns;

  std::vector<cudf::type_id> types(num_columns, data_type);

  data_profile profile;
  profile.set_struct_depth(1);
  profile.set_list_depth(1);

  auto const table            = create_random_table(types, row_count{num_rows}, profile);
  cudf::table_view table_view = table->view();
  int64_t const size_bytes    = estimate_size(table_view);

  std::vector<cudf::column_metadata> table_metadata;

  std::transform(thrust::make_counting_iterator(0),
                 thrust::make_counting_iterator(num_columns),
                 std::back_inserter(table_metadata),
                 [&](auto const column) {
                   cudf::column_metadata column_metadata{""};
                   column_metadata.children_meta = std::vector(
                     table->get_column(column).num_children(), cudf::column_metadata{""});
                   return column_metadata;
                 });

  cudf::unique_schema_t schema      = cudf::to_arrow_schema(table_view, table_metadata);
  cudf::unique_device_array_t input = cudf::to_arrow_device(table_view);

  state.add_element_count(num_elements, "num_elements");
  state.add_global_memory_reads(size_bytes);
  state.add_global_memory_writes(size_bytes);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::from_arrow_device_column(
      schema.get(), input.get(), rmm::cuda_stream_view{launch.get_stream()});
  });
}

template <cudf::type_id data_type>
void BM_from_arrow_host(nvbench::state& state, nvbench::type_list<nvbench::enum_type<data_type>>)
{
  auto const num_rows     = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_columns  = static_cast<cudf::size_type>(state.get_int64("num_columns"));
  auto const num_elements = static_cast<int64_t>(num_rows) * num_columns;

  std::vector<cudf::type_id> types(num_columns, data_type);

  data_profile profile;
  profile.set_struct_depth(1);
  profile.set_list_depth(1);

  auto const table            = create_random_table(types, row_count{num_rows}, profile);
  cudf::table_view table_view = table->view();
  int64_t const size_bytes    = estimate_size(table_view);

  std::vector<cudf::column_metadata> table_metadata;

  std::transform(thrust::make_counting_iterator(0),
                 thrust::make_counting_iterator(num_columns),
                 std::back_inserter(table_metadata),
                 [&](auto const column) {
                   cudf::column_metadata column_metadata{""};
                   column_metadata.children_meta = std::vector(
                     table->get_column(column).num_children(), cudf::column_metadata{""});
                   return column_metadata;
                 });

  cudf::unique_schema_t schema      = cudf::to_arrow_schema(table_view, table_metadata);
  cudf::unique_device_array_t input = cudf::to_arrow_host(table_view);

  state.add_element_count(num_elements, "num_elements");
  state.add_global_memory_reads(size_bytes);
  state.add_global_memory_writes(size_bytes);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::from_arrow_host_column(
      schema.get(), input.get(), rmm::cuda_stream_view{launch.get_stream()});
  });
}

using data_types = nvbench::enum_type_list<cudf::type_id::INT8,
                                           cudf::type_id::INT16,
                                           cudf::type_id::INT32,
                                           cudf::type_id::INT64,
                                           cudf::type_id::UINT8,
                                           cudf::type_id::UINT16,
                                           cudf::type_id::UINT32,
                                           cudf::type_id::UINT64,
                                           cudf::type_id::FLOAT32,
                                           cudf::type_id::FLOAT64,
                                           cudf::type_id::BOOL8,
                                           cudf::type_id::TIMESTAMP_SECONDS,
                                           cudf::type_id::TIMESTAMP_MILLISECONDS,
                                           cudf::type_id::TIMESTAMP_MICROSECONDS,
                                           cudf::type_id::TIMESTAMP_NANOSECONDS,
                                           cudf::type_id::DURATION_SECONDS,
                                           cudf::type_id::DURATION_MILLISECONDS,
                                           cudf::type_id::DURATION_MICROSECONDS,
                                           cudf::type_id::DURATION_NANOSECONDS,
                                           cudf::type_id::STRING,
                                           cudf::type_id::LIST,
                                           cudf::type_id::DECIMAL32,
                                           cudf::type_id::DECIMAL64,
                                           cudf::type_id::DECIMAL128,
                                           cudf::type_id::STRUCT>;

static char const* stringify_type(cudf::type_id value)
{
  switch (value) {
    case cudf::type_id::INT8: return "INT8";
    case cudf::type_id::INT16: return "INT16";
    case cudf::type_id::INT32: return "INT32";
    case cudf::type_id::INT64: return "INT64";
    case cudf::type_id::UINT8: return "UINT8";
    case cudf::type_id::UINT16: return "UINT16";
    case cudf::type_id::UINT32: return "UINT32";
    case cudf::type_id::UINT64: return "UINT64";
    case cudf::type_id::FLOAT32: return "FLOAT32";
    case cudf::type_id::FLOAT64: return "FLOAT64";
    case cudf::type_id::BOOL8: return "BOOL8";
    case cudf::type_id::TIMESTAMP_DAYS: return "TIMESTAMP_DAYS";
    case cudf::type_id::TIMESTAMP_SECONDS: return "TIMESTAMP_SECONDS";
    case cudf::type_id::TIMESTAMP_MILLISECONDS: return "TIMESTAMP_MILLISECONDS";
    case cudf::type_id::TIMESTAMP_MICROSECONDS: return "TIMESTAMP_MICROSECONDS";
    case cudf::type_id::TIMESTAMP_NANOSECONDS: return "TIMESTAMP_NANOSECONDS";
    case cudf::type_id::DURATION_DAYS: return "DURATION_DAYS";
    case cudf::type_id::DURATION_SECONDS: return "DURATION_SECONDS";
    case cudf::type_id::DURATION_MILLISECONDS: return "DURATION_MILLISECONDS";
    case cudf::type_id::DURATION_MICROSECONDS: return "DURATION_MICROSECONDS";
    case cudf::type_id::DURATION_NANOSECONDS: return "DURATION_NANOSECONDS";
    case cudf::type_id::DICTIONARY32: return "DICTIONARY32";
    case cudf::type_id::STRING: return "STRING";
    case cudf::type_id::LIST: return "LIST";
    case cudf::type_id::DECIMAL32: return "DECIMAL32";
    case cudf::type_id::DECIMAL64: return "DECIMAL64";
    case cudf::type_id::DECIMAL128: return "DECIMAL128";
    case cudf::type_id::STRUCT: return "STRUCT";
    default: return "unknown";
  }
}

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(cudf::type_id, stringify_type, stringify_type)

NVBENCH_BENCH_TYPES(BM_to_arrow_host, NVBENCH_TYPE_AXES(data_types))
  .set_type_axes_names({"data_type"})
  .set_name("to_arrow_host")
  .add_int64_axis("num_rows", {10'000, 100'000, 1'000'000, 10'000'000})
  .add_int64_axis("num_columns", {1});

NVBENCH_BENCH_TYPES(BM_to_arrow_device, NVBENCH_TYPE_AXES(data_types))
  .set_type_axes_names({"data_type"})
  .set_name("to_arrow_device")
  .add_int64_axis("num_rows", {10'000, 100'000, 1'000'000, 10'000'000})
  .add_int64_axis("num_columns", {1});

NVBENCH_BENCH_TYPES(BM_from_arrow_host, NVBENCH_TYPE_AXES(data_types))
  .set_type_axes_names({"data_type"})
  .set_name("from_arrow_host")
  .add_int64_axis("num_rows", {10'000, 100'000, 1'000'000, 10'000'000})
  .add_int64_axis("num_columns", {1});

NVBENCH_BENCH_TYPES(BM_from_arrow_device, NVBENCH_TYPE_AXES(data_types))
  .set_type_axes_names({"data_type"})
  .set_name("from_arrow_device")
  .add_int64_axis("num_rows", {10'000, 100'000, 1'000'000, 10'000'000})
  .add_int64_axis("num_columns", {1});
