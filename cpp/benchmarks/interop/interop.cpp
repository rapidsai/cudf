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

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <vector>

enum class System : uint8_t { Host = 0, Device = 1 };

template <cudf::type_id data_type, System dst>
void BM_to_arrow(nvbench::state& state, nvbench::type_list<nvbench::enum_type<data_type>>)
{
  auto const num_rows       = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  int32_t const num_columns = 1;
  auto const num_elements   = static_cast<int64_t>(num_rows) * num_columns;

  std::vector<cudf::type_id> types;

  std::fill_n(std::back_inserter(types), num_columns, data_type);

  auto const source_table  = create_random_table(types, row_count{num_rows});
  int64_t const size_bytes = estimate_size(source_table->view());

  state.add_element_count(num_elements, "num_elements");

  state.add_global_memory_reads(size_bytes);
  state.add_global_memory_writes(size_bytes);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    if constexpr (dst == Destination::Host) {
      cudf::to_arrow_device(source_table->view(), rmm::cuda_stream_view{launch.get_stream()});
    } else {
      cudf::to_arrow_host(source_table->view(), rmm::cuda_stream_view{launch.get_stream()});
    }
  });
}

struct ArrowTypeInfo {
  ArrowType type                         = NANOARROW_TYPE_UNINITIALIZED;
  std::optional<ArrowTimeUnit> time_unit = std::nullopt;
  std::optional<int32_t> precision       = std::nullopt;
};

static ArrowTypeInfo to_arrow_type(cudf::type_id dtype)
{
  switch (dtype) {
    case cudf::type_id::INT8: return {NANOARROW_TYPE_INT8};
    case cudf::type_id::INT16: return {NANOARROW_TYPE_INT16};
    case cudf::type_id::INT32: return {NANOARROW_TYPE_INT32};
    case cudf::type_id::INT64: return {NANOARROW_TYPE_INT64};
    case cudf::type_id::UINT8: return {NANOARROW_TYPE_UINT8};
    case cudf::type_id::UINT16: return {NANOARROW_TYPE_UINT16};
    case cudf::type_id::UINT32: return {NANOARROW_TYPE_UINT32};
    case cudf::type_id::UINT64: return {NANOARROW_TYPE_UINT64};
    case cudf::type_id::FLOAT32: return {NANOARROW_TYPE_FLOAT};
    case cudf::type_id::FLOAT64: return {NANOARROW_TYPE_DOUBLE};
    case cudf::type_id::BOOL8: return {NANOARROW_TYPE_BOOL};
    case cudf::type_id::TIMESTAMP_SECONDS:
      return {NANOARROW_TYPE_TIMESTAMP, NANOARROW_TIME_UNIT_SECOND};
    case cudf::type_id::TIMESTAMP_MILLISECONDS:
      return {NANOARROW_TYPE_TIMESTAMP, NANOARROW_TIME_UNIT_MILLI};
    case cudf::type_id::TIMESTAMP_MICROSECONDS:
      return {NANOARROW_TYPE_TIMESTAMP, NANOARROW_TIME_UNIT_MICRO};
    case cudf::type_id::TIMESTAMP_NANOSECONDS:
      return {NANOARROW_TYPE_TIMESTAMP, NANOARROW_TIME_UNIT_NANO};
    case cudf::type_id::DURATION_SECONDS:
      return {NANOARROW_TYPE_DURATION, NANOARROW_TIME_UNIT_SECOND};
    case cudf::type_id::DURATION_MILLISECONDS:
      return {NANOARROW_TYPE_DURATION, NANOARROW_TIME_UNIT_MILLI};
    case cudf::type_id::DURATION_MICROSECONDS:
      return {NANOARROW_TYPE_DURATION, NANOARROW_TIME_UNIT_MICRO};
    case cudf::type_id::DURATION_NANOSECONDS:
      return {NANOARROW_TYPE_DURATION, NANOARROW_TIME_UNIT_NANO};
    case cudf::type_id::STRING: return {NANOARROW_TYPE_STRING};
    case cudf::type_id::LIST: return {NANOARROW_TYPE_LIST};
    case cudf::type_id::DECIMAL32:
      return {NANOARROW_TYPE_DECIMAL, std::nullopt, cudf::detail::max_precision<int32_t>()};
    case cudf::type_id::DECIMAL64:
      return {NANOARROW_TYPE_DECIMAL, std::nullopt, cudf::detail::max_precision<int64_t>()};
    case cudf::type_id::DECIMAL128:
      return {NANOARROW_TYPE_DECIMAL128, std::nullopt, cudf::detail::max_precision<__int128_t>()};
    case cudf::type_id::STRUCT: return {NANOARROW_TYPE_STRUCT};
    default: CUDF_FAIL("Unimplemented data type");
  }
}

static void set_arrow_type(ArrowSchema& schema, cudf::type_id dtype)
{
  ArrowTypeInfo info = to_arrow_type(dtype);
  if (!info.precision.has_value() && !info.time_unit.has_value()) {
    NANOARROW_THROW_NOT_OK(ArrowSchemaSetType(&schema, info.type));
    return;
  }

  if (info.precision.has_value()) {
    NANOARROW_THROW_NOT_OK(
      ArrowSchemaSetTypeDecimal(&schema, info.type, info.precision.value(), 1));
    return;
  }

  if (info.time_unit.has_value()) {
    NANOARROW_THROW_NOT_OK(
      ArrowSchemaSetTypeDateTime(&schema, info.type, info.time_unit.value(), nullptr));
    return;
  }

  CUDF_UNREACHABLE("Unexpected Error");
}

template <cudf::type_id data_type, System src>
void BM_from_arrow(nvbench::state& state, nvbench::type_list<nvbench::enum_type<data_type>>)
{
  auto const num_rows     = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_columns  = static_cast<cudf::size_type>(state.get_int64("num_columns"));
  auto const num_elements = static_cast<int64_t>(num_rows) * num_columns;

  nanoarrow::UniqueSchema input_schema;
  ArrowSchemaInit(input_schema.get());
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(input_schema.get(), 1));
  ArrowSchemaInit(input_schema->children[0]);
  set_arrow_type(*input->children[0], data_type);
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(input_schema->children[0], "a"));

  nanoarrow::UniqueArray input_array;
  std::unique_ptr<cudf::column> cudf_column;
  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromSchema(input_array.get(), input_schema.get(), nullptr));
  input_array->length                  = num_rows;
  input_array->null_count              = 0;
  input_array->offset                  = 0;
  input_array->children[0]->length     = num_rows;
  input_array->children[0]->null_count = 0;

  ArrowDeviceArray input_device_array;
  input_device_array.array      = *input_array;
  input_device_array.sync_event = nullptr;

  if constexpr (src == System::Device) {
    cudf_column = create_random_column(data_type, row_count{num_rows});
    NANOARROW_THROW_NOT_OK(
      ArrowBufferSetAllocator(ArrowArrayBuffer(input_array->children[0], 1), noop_alloc));
    ArrowArrayBuffer(input_array->children[0], 1)->data =
      const_cast<uint8_t*>(cudf_column->get_data());
    ArrowArrayBuffer(input_array->children[0], 1)->size_bytes = sizeof(T) * data_size;

    NANOARROW_THROW_NOT_OK(
      ArrowArrayFinishBuilding(input_array.get(), NANOARROW_VALIDATION_LEVEL_MINIMAL, nullptr));

    input_device_array.device_id   = rmm::get_current_cuda_device().value();
    input_device_array.device_type = ARROW_DEVICE_CUDA;
  } else {
    input_device_array.device_id   = -1;
    input_device_array.device_type = ARROW_DEVICE_CPU;
  }

  state.add_element_count(num_elements, "num_elements");
  state.add_global_memory_reads(size_bytes);
  state.add_global_memory_writes(size_bytes);

  // must use stream

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    if constexpr (src == System::Device) {
      cudf::from_arrow_device(
        input_schema.get(), &input_device_array, rmm::cuda_stream_view{launch.get_stream()});
    } else {
      cudf::from_arrow_host(
        input_schema.get(), &input_device_array, rmm::cuda_stream_view{launch.get_stream()});
    }
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
                                           cudf::type_id::TIMESTAMP_DAYS,
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

using systems = nvbench::enum_type_list<System::Device, System::Host>;

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

static char const* stringify_system(System sys)
{
  switch (sys) {
    case System::Device: return "Device";
    case System::Host: return "Host";
    default: return "unknown";
  }
}

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(cudf::type_id, stringify_type, stringify_type)
NVBENCH_DECLARE_ENUM_TYPE_STRINGS(System, stringify_system, stringify_system)

NVBENCH_BENCH_TYPES(BM_to_arrow, NVBENCH_TYPE_AXES(data_types), NVBENCH_TYPE_AXES(systems))
  .set_type_axes_names({"data_type", "system"})
  .set_name("to_arrow")
  .add_int64_axis("num_rows", {10'000, 100'000, 1'000'000, 10'000'000});
