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

#include <cudf/detail/interop.hpp>
#include <cudf/interop.hpp>

#include <nanoarrow/nanoarrow.hpp>
#include <nanoarrow/nanoarrow_device.h>
#include <nanoarrow_utils.hpp>
#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <vector>

template <cudf::type_id data_type>
void BM_to_arrow_device(nvbench::state& state, nvbench::type_list<nvbench::enum_type<data_type>>)
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
    cudf::to_arrow_device(source_table->view(), rmm::cuda_stream_view{launch.get_stream()});
  });
}

template <cudf::type_id data_type>
void BM_to_arrow_host(nvbench::state& state, nvbench::type_list<nvbench::enum_type<data_type>>)
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
    cudf::to_arrow_host(source_table->view(), rmm::cuda_stream_view{launch.get_stream()});
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

template <cudf::type_id data_type>
struct nanoarrow_rep {};

#define DEFINE_ARROW_REP(data_type, rep)           \
  template <>                                      \
  struct nanoarrow_rep<cudf::type_id::data_type> { \
    using type = rep;                              \
  };

DEFINE_ARROW_REP(UINT8, uint8_t)
DEFINE_ARROW_REP(UINT16, uint16_t)
DEFINE_ARROW_REP(UINT32, uint32_t)
DEFINE_ARROW_REP(UINT64, uint64_t)
DEFINE_ARROW_REP(INT8, int8_t)
DEFINE_ARROW_REP(INT16, int16_t)
DEFINE_ARROW_REP(INT32, int32_t)
DEFINE_ARROW_REP(INT64, int64_t)
DEFINE_ARROW_REP(FLOAT32, float)
DEFINE_ARROW_REP(FLOAT64, double)
DEFINE_ARROW_REP(BOOL8, bool)
DEFINE_ARROW_REP(TIMESTAMP_DAYS, int64_t)
DEFINE_ARROW_REP(TIMESTAMP_SECONDS, int64_t)
DEFINE_ARROW_REP(TIMESTAMP_MILLISECONDS, int64_t)
DEFINE_ARROW_REP(TIMESTAMP_MICROSECONDS, int64_t)
DEFINE_ARROW_REP(TIMESTAMP_NANOSECONDS, int64_t)
DEFINE_ARROW_REP(DURATION_DAYS, int64_t)
DEFINE_ARROW_REP(DURATION_SECONDS, int64_t)
DEFINE_ARROW_REP(DURATION_MILLISECONDS, int64_t)
DEFINE_ARROW_REP(DURATION_MICROSECONDS, int64_t)
DEFINE_ARROW_REP(DURATION_NANOSECONDS, int64_t)
DEFINE_ARROW_REP(DECIMAL32, int32_t)
DEFINE_ARROW_REP(DECIMAL64, int64_t)
DEFINE_ARROW_REP(DECIMAL128, __int128_t)

template <cudf::type_id data_type>
void BM_from_arrow_device(nvbench::state& state, nvbench::type_list<nvbench::enum_type<data_type>>)
{
  auto const num_rows     = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_columns  = 1;
  auto const num_elements = static_cast<int64_t>(num_rows) * num_columns;

  nanoarrow::UniqueSchema schema;
  ArrowSchemaInit(schema.get());
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(schema.get(), 1));
  ArrowSchemaInit(schema->children[0]);
  set_arrow_type(*schema->children[0], data_type);
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(schema->children[0], "a"));

  nanoarrow::UniqueArray input;

  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromSchema(input.get(), schema.get(), nullptr));

  input->length = num_rows;

  std::unique_ptr<cudf::column> cudf_column = create_random_column(data_type, row_count{num_rows});
  populate_from_col<cudf::id_to_type<data_type>>(input->children[0], cudf_column->view());

  NANOARROW_THROW_NOT_OK(
    ArrowArrayFinishBuilding(input.get(), NANOARROW_VALIDATION_LEVEL_MINIMAL, nullptr));

  ArrowDeviceArray device_array;
  device_array.array       = *input.get();
  device_array.device_id   = rmm::get_current_cuda_device().value();
  device_array.device_type = ARROW_DEVICE_CUDA;
  device_array.sync_event  = nullptr;

  state.add_element_count(num_elements, "num_elements");
  state.add_global_memory_reads<cudf::id_to_type<data_type>>(num_rows);
  state.add_global_memory_writes<cudf::id_to_type<data_type>>(num_rows);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::from_arrow_device_column(
      schema.get(), &device_array, rmm::cuda_stream_view{launch.get_stream()});
  });
}

template <cudf::type_id data_type>
void BM_from_arrow_host(nvbench::state& state, nvbench::type_list<nvbench::enum_type<data_type>>)
{
  auto const num_rows     = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_columns  = 1;
  auto const num_elements = static_cast<int64_t>(num_rows) * num_columns;

  nanoarrow::UniqueSchema schema;
  ArrowSchemaInit(schema.get());
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetTypeStruct(schema.get(), 1));
  ArrowSchemaInit(schema->children[0]);
  set_arrow_type(*schema->children[0], data_type);
  NANOARROW_THROW_NOT_OK(ArrowSchemaSetName(schema->children[0], "a"));

  nanoarrow::UniqueArray input;

  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromSchema(input.get(), schema.get(), nullptr));
  input->length                  = 0;
  input->null_count              = 0;
  input->children[0]->length     = num_rows;
  input->children[0]->null_count = 0;

  std::vector<typename nanoarrow_rep<data_type>::type> rep;
  rep.resize(num_rows, {});

  ArrowBuffer* buffer    = ArrowArrayBuffer(input->children[0], 1);
  buffer->allocator      = noop_alloc;
  buffer->data           = reinterpret_cast<uint8_t*>(rep.data());
  buffer->size_bytes     = rep.size() * sizeof(typename nanoarrow_rep<data_type>::type);
  buffer->capacity_bytes = buffer->size_bytes;

  NANOARROW_THROW_NOT_OK(
    ArrowArrayFinishBuilding(input.get(), NANOARROW_VALIDATION_LEVEL_MINIMAL, nullptr));

  ArrowDeviceArray host_array;
  host_array.array       = *input.get();
  host_array.device_id   = -1;
  host_array.device_type = ARROW_DEVICE_CPU;
  host_array.sync_event  = nullptr;

  state.add_element_count(num_elements, "num_elements");
  state.add_global_memory_reads<typename nanoarrow_rep<data_type>::type>(num_rows);
  state.add_global_memory_writes<cudf::id_to_type<data_type>>(num_rows);

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::from_arrow_host_column(
      schema.get(), &host_array, rmm::cuda_stream_view{launch.get_stream()});
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
                                           cudf::type_id::TIMESTAMP_SECONDS,
                                           cudf::type_id::TIMESTAMP_MILLISECONDS,
                                           cudf::type_id::TIMESTAMP_MICROSECONDS,
                                           cudf::type_id::TIMESTAMP_NANOSECONDS,
                                           cudf::type_id::DURATION_SECONDS,
                                           cudf::type_id::DURATION_MILLISECONDS,
                                           cudf::type_id::DURATION_MICROSECONDS,
                                           cudf::type_id::DURATION_NANOSECONDS,
                                           cudf::type_id::DECIMAL128>;

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
    case cudf::type_id::DECIMAL32: return "DECIMAL32";
    case cudf::type_id::DECIMAL64: return "DECIMAL64";
    case cudf::type_id::DECIMAL128: return "DECIMAL128";
    default: return "unknown";
  }
}

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(cudf::type_id, stringify_type, stringify_type)

NVBENCH_BENCH_TYPES(BM_to_arrow_host, NVBENCH_TYPE_AXES(data_types))
  .set_name("to_arrow_host")
  .add_int64_axis("num_rows", {10'000, 100'000, 1'000'000, 10'000'000});

NVBENCH_BENCH_TYPES(BM_to_arrow_device, NVBENCH_TYPE_AXES(data_types))
  .set_name("to_arrow_device")
  .add_int64_axis("num_rows", {10'000, 100'000, 1'000'000, 10'000'000});

NVBENCH_BENCH_TYPES(BM_from_arrow_host, NVBENCH_TYPE_AXES(data_types))
  .set_name("from_arrow_host")
  .add_int64_axis("num_rows", {10'000, 100'000, 1'000'000, 10'000'000});

NVBENCH_BENCH_TYPES(BM_from_arrow_device, NVBENCH_TYPE_AXES(data_types))
  .set_name("from_arrow_device")
  .add_int64_axis("num_rows", {10'000, 100'000, 1'000'000, 10'000'000});
