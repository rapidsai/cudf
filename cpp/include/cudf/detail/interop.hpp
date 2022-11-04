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

#pragma once

// We disable warning 611 because the `arrow::TableBatchReader` only partially
// override the `ReadNext` method of `arrow::RecordBatchReader::ReadNext`
// triggering warning 611-D from nvcc.
#pragma nv_diag_suppress 611
#include <arrow/api.h>
#pragma nv_diag_default 611

#include <cudf/interop.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <string>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {

/**
 * @copydoc cudf::from_dlpack
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> from_dlpack(
  DLManagedTensor const* managed_tensor,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @copydoc cudf::to_dlpack
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
DLManagedTensor* to_dlpack(
  table_view const& input,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

// Creating arrow as per given type_id and buffer arguments
template <typename... Ts>
std::shared_ptr<arrow::Array> to_arrow_array(cudf::type_id id, Ts&&... args)
{
  switch (id) {
    case type_id::BOOL8: return std::make_shared<arrow::BooleanArray>(std::forward<Ts>(args)...);
    case type_id::INT8: return std::make_shared<arrow::Int8Array>(std::forward<Ts>(args)...);
    case type_id::INT16: return std::make_shared<arrow::Int16Array>(std::forward<Ts>(args)...);
    case type_id::INT32: return std::make_shared<arrow::Int32Array>(std::forward<Ts>(args)...);
    case type_id::INT64: return std::make_shared<arrow::Int64Array>(std::forward<Ts>(args)...);
    case type_id::UINT8: return std::make_shared<arrow::UInt8Array>(std::forward<Ts>(args)...);
    case type_id::UINT16: return std::make_shared<arrow::UInt16Array>(std::forward<Ts>(args)...);
    case type_id::UINT32: return std::make_shared<arrow::UInt32Array>(std::forward<Ts>(args)...);
    case type_id::UINT64: return std::make_shared<arrow::UInt64Array>(std::forward<Ts>(args)...);
    case type_id::FLOAT32: return std::make_shared<arrow::FloatArray>(std::forward<Ts>(args)...);
    case type_id::FLOAT64: return std::make_shared<arrow::DoubleArray>(std::forward<Ts>(args)...);
    case type_id::TIMESTAMP_DAYS:
      return std::make_shared<arrow::Date32Array>(std::make_shared<arrow::Date32Type>(),
                                                  std::forward<Ts>(args)...);
    case type_id::TIMESTAMP_SECONDS:
      return std::make_shared<arrow::TimestampArray>(arrow::timestamp(arrow::TimeUnit::SECOND),
                                                     std::forward<Ts>(args)...);
    case type_id::TIMESTAMP_MILLISECONDS:
      return std::make_shared<arrow::TimestampArray>(arrow::timestamp(arrow::TimeUnit::MILLI),
                                                     std::forward<Ts>(args)...);
    case type_id::TIMESTAMP_MICROSECONDS:
      return std::make_shared<arrow::TimestampArray>(arrow::timestamp(arrow::TimeUnit::MICRO),
                                                     std::forward<Ts>(args)...);
    case type_id::TIMESTAMP_NANOSECONDS:
      return std::make_shared<arrow::TimestampArray>(arrow::timestamp(arrow::TimeUnit::NANO),
                                                     std::forward<Ts>(args)...);
    case type_id::DURATION_SECONDS:
      return std::make_shared<arrow::DurationArray>(arrow::duration(arrow::TimeUnit::SECOND),
                                                    std::forward<Ts>(args)...);
    case type_id::DURATION_MILLISECONDS:
      return std::make_shared<arrow::DurationArray>(arrow::duration(arrow::TimeUnit::MILLI),
                                                    std::forward<Ts>(args)...);
    case type_id::DURATION_MICROSECONDS:
      return std::make_shared<arrow::DurationArray>(arrow::duration(arrow::TimeUnit::MICRO),
                                                    std::forward<Ts>(args)...);
    case type_id::DURATION_NANOSECONDS:
      return std::make_shared<arrow::DurationArray>(arrow::duration(arrow::TimeUnit::NANO),
                                                    std::forward<Ts>(args)...);
    default: CUDF_FAIL("Unsupported type_id conversion to arrow");
  }
}

// Converting arrow type to cudf type
data_type arrow_to_cudf_type(arrow::DataType const& arrow_type);

/**
 * @copydoc cudf::to_arrow
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::shared_ptr<arrow::Table> to_arrow(table_view input,
                                       std::vector<column_metadata> const& metadata = {},
                                       rmm::cuda_stream_view stream = cudf::get_default_stream(),
                                       arrow::MemoryPool* ar_mr     = arrow::default_memory_pool());

/**
 * @copydoc cudf::arrow_to_cudf
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> from_arrow(
  arrow::Table const& input_table,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace detail
}  // namespace cudf
