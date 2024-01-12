/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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
#ifdef __CUDACC__
#pragma nv_diag_suppress 611
#pragma nv_diag_suppress 2810
#endif
#include <arrow/api.h>
#ifdef __CUDACC__
#pragma nv_diag_default 611
#pragma nv_diag_default 2810
#endif

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
std::unique_ptr<table> from_dlpack(DLManagedTensor const* managed_tensor,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr);

/**
 * @copydoc cudf::to_dlpack
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
DLManagedTensor* to_dlpack(table_view const& input,
                           rmm::cuda_stream_view stream,
                           rmm::mr::device_memory_resource* mr);

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

/**
 * @brief Invokes an `operator()` template with the type instantiation based on
 * the specified `arrow::DataType`'s `id()`.
 *
 * This function is analogous to libcudf's type_dispatcher, but instead applies
 * to Arrow functions. Its primary use case is to leverage Arrow's
 * metaprogramming facilities like arrow::TypeTraits that require translating
 * the runtime dtype information into compile-time types.
 */
template <typename Functor, typename... Ts>
constexpr decltype(auto) arrow_type_dispatcher(arrow::DataType const& dtype,
                                               Functor f,
                                               Ts&&... args)
{
  switch (dtype.id()) {
    case arrow::Type::INT8:
      return f.template operator()<arrow::Int8Type>(std::forward<Ts>(args)...);
    case arrow::Type::INT16:
      return f.template operator()<arrow::Int16Type>(std::forward<Ts>(args)...);
    case arrow::Type::INT32:
      return f.template operator()<arrow::Int32Type>(std::forward<Ts>(args)...);
    case arrow::Type::INT64:
      return f.template operator()<arrow::Int64Type>(std::forward<Ts>(args)...);
    case arrow::Type::UINT8:
      return f.template operator()<arrow::UInt8Type>(std::forward<Ts>(args)...);
    case arrow::Type::UINT16:
      return f.template operator()<arrow::UInt16Type>(std::forward<Ts>(args)...);
    case arrow::Type::UINT32:
      return f.template operator()<arrow::UInt32Type>(std::forward<Ts>(args)...);
    case arrow::Type::UINT64:
      return f.template operator()<arrow::UInt64Type>(std::forward<Ts>(args)...);
    case arrow::Type::FLOAT:
      return f.template operator()<arrow::FloatType>(std::forward<Ts>(args)...);
    case arrow::Type::DOUBLE:
      return f.template operator()<arrow::DoubleType>(std::forward<Ts>(args)...);
    case arrow::Type::BOOL:
      return f.template operator()<arrow::BooleanType>(std::forward<Ts>(args)...);
    case arrow::Type::TIMESTAMP:
      return f.template operator()<arrow::TimestampType>(std::forward<Ts>(args)...);
    case arrow::Type::DURATION:
      return f.template operator()<arrow::DurationType>(std::forward<Ts>(args)...);
    case arrow::Type::STRING:
      return f.template operator()<arrow::StringType>(std::forward<Ts>(args)...);
    case arrow::Type::LIST:
      return f.template operator()<arrow::ListType>(std::forward<Ts>(args)...);
    case arrow::Type::DECIMAL128:
      return f.template operator()<arrow::Decimal128Type>(std::forward<Ts>(args)...);
    case arrow::Type::STRUCT:
      return f.template operator()<arrow::StructType>(std::forward<Ts>(args)...);
    default: {
      CUDF_FAIL("Invalid type.");
    }
  }
}

// Converting arrow type to cudf type
data_type arrow_to_cudf_type(arrow::DataType const& arrow_type);

/**
 * @copydoc cudf::to_arrow(table_view input, std::vector<column_metadata> const& metadata,
 * rmm::cuda_stream_view stream, arrow::MemoryPool* ar_mr)
 */
std::shared_ptr<arrow::Table> to_arrow(table_view input,
                                       std::vector<column_metadata> const& metadata,
                                       rmm::cuda_stream_view stream,
                                       arrow::MemoryPool* ar_mr);

/**
 * @copydoc cudf::to_arrow(cudf::scalar const& input, column_metadata const& metadata,
 * rmm::cuda_stream_view stream, arrow::MemoryPool* ar_mr)
 */
std::shared_ptr<arrow::Scalar> to_arrow(cudf::scalar const& input,
                                        column_metadata const& metadata,
                                        rmm::cuda_stream_view stream,
                                        arrow::MemoryPool* ar_mr);
/**
 * @copydoc cudf::from_arrow(arrow::Table const& input_table, rmm::cuda_stream_view stream,
 * rmm::mr::device_memory_resource* mr)
 */
std::unique_ptr<table> from_arrow(arrow::Table const& input_table,
                                  rmm::cuda_stream_view stream,
                                  rmm::mr::device_memory_resource* mr);

/**
 * @copydoc cudf::from_arrow(arrow::Scalar const& input, rmm::cuda_stream_view stream,
 * rmm::mr::device_memory_resource* mr)
 */
std::unique_ptr<cudf::scalar> from_arrow(arrow::Scalar const& input,
                                         rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource* mr);

/**
 * @brief Return a maximum precision for a given type.
 *
 * @tparam T the type to get the maximum precision for
 */
template <typename T>
constexpr std::size_t max_precision()
{
  auto constexpr num_bits = sizeof(T) * 8;
  return std::floor(num_bits * std::log(2) / std::log(10));
}

}  // namespace detail
}  // namespace cudf
