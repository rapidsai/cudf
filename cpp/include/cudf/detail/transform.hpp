/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/transform.hpp>
#include <arrow/api.h>
#include <string>
#include <cudf/utilities/error.hpp>

namespace cudf {
namespace detail {
/**
 * @copydoc cudf::transform
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 **/
std::unique_ptr<column> transform(
  column_view const& input,
  std::string const& unary_udf,
  data_type output_type,
  bool is_ptx,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0);

/**
 * @copydoc cudf::nans_to_nulls
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 **/
std::pair<std::unique_ptr<rmm::device_buffer>, size_type> nans_to_nulls(
  column_view const& input,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0);

/**
 * @copydoc cudf::bools_to_mask
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 **/
std::pair<std::unique_ptr<rmm::device_buffer>, cudf::size_type> bools_to_mask(
  column_view const& input,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0);


std::shared_ptr<arrow::Table> to_arrow(table_view input,
                                       std::vector<std::string> const& column_names = {},
                                       arrow::MemoryPool* ar_mr = arrow::default_memory_pool(),
                                       cudaStream_t stream                 = 0);

template<typename... Ts>
std::shared_ptr<arrow::Array> to_arrow_array(cudf::type_id id, Ts&&... args) {
  switch(id) {
    case type_id::BOOL8:
      return std::make_shared<arrow::BooleanArray>(std::forward<Ts>(args)...);
    case type_id::INT8:
      return std::make_shared<arrow::Int8Array>(std::forward<Ts>(args)...);
    case type_id::INT16:
      return std::make_shared<arrow::Int16Array>(std::forward<Ts>(args)...);
    case type_id::INT32:
      return std::make_shared<arrow::Int32Array>(std::forward<Ts>(args)...);
    case type_id::INT64:
      return std::make_shared<arrow::Int64Array>(std::forward<Ts>(args)...);
    case type_id::UINT8:
      return std::make_shared<arrow::UInt8Array>(std::forward<Ts>(args)...);
    case type_id::UINT16:
      return std::make_shared<arrow::UInt16Array>(std::forward<Ts>(args)...);
    case type_id::UINT32:
      return std::make_shared<arrow::UInt32Array>(std::forward<Ts>(args)...);
    case type_id::UINT64:
      return std::make_shared<arrow::UInt64Array>(std::forward<Ts>(args)...);
    case type_id::FLOAT32:
      return std::make_shared<arrow::FloatArray>(std::forward<Ts>(args)...);
    case type_id::FLOAT64:
      return std::make_shared<arrow::DoubleArray>(std::forward<Ts>(args)...);
    case type_id::TIMESTAMP_DAYS:
      return std::make_shared<arrow::Date32Array>(std::forward<Ts>(args)...);
    case type_id::TIMESTAMP_MILLISECONDS:
      return std::make_shared<arrow::Date64Array>(std::forward<Ts>(args)...);
//    case type_id::STRING:
//      return std::make_shared<arrow::StringArray>(std::forward<Ts>(args)...);
//    case type_id::DICTIONARY32:
//      return std::make_shared<arrow::DictionaryArray>(std::forward<Ts>(args)...);
//    case type_id::LIST:
//      return std::make_shared<arrow::ListArray>(std::forward<Ts>(args)...);
    default:
      CUDF_FAIL("Unsupported type_id conversion to arrow");
}
}
}  // namespace detail
}  // namespace cudf
