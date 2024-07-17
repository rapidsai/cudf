/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include "arrow_utilities.hpp"

#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <nanoarrow/nanoarrow.h>

namespace cudf {
namespace detail {
data_type arrow_to_cudf_type(ArrowSchemaView const* arrow_view)
{
  switch (arrow_view->type) {
    case NANOARROW_TYPE_NA: return data_type(type_id::EMPTY);
    case NANOARROW_TYPE_BOOL: return data_type(type_id::BOOL8);
    case NANOARROW_TYPE_INT8: return data_type(type_id::INT8);
    case NANOARROW_TYPE_INT16: return data_type(type_id::INT16);
    case NANOARROW_TYPE_INT32: return data_type(type_id::INT32);
    case NANOARROW_TYPE_INT64: return data_type(type_id::INT64);
    case NANOARROW_TYPE_UINT8: return data_type(type_id::UINT8);
    case NANOARROW_TYPE_UINT16: return data_type(type_id::UINT16);
    case NANOARROW_TYPE_UINT32: return data_type(type_id::UINT32);
    case NANOARROW_TYPE_UINT64: return data_type(type_id::UINT64);
    case NANOARROW_TYPE_FLOAT: return data_type(type_id::FLOAT32);
    case NANOARROW_TYPE_DOUBLE: return data_type(type_id::FLOAT64);
    case NANOARROW_TYPE_DATE32: return data_type(type_id::TIMESTAMP_DAYS);
    case NANOARROW_TYPE_STRING:
    case NANOARROW_TYPE_LARGE_STRING: return data_type(type_id::STRING);
    case NANOARROW_TYPE_LIST: return data_type(type_id::LIST);
    case NANOARROW_TYPE_DICTIONARY: return data_type(type_id::DICTIONARY32);
    case NANOARROW_TYPE_STRUCT: return data_type(type_id::STRUCT);
    case NANOARROW_TYPE_TIMESTAMP: {
      switch (arrow_view->time_unit) {
        case NANOARROW_TIME_UNIT_SECOND: return data_type(type_id::TIMESTAMP_SECONDS);
        case NANOARROW_TIME_UNIT_MILLI: return data_type(type_id::TIMESTAMP_MILLISECONDS);
        case NANOARROW_TIME_UNIT_MICRO: return data_type(type_id::TIMESTAMP_MICROSECONDS);
        case NANOARROW_TIME_UNIT_NANO: return data_type(type_id::TIMESTAMP_NANOSECONDS);
        default: CUDF_FAIL("Unsupported timestamp unit in arrow", cudf::data_type_error);
      }
    }
    case NANOARROW_TYPE_DURATION: {
      switch (arrow_view->time_unit) {
        case NANOARROW_TIME_UNIT_SECOND: return data_type(type_id::DURATION_SECONDS);
        case NANOARROW_TIME_UNIT_MILLI: return data_type(type_id::DURATION_MILLISECONDS);
        case NANOARROW_TIME_UNIT_MICRO: return data_type(type_id::DURATION_MICROSECONDS);
        case NANOARROW_TIME_UNIT_NANO: return data_type(type_id::DURATION_NANOSECONDS);
        default: CUDF_FAIL("Unsupported duration unit in arrow", cudf::data_type_error);
      }
    }
    case NANOARROW_TYPE_DECIMAL128:
      return data_type{type_id::DECIMAL128, -arrow_view->decimal_scale};
    default: CUDF_FAIL("Unsupported type_id conversion to cudf", cudf::data_type_error);
  }
}

ArrowType id_to_arrow_type(cudf::type_id id)
{
  switch (id) {
    case cudf::type_id::BOOL8: return NANOARROW_TYPE_BOOL;
    case cudf::type_id::INT8: return NANOARROW_TYPE_INT8;
    case cudf::type_id::INT16: return NANOARROW_TYPE_INT16;
    case cudf::type_id::INT32: return NANOARROW_TYPE_INT32;
    case cudf::type_id::INT64: return NANOARROW_TYPE_INT64;
    case cudf::type_id::UINT8: return NANOARROW_TYPE_UINT8;
    case cudf::type_id::UINT16: return NANOARROW_TYPE_UINT16;
    case cudf::type_id::UINT32: return NANOARROW_TYPE_UINT32;
    case cudf::type_id::UINT64: return NANOARROW_TYPE_UINT64;
    case cudf::type_id::FLOAT32: return NANOARROW_TYPE_FLOAT;
    case cudf::type_id::FLOAT64: return NANOARROW_TYPE_DOUBLE;
    case cudf::type_id::TIMESTAMP_DAYS: return NANOARROW_TYPE_DATE32;
    default: CUDF_FAIL("Unsupported type_id conversion to arrow type", cudf::data_type_error);
  }
}

}  // namespace detail
}  // namespace cudf
