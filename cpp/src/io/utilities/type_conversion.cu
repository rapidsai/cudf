/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/cudf.h>

namespace cudf {
namespace experimental {
namespace io {
/**
 * @copydoc cudf::experimental::io:convert_string_to_dtype
 *
 **/
data_type convert_string_to_dtype(const std::string &dtype)
{
  if (dtype == "str") return data_type(cudf::type_id::STRING);
  if (dtype == "timestamp[s]") return data_type(cudf::type_id::TIMESTAMP_SECONDS);
  // backwards compat: "timestamp" defaults to milliseconds
  if (dtype == "timestamp[ms]" || dtype == "timestamp")
    return data_type(cudf::type_id::TIMESTAMP_MILLISECONDS);
  if (dtype == "timestamp[us]") return data_type(cudf::type_id::TIMESTAMP_MICROSECONDS);
  if (dtype == "timestamp[ns]") return data_type(cudf::type_id::TIMESTAMP_NANOSECONDS);
  if (dtype == "date32") return data_type(cudf::type_id::TIMESTAMP_DAYS);
  if (dtype == "bool" || dtype == "boolean") return data_type(cudf::type_id::BOOL8);
  if (dtype == "date" || dtype == "date64") return data_type(cudf::type_id::TIMESTAMP_MILLISECONDS);
  if (dtype == "float" || dtype == "float32") return data_type(cudf::type_id::FLOAT32);
  if (dtype == "double" || dtype == "float64") return data_type(cudf::type_id::FLOAT64);
  if (dtype == "byte" || dtype == "int8") return data_type(cudf::type_id::INT8);
  if (dtype == "short" || dtype == "int16") return data_type(cudf::type_id::INT16);
  if (dtype == "int" || dtype == "int32") return data_type(cudf::type_id::INT32);
  if (dtype == "long" || dtype == "int64") return data_type(cudf::type_id::INT64);

  return data_type(cudf::type_id::EMPTY);
}

}  // namespace io
}  // namespace experimental
}  // namespace cudf