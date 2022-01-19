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

#include <cudf/types.hpp>

#include <algorithm>
#include <cctype>
#include <string>

namespace cudf {
namespace io {
/**
 * @copydoc cudf::io:convert_string_to_dtype
 */
data_type convert_string_to_dtype(const std::string& dtype_in)
{
  // TODO: This function should be cleanup to take only libcudf type instances.
  std::string dtype = dtype_in;
  // first, convert to all lower-case
  std::transform(dtype_in.begin(), dtype_in.end(), dtype.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  if (dtype == "str") return data_type(cudf::type_id::STRING);
  if (dtype == "timestamp[s]" || dtype == "datetime64[s]")
    return data_type(cudf::type_id::TIMESTAMP_SECONDS);
  // backwards compat: "timestamp" defaults to milliseconds
  if (dtype == "timestamp[ms]" || dtype == "timestamp" || dtype == "datetime64[ms]")
    return data_type(cudf::type_id::TIMESTAMP_MILLISECONDS);
  if (dtype == "timestamp[us]" || dtype == "datetime64[us]")
    return data_type(cudf::type_id::TIMESTAMP_MICROSECONDS);
  if (dtype == "timestamp[ns]" || dtype == "datetime64[ns]")
    return data_type(cudf::type_id::TIMESTAMP_NANOSECONDS);
  if (dtype == "date32") return data_type(cudf::type_id::TIMESTAMP_DAYS);
  if (dtype == "bool" || dtype == "boolean") return data_type(cudf::type_id::BOOL8);
  if (dtype == "date" || dtype == "date64") return data_type(cudf::type_id::TIMESTAMP_MILLISECONDS);
  if (dtype == "timedelta[d]") return data_type(cudf::type_id::DURATION_DAYS);
  if (dtype == "timedelta64[s]") return data_type(cudf::type_id::DURATION_SECONDS);
  if (dtype == "timedelta64[ms]") return data_type(cudf::type_id::DURATION_MILLISECONDS);
  if (dtype == "timedelta64[us]") return data_type(cudf::type_id::DURATION_MICROSECONDS);
  if (dtype == "timedelta" || dtype == "timedelta64[ns]")
    return data_type(cudf::type_id::DURATION_NANOSECONDS);
  if (dtype == "float" || dtype == "float32") return data_type(cudf::type_id::FLOAT32);
  if (dtype == "double" || dtype == "float64") return data_type(cudf::type_id::FLOAT64);
  if (dtype == "byte" || dtype == "int8") return data_type(cudf::type_id::INT8);
  if (dtype == "short" || dtype == "int16") return data_type(cudf::type_id::INT16);
  if (dtype == "int" || dtype == "int32") return data_type(cudf::type_id::INT32);
  if (dtype == "long" || dtype == "int64") return data_type(cudf::type_id::INT64);
  if (dtype == "uint8") return data_type(cudf::type_id::UINT8);
  if (dtype == "uint16") return data_type(cudf::type_id::UINT16);
  if (dtype == "uint32") return data_type(cudf::type_id::UINT32);
  if (dtype == "uint64") return data_type(cudf::type_id::UINT64);

  return data_type(cudf::type_id::EMPTY);
}

}  // namespace io
}  // namespace cudf
