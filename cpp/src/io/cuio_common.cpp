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

#include "cuio_common.hpp"

#include <algorithm>

#include <utilities/error_utils.hpp>

namespace cudf {

gdf_dtype convertStringToDtype(const std::string &dtype) {
  if (dtype == "str")
    return GDF_STRING;
  if (dtype == "timestamp")
    return GDF_TIMESTAMP;
  if (dtype == "category")
    return GDF_CATEGORY;
  if (dtype == "date32")
    return GDF_DATE32;
  if (dtype == "bool" || dtype == "boolean")
    return GDF_BOOL8;
  if (dtype == "date" || dtype == "date64")
    return GDF_DATE64;
  if (dtype == "float" || dtype == "float32")
    return GDF_FLOAT32;
  if (dtype == "double" || dtype == "float64")
    return GDF_FLOAT64;
  if (dtype == "byte" || dtype == "int8")
    return GDF_INT8;
  if (dtype == "short" || dtype == "int16")
    return GDF_INT16;
  if (dtype == "int" || dtype == "int32")
    return GDF_INT32;
  if (dtype == "long" || dtype == "int64")
    return GDF_INT64;

  return GDF_invalid;
}

/**---------------------------------------------------------------------------*
 * @brief Infer the compression type from the compression parameter and
 * the input data.
 *
 * Returns "none" if the input is not compressed.
 * Throws if the input is not not valid.
 *
 * @param[in] compression_arg Input string that is potentially describing
 * the compression type. Can also be "none" or "infer".
 * @param[in] source_type Enum describing the type of the data source
 * @param[in] source If source_type is FILE_PATH, contains the filepath.
 * If source_type is HOST_BUFFER, contains the input JSON data.
 *
 * @return string representing the compression type.
 *---------------------------------------------------------------------------**/
std::string inferCompressionType(const std::string &compression_arg, gdf_input_type source_type,
                                 const std::string &source,
                                 const std::map<std::string, std::string> &ext_to_compression) {
  auto str_tolower = [](const auto &begin, const auto &end) {
    std::string out;
    std::transform(begin, end, std::back_inserter(out), ::tolower);
    return out;
  };

  const std::string comp_arg_lower = str_tolower(compression_arg.begin(), compression_arg.end());
  if (comp_arg_lower != "infer")
    return comp_arg_lower;

  // Cannot infer compression type from a buffer, treat as uncompressed
  if (source_type == gdf_input_type::HOST_BUFFER)
    return "none";

  // Need to infer compression from the file extension
  const auto ext_start = std::find(source.rbegin(), source.rend(), '.').base();
  const std::string file_ext = str_tolower(ext_start, source.end());
  if (ext_to_compression.find(file_ext) != ext_to_compression.end())
    return ext_to_compression.find(file_ext)->second;

  // None of the supported compression types match, treat as uncompressed
  return "none";
}

} // namespace cudf
