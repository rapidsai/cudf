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

#include <utility>
#include <algorithm>

#include <utilities/error_utils.hpp>

namespace cudf {

std::pair<gdf_dtype, gdf_dtype_extra_info> convertStringToDtype(const std::string &dtype) {
  if (dtype == "str")
    return std::make_pair(GDF_STRING, gdf_dtype_extra_info{ TIME_UNIT_NONE });
  if (dtype == "timestamp[s]")
    return std::make_pair(GDF_TIMESTAMP, gdf_dtype_extra_info{ TIME_UNIT_s });
  // backwards compat: "timestamp" defaults to milliseconds
  if (dtype == "timestamp[ms]" || dtype == "timestamp")
    return std::make_pair(GDF_TIMESTAMP, gdf_dtype_extra_info{ TIME_UNIT_ms });
  if (dtype == "timestamp[us]")
    return std::make_pair(GDF_TIMESTAMP, gdf_dtype_extra_info{ TIME_UNIT_us });
  if (dtype == "timestamp[ns]")
    return std::make_pair(GDF_TIMESTAMP, gdf_dtype_extra_info{ TIME_UNIT_ns });
  if (dtype == "category")
    return std::make_pair(GDF_CATEGORY, gdf_dtype_extra_info{ TIME_UNIT_NONE });
  if (dtype == "date32")
    return std::make_pair(GDF_DATE32, gdf_dtype_extra_info{ TIME_UNIT_NONE });
  if (dtype == "bool" || dtype == "boolean")
    return std::make_pair(GDF_BOOL8, gdf_dtype_extra_info{ TIME_UNIT_NONE });
  if (dtype == "date" || dtype == "date64")
    return std::make_pair(GDF_DATE64, gdf_dtype_extra_info{ TIME_UNIT_NONE });
  if (dtype == "float" || dtype == "float32")
    return std::make_pair(GDF_FLOAT32, gdf_dtype_extra_info{ TIME_UNIT_NONE });
  if (dtype == "double" || dtype == "float64")
    return std::make_pair(GDF_FLOAT64, gdf_dtype_extra_info{ TIME_UNIT_NONE });
  if (dtype == "byte" || dtype == "int8")
    return std::make_pair(GDF_INT8, gdf_dtype_extra_info{ TIME_UNIT_NONE });
  if (dtype == "short" || dtype == "int16")
    return std::make_pair(GDF_INT16, gdf_dtype_extra_info{ TIME_UNIT_NONE });
  if (dtype == "int" || dtype == "int32")
    return std::make_pair(GDF_INT32, gdf_dtype_extra_info{ TIME_UNIT_NONE });
  if (dtype == "long" || dtype == "int64")
    return std::make_pair(GDF_INT64, gdf_dtype_extra_info{ TIME_UNIT_NONE });

  return std::make_pair(GDF_invalid, gdf_dtype_extra_info{ TIME_UNIT_NONE });
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
