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

std::string infer_compression_type(
    const std::string &compression_arg, const std::string &filename,
    const std::vector<std::pair<std::string, std::string>> &ext_to_comp_map) {
  auto str_tolower = [](const auto &begin, const auto &end) {
    std::string out;
    std::transform(begin, end, std::back_inserter(out), ::tolower);
    return out;
  };

  // Attempt to infer from user-supplied argument
  const auto arg = str_tolower(compression_arg.begin(), compression_arg.end());
  if (arg != "infer") {
    return arg;
  }

  // Attempt to infer from the file extension
  const auto pos = filename.find_last_of('.');
  if (pos != std::string::npos) {
    const auto ext = str_tolower(filename.begin() + pos + 1, filename.end());
    for (const auto &mapping : ext_to_comp_map) {
      if (mapping.first == ext) {
        return mapping.second;
      }
    }
  }

  return "none";
}

}  // namespace cudf
