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

#include <string>
#include <utility>
#include <vector>

#include <cudf/cudf.h>

namespace cudf {
/*
 * @brief Convert dtype strings into gdf_dtype enum;
 *
 * Returns GDF_invalid if the input string is not a valid dtype string
 */
std::pair<gdf_dtype, gdf_dtype_extra_info> convertStringToDtype(const std::string &dtype);

/*
 * @brief Returns the compression type from the incoming compression parameter
 * or inferring from the specified filename
 *
 * @param[in] compression_arg User-supplied string specifying compression type
 * @param[in] filename Name of the file to infer if necessary
 * @param[in] ext_to_compression Mapping between file extensions and compression
 *
 * @return std::string Compression type
 */
std::string infer_compression_type(
  const std::string &compression_arg,
  const std::string &filename,
  const std::vector<std::pair<std::string, std::string>> &ext_to_comp_map);

}  // namespace cudf
