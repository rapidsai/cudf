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

#include <map>
#include <string>
#include <vector>

#include <cudf/cudf.h>

namespace cudf {

/*
 * @brief Convert dtype strings into gdf_dtype enum;
 *
 * Returns GDF_invalid if the input string is not a valid dtype string
 */
gdf_dtype convertStringToDtype(const std::string &dtype);

/**---------------------------------------------------------------------------*
 * @brief Infer the compression type from the compression parameter and
 * the input file extension.
 *
 * Returns "none" if the input is not compressed.
 * Throws if the input is not valid.
 *
 * @param[in] compression_arg Input string that is potentially describing
 * the compression type. Can also be "none" or "infer".
 * @param[in] source_type Enum describing the type of the data source.
 * @param[in] source If source_type is FILE_PATH, contains the filepath.
 * If source_type is HOST_BUFFER, contains the input data.
 * @param[in] ext_to_compression Map between file extensions and
 * compression types.
 *
 * @return string representing the compression type.
 *---------------------------------------------------------------------------**/
std::string inferCompressionType(const std::string &compression_arg, gdf_input_type source_type,
                                 const std::string &source,
                                 const std::map<std::string, std::string> &ext_to_compression);

} // namespace cudf
