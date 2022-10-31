/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/io/text/byte_range_info.hpp>
#include <cudf/io/text/data_chunk_source.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/per_device_resource.hpp>

#include <memory>
#include <optional>

namespace cudf {
namespace io {
namespace text {

/**
 * @brief Parsing options for multibyte_split.
 */
struct parse_options {
  /**
   * @brief Only rows starting inside this byte range will be part of the output column.
   */
  byte_range_info byte_range = create_byte_range_info_max();
  /**
   * @brief Whether delimiters at the end of rows should be stripped from the output column
   */
  bool strip_delimiters = false;
};

/**
 * @brief Splits the source text into a strings column using a multiple byte delimiter.
 *
 * Providing a byte range allows multibyte_split to read a file partially, only returning the
 * offsets of delimiters which begin within the range. If thinking in terms of "records", where each
 * delimiter dictates the end of a record, all records which begin within the byte range provided
 * will be returned, including any record which may begin in the range but end outside of the
 * range. Records which begin outside of the range will ignored, even if those records end inside
 * the range.
 *
 * @code{.pseudo}
 * Examples:
 *  source:     "abc..def..ghi..jkl.."
 *  delimiter:  ".."
 *
 *  byte_range: nullopt
 *  return:     ["abc..", "def..", "ghi..", jkl..", ""]
 *
 *  byte_range: [0, 2)
 *  return:     ["abc.."]
 *
 *  byte_range: [2, 9)
 *  return:     ["def..", "ghi.."]
 *
 *  byte_range: [11, 2)
 *  return:     []
 *
 *  byte_range: [13, 7)
 *  return:     ["jkl..", ""]
 * @endcode
 *
 * @param source The source string
 * @param delimiter UTF-8 encoded string for which to find offsets in the source
 * @param options the parsing options to use (including byte range)
 * @param mr Memory resource to use for the device memory allocation
 * @return The strings found by splitting the source by the delimiter within the relevant byte
 * range.
 */
std::unique_ptr<cudf::column> multibyte_split(
  data_chunk_source const& source,
  std::string const& delimiter,
  parse_options options               = {},
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

std::unique_ptr<cudf::column> multibyte_split(
  data_chunk_source const& source,
  std::string const& delimiter,
  std::optional<byte_range_info> byte_range,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

std::unique_ptr<cudf::column> multibyte_split(data_chunk_source const& source,
                                              std::string const& delimiter,
                                              rmm::mr::device_memory_resource* mr);

}  // namespace text
}  // namespace io
}  // namespace cudf
