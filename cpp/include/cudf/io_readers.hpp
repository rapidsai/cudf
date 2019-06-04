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
#include <vector>
#include <memory>

#include <cudf.h>
#include <table.hpp>

namespace cudf {
 
 /**---------------------------------------------------------------------------*
 * @brief Arguments to the read_json interface.
 *---------------------------------------------------------------------------**/
struct json_reader_args{
  gdf_input_type  source_type = HOST_BUFFER;      ///< Type of the data source.
  std::string     source;                         ///< If source_type is FILE_PATH, contains the filepath. If source_type is HOST_BUFFER, contains the input JSON data.

  std::vector<std::string>  dtype;                ///< Ordered list of data types; pass an empty vector to use data type deduction.
  std::string               compression = "infer";///< Compression type ("none", "infer", "gzip", "zip"); default is "infer".
  bool                      lines = false;        ///< Read the file as a json object per line; default is false.

  json_reader_args() = default;
 
  json_reader_args(json_reader_args const &) = default;

  /**---------------------------------------------------------------------------*
   * @brief json_reader_args constructor that sets the source data members.
   * 
   * @param[in] src_type Enum describing the type of the data source.
   * @param[in] src If src_type is FILE_PATH, contains the filepath.
   * If source_type is HOST_BUFFER, contains the input JSON data.
   *---------------------------------------------------------------------------**/
  json_reader_args(gdf_input_type src_type, std::string const &src) : source_type(src_type), source(src) {}
};

/**---------------------------------------------------------------------------*
 * @brief Class used to parse Json input and convert it into gdf columns.
 *
 *---------------------------------------------------------------------------**/
class JsonReader {
private:
  class Impl;
  std::unique_ptr<Impl> impl_;

public:
  /**---------------------------------------------------------------------------*
   * @brief JsonReader defult constructor;
   * 
   * Calling read_XYZ() on a default-constructed object yields an empty table.
   *---------------------------------------------------------------------------**/
  JsonReader() noexcept;

  JsonReader(JsonReader const &rhs);
  JsonReader& operator=(JsonReader const &rhs);

  JsonReader(JsonReader &&rhs);
  JsonReader& operator=(JsonReader &&rhs);

  /**---------------------------------------------------------------------------*
   * @brief JsonReader constructor; throws if the arguments are not supported.
   *---------------------------------------------------------------------------**/
  explicit JsonReader(json_reader_args const &args);

  /**---------------------------------------------------------------------------*
   * @brief Parse the input JSON file as specified with the json_reader_args
   * constuctor parameter.
   *
   * @return cudf::table object that contains the array of gdf_columns.
   *---------------------------------------------------------------------------**/
  table read();

  /**---------------------------------------------------------------------------*
   * @brief Parse the input JSON file as specified with the args_ data member.
   *
   * Stores the parsed gdf columns in an internal data member.
   * @param[in] offset ///< Offset of the byte range to read.
   * @param[in] size   ///< Size of the byte range to read. If set to zero,
   * all data after byte_range_offset is read.
   *
   * @return cudf::table object that contains the array of gdf_columns.
   *---------------------------------------------------------------------------**/
  table read_byte_range(size_t offset, size_t size);

  ~JsonReader();
};

} // namespace cudf
