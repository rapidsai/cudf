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

#include <memory>
#include <vector>

#include <cudf.h>

#include "../csv/type_conversion.cuh"
#include "io/utilities/file_utils.hpp"
#include "io/utilities/wrapper_utils.hpp"

/**---------------------------------------------------------------------------*
 * @brief Class used to parse Json input and convert it into gdf columns
 *
 *---------------------------------------------------------------------------**/
class JsonReader {
public:
  struct ColumnInfo {
    gdf_size_type float_count;
    gdf_size_type datetime_count;
    gdf_size_type string_count;
    gdf_size_type int_count;
    gdf_size_type null_count;
  };

private:
  const json_read_arg *args_ = nullptr;

  std::unique_ptr<MappedFile> map_file_;
  const char *input_data_ = nullptr;
  size_t input_size_ = 0;
  const char *uncomp_data_ = nullptr;
  size_t uncomp_size_ = 0;
  // Used when the input data is compressed, to ensure the allocated uncompressed data is freed
  std::vector<char> uncomp_data_owner_;
  device_buffer<char> d_data_;

  std::vector<std::string> column_names_;
  std::vector<gdf_dtype> dtypes_;
  std::vector<gdf_column_wrapper> columns_;

  device_buffer<uint64_t> rec_starts_;

  // parsing options
  const bool allow_newlines_in_strings_ = false;
  const ParseOptions opts_{',', '\n', '\"', '.'};

  /**---------------------------------------------------------------------------*
   * @brief Ingest input JSON file/buffer, without decompression
   *
   * Sets the input_data_ and input_size_ data members
   *
   * @return void
   *---------------------------------------------------------------------------**/
  void ingestRawInput();

  /**---------------------------------------------------------------------------*
   * @brief Decompress the input data, if needed
   *
   * Sets the uncomp_data_ and uncomp_size_ data members
   *
   * @return void
   *---------------------------------------------------------------------------**/
  void decompressInput();

  /**---------------------------------------------------------------------------*
   * @brief Finds all record starts in the file and stores them in rec_starts_
   *
   * Does not upload the entire file to the GPU
   *
   * @return void
   *---------------------------------------------------------------------------**/
  void setRecordStarts();

  /**---------------------------------------------------------------------------*
   * @brief Uploads the relevant segment of the input json data onto the GPU.
   *
   * Sets the d_data_ data member.
   * Only rows that need to be parsed are copied, based on the byte range
   * Also updates the array of record starts to match the device data offset.
   *
   * @return void
   *---------------------------------------------------------------------------**/
  void uploadDataToDevice();

  /**---------------------------------------------------------------------------*
   * @brief Parse the first row to set the column name
   *
   * Sets the column_names_ data member
   *
   * @return void
   *---------------------------------------------------------------------------**/
  void setColumnNames();

  /**---------------------------------------------------------------------------*
   * @brief Set the data type array data member
   *
   * If user does not pass the data types, deduces types from the file content
   *
   * @return void
   *---------------------------------------------------------------------------**/
  void setDataTypes();

  /**---------------------------------------------------------------------------*
   * @brief Set up and launches JSON data type detect CUDA kernel.
   *
   * @param[out] column_infos The count for each column data type
   *
   * @return void
   *---------------------------------------------------------------------------**/
  void detectDataTypes(ColumnInfo *column_infos);

  /**---------------------------------------------------------------------------*
   * @brief Parse the input data and store results in dsf columns
   *
   * Allocates columns in the column_names_ data member and populates them with
   * parsed data
   *
   * @return void
   *---------------------------------------------------------------------------**/
  void convertDataToColumns();

  /**---------------------------------------------------------------------------*
   * @brief Helper function to setup and launch JSON parsing CUDA kernel.
   *
   * @param[in] dtypes The data type of each column
   * @param[out] gdf_columns The output column data
   * @param[out] valid_fields The bitmaps indicating which column fields are valid
   * @param[out] num_valid_fields The numbers of valid fields in columns
   *
   * @return void
   *---------------------------------------------------------------------------**/
  void convertJsonToColumns(gdf_dtype *const dtypes, void *const *gdf_columns, gdf_valid_type *const *valid_fields,
                            gdf_size_type *num_valid_fields);

public:
  /**---------------------------------------------------------------------------*
   * @brief JsonReader constructor; throws if the arguments are not supported
   *---------------------------------------------------------------------------**/
  explicit JsonReader(json_read_arg *args);

  /**---------------------------------------------------------------------------*
   * @brief Parse the input JSON file as specified with the args_ data member
   *
   * Stores the parsed gdf columns in an internal data member
   *
   * @return void
   *---------------------------------------------------------------------------**/
  void parse();

  /**---------------------------------------------------------------------------*
   * @brief Sets the output paramenters of the read_json call
   *
   * Transfers the ownership from private data members without copying
   * the gdf column data
   *
   * @param[out] out_args Pointer to the output structure to be populated
   *
   * @return void
   *---------------------------------------------------------------------------**/
  void setOutputArguments(json_read_arg *out_args);
};
