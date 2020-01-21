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

#include <cudf/cudf.h>

#include <memory>
#include <string>
#include <vector>

#include "../../csv/legacy/type_conversion.cuh"

#include <cudf/legacy/table.hpp>
#include <io/utilities/datasource.hpp>
#include <io/utilities/wrapper_utils.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/thrust_rmm_allocator.h>

namespace cudf {
namespace io {
namespace json {

struct ColumnInfo {
  cudf::size_type float_count;
  cudf::size_type datetime_count;
  cudf::size_type string_count;
  cudf::size_type int_count;
  cudf::size_type bool_count;
  cudf::size_type null_count;
};

/**---------------------------------------------------------------------------*
 * @brief Class used to parse Json input and convert it into gdf columns
 *
 *---------------------------------------------------------------------------**/
class reader::Impl {
public:
private:
  const reader_options args_{};

  std::unique_ptr<datasource> source_;
  std::string filepath_;
  std::shared_ptr<arrow::Buffer> buffer_;

  const char *uncomp_data_ = nullptr;
  size_t uncomp_size_ = 0;

  // Used when the input data is compressed, to ensure the allocated uncompressed data is freed
  std::vector<char> uncomp_data_owner_;
  rmm::device_buffer data_;
  rmm::device_vector<uint64_t> rec_starts_;

  size_t byte_range_offset_ = 0;
  size_t byte_range_size_ = 0;

  std::vector<std::string> column_names_;
  std::vector<gdf_dtype> dtypes_;
  std::vector<gdf_dtype_extra_info> dtypes_extra_info_;
  std::vector<gdf_column_wrapper> columns_;

  // parsing options
  const bool allow_newlines_in_strings_ = false;
  ParseOptions opts_{',', '\n', '\"', '.'};
  rmm::device_vector<SerialTrieNode> d_true_trie_;
  rmm::device_vector<SerialTrieNode> d_false_trie_;
  rmm::device_vector<SerialTrieNode> d_na_trie_;

  /**---------------------------------------------------------------------------*
   * @brief Ingest input JSON file/buffer, without decompression
   *
   * Sets the source_, byte_range_offset_, and byte_range_size_ data members
   *
   * @param[in] range_offset Number of bytes offset from the start
   * @param[in] range_size Bytes to read; use `0` for all remaining data
   *
   * @return void
   *---------------------------------------------------------------------------**/
  void ingestRawInput(size_t range_offset, size_t range_size);

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
  void convertJsonToColumns(gdf_dtype *const dtypes, void *const *gdf_columns, cudf::valid_type *const *valid_fields,
                            cudf::size_type *num_valid_fields);

 public:
  /**
   * @brief Constructor from a dataset source with reader options.
   **/
  explicit Impl(std::unique_ptr<datasource> source, std::string filepath,
                reader_options const &args);

  /**
   * @brief Read an entire set or a subset of data from the source
   *
   * @param[in] range_offset Number of bytes offset from the start
   * @param[in] range_size Bytes to read; use `0` for all remaining data
   *
   * @return Object that contains the array of gdf_columns
   **/
  table read(size_t range_offset, size_t range_size);
};

}  // namespace json
}  // namespace io
}  // namespace cudf
