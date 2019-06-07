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

#include <cudf.h>
#include <table.hpp>

#include <string>
#include <vector>

#include "io/utilities/wrapper_utils.hpp"
#include "type_conversion.cuh"

namespace cudf {

struct column_data_t {
  gdf_size_type countFloat;
  gdf_size_type countDateAndTime;
  gdf_size_type countString;
  gdf_size_type countBool;
  gdf_size_type countInt8;
  gdf_size_type countInt16;
  gdf_size_type countInt32;
  gdf_size_type countInt64;
  gdf_size_type countNULL;
};

/**---------------------------------------------------------------------------*
 * @brief Class used to parse Json input and convert it into gdf columns
 *
 *---------------------------------------------------------------------------**/
class CsvReader::Impl {
private:
  const csv_reader_args args_;
  device_buffer<char> data;         // on-device: the raw unprocessed CSV data - loaded as a large char * array
  device_buffer<uint64_t> recStart; // on-device: Starting position of the records.

  ParseOptions opts; // options to control parsing behavior

  long num_bytes;            // host: the number of bytes in the data
  long num_bits;             // host: the number of 64-bit bitmaps (different than valid)
  gdf_size_type num_records; // host: number of records loaded into device memory, and then number of records to read
  int num_active_cols;       // host: number of columns that will be return to user.
  int num_actual_cols;       // host: number of columns in the file --- based on the number of columns in header
  std::vector<gdf_dtype> dtypes;      // host: array of dtypes (since gdf_columns are not created until end)
  std::vector<std::string> col_names; // host: array of column names

  thrust::host_vector<bool> h_parseCol; // host   : array of booleans stating if column should be parsed in reading
                                        // process: parseCol[x]=false means that the column x needs to be filtered out.
  rmm::device_vector<bool> d_parseCol;  // device : array of booleans stating if column should be parsed in reading
                                        // process: parseCol[x]=false means that the column x needs to be filtered out.

  long byte_range_offset; // offset into the data to start parsing
  long byte_range_size;   // length of the data of interest to parse

  gdf_size_type header_row; ///< host: Row index of the header
  gdf_size_type nrows;      ///< host: Number of rows to read. -1 for all rows
  gdf_size_type skiprows;   ///< host: Number of rows to skip from the start
  gdf_size_type skipfooter; ///< host: Number of rows to skip from the end
  std::vector<char> header; ///< host: Header row data, for parsing column names
  std::string prefix;       ///< host: Prepended to column ID if there is no header or input column names

  rmm::device_vector<SerialTrieNode> d_trueTrie;  // device: serialized trie of values to recognize as true
  rmm::device_vector<SerialTrieNode> d_falseTrie; // device: serialized trie of values to recognize as false
  rmm::device_vector<SerialTrieNode> d_naTrie;    // device: serialized trie of NA values

  void setColumnNamesFromCsv();
  void countRecordsAndQuotes(const char *h_data, size_t h_size);
  void setRecordStarts(const char *h_data, size_t h_size);
  void uploadDataToDevice(const char *h_uncomp_data, size_t h_uncomp_size);

  void launch_dataConvertColumns(void **d_gdf, gdf_valid_type **valid, gdf_dtype *d_dtypes, gdf_size_type *num_valid);
  void launch_dataTypeDetection(column_data_t *d_columnData);

public:
  /**---------------------------------------------------------------------------*
   * @brief CsvReader::Impl constructor; throws if arguments are not supported
   *---------------------------------------------------------------------------**/
  explicit Impl(csv_reader_args const &args);

  /**---------------------------------------------------------------------------*
   * @brief Parse the input CSV file as specified with the args_ data member
   *
   * @return cudf::table object that contains the array of gdf_columns
   *---------------------------------------------------------------------------**/
  table read();

  auto getArgs() const { return args_; }
};

} // namespace cudf
