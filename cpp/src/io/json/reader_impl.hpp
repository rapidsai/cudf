/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

/**
 * @file reader_impl.hpp
 * @brief cuDF-IO JSON reader class implementation header
 */

#pragma once

#include "json_common.h"
#include "json_gpu.h"

#include <io/utilities/column_buffer.hpp>

#include <hash/concurrent_unordered_map.cuh>

#include <cudf/io/datasource.hpp>
#include <cudf/io/detail/json.hpp>
#include <cudf/io/json.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

namespace cudf {
namespace io {
namespace detail {
namespace json {
using namespace cudf::io::json;
using namespace cudf::io;

using col_map_type     = cudf::io::json::gpu::col_map_type;
using col_map_ptr_type = std::unique_ptr<col_map_type, std::function<void(col_map_type*)>>;

/**
 * @brief Class used to parse Json input and convert it into gdf columns.
 */
class reader_impl {
 private:
  /**
   * @brief Ingest input JSON file/buffer, without decompression
   *
   * Sets the source_, byte_range_offset_, and byte_range_size_ data members
   *
   * @param[in] buffer Buffer to read the bytes in to
   * @param[in] range_offset Number of bytes offset from the start
   * @param[in] range_size Bytes to read; use `0` for all remaining data
   * @param[in] range_size_padded Bytes to read with padding; use `0` for all remaining data
   */
  std::vector<char> ingest_raw_input(std::vector<std::unique_ptr<datasource>> const& sources,
                                     compression_type compression,
                                     size_t range_offset,
                                     size_t range_size,
                                     size_t range_size_padded);

  /**
   * @brief Extract the JSON objects keys from the input file with object rows.
   *
   * @return Array of keys and a map that maps their hash values to column indices
   */
  std::pair<std::vector<std::string>, col_map_ptr_type> get_json_object_keys_hashes(
    parse_options_view const& parse_opts,
    host_span<char const> h_data,
    device_span<uint64_t const> rec_starts,
    device_span<char const> d_data,
    rmm::cuda_stream_view stream);

  /**
   * @brief Finds all record starts in the file.
   *
   * Does not upload the entire file to the GPU
   *
   * @param[in] stream CUDA stream used for device memory operations and kernel launches.
   * @return Record starts in the device memory
   */
  rmm::device_uvector<uint64_t> find_record_starts(json_reader_options const& reader_opts,
                                                   host_span<char const> h_data,
                                                   device_span<char const> d_data,
                                                   rmm::cuda_stream_view stream);

  /**
   * @brief Uploads the relevant segment of the input json data onto the GPU.
   *
   * Sets the d_data_ data member.
   * Only rows that need to be parsed are copied, based on the byte range
   * Also updates the array of record starts to match the device data offset.
   */
  rmm::device_uvector<char> upload_data_to_device(json_reader_options const& reader_opts,
                                                  host_span<char const> h_data,
                                                  rmm::device_uvector<uint64_t>& rec_starts,
                                                  rmm::cuda_stream_view stream);

  /**
   * @brief Parse the first row to set the column name
   *
   * Sets the column_names_ data member
   *
   * @param[in] rec_starts Record starts in device memory
   * @param[in] stream CUDA stream used for device memory operations and kernel launches.
   */
  std::pair<std::vector<std::string>, col_map_ptr_type> get_column_names_and_map(
    parse_options_view const& parse_opts,
    host_span<char const> h_data,
    device_span<uint64_t const> rec_starts,
    device_span<char const> d_data,
    rmm::cuda_stream_view stream);

  std::vector<data_type> parse_data_types(std::vector<std::string> const& column_names,
                                          std::vector<std::string> const& types_as_strings);

  /**
   * @brief Set the data type array data member
   *
   * If user does not pass the data types, deduces types from the file content
   *
   * @param[in] reader_opts Settings for controlling reading behavior
   * @param[in] rec_starts Record starts in device memory
   * @param[in] stream CUDA stream used for device memory operations and kernel launches.
   */
  std::vector<data_type> get_data_types(json_reader_options const& reader_opts,
                                        parse_options_view const& parse_opts,
                                        std::vector<std::string> const& column_names,
                                        col_map_type* column_map,
                                        device_span<uint64_t const> rec_starts,
                                        device_span<char const> data,
                                        rmm::cuda_stream_view stream);

  /**
   * @brief Parse the input data and store results a table
   *
   * @param[in] rec_starts Record starts in device memory
   * @param[in] stream CUDA stream used for device memory operations and kernel launches.
   * @param[in] mr Device memory resource to use for device memory allocation
   *
   * @return Table and its metadata
   */
  table_with_metadata convert_data_to_table(parse_options_view const& parse_opts,
                                            std::vector<data_type> const& dtypes,
                                            std::vector<std::string> const& column_names,
                                            col_map_type* column_map,
                                            device_span<uint64_t const> rec_starts,
                                            device_span<char const> data,
                                            rmm::cuda_stream_view stream,
                                            rmm::mr::device_memory_resource* mr);

 public:
  /**
   * @brief Read an entire set or a subset of data from the source
   *
   * @param[in] sources Input `datasource` objects to read the dataset from
   * @param[in] options Settings for controlling reading behavior
   * @param[in] stream CUDA stream used for device memory operations and kernel launches.
   * @param[in] mr Device memory resource to use for device memory allocation
   *
   * @return Table and its metadata
   */
  table_with_metadata read(std::vector<std::unique_ptr<datasource>>& sources,
                           json_reader_options const& options,
                           rmm::cuda_stream_view stream,
                           rmm::mr::device_memory_resource* mr);
};

}  // namespace json
}  // namespace detail
}  // namespace io
}  // namespace cudf
