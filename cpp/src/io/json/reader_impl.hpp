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

#include "json.h"
#include "json_gpu.h"

#include <thrust/device_vector.h>
#include <hash/concurrent_unordered_map.cuh>
#include <rmm/device_buffer.hpp>

#include <io/utilities/column_buffer.hpp>

#include <cudf/io/datasource.hpp>
#include <cudf/io/readers.hpp>

namespace cudf {
namespace io {
namespace detail {
namespace json {
using namespace cudf::io::json;
using namespace cudf::io;

using col_map_type     = cudf::io::json::gpu::col_map_type;
using col_map_ptr_type = std::unique_ptr<col_map_type, std::function<void(col_map_type *)>>;

/**
 * @brief Class used to parse Json input and convert it into gdf columns.
 */
class reader::impl {
 public:
 private:
  const reader_options args_{};

  rmm::mr::device_memory_resource *mr_ = nullptr;

  std::unique_ptr<datasource> source_;
  std::string filepath_;
  std::unique_ptr<datasource::buffer> buffer_;

  const char *uncomp_data_ = nullptr;
  size_t uncomp_size_      = 0;

  // Used when the input data is compressed, to ensure the allocated uncompressed data is freed
  std::vector<char> uncomp_data_owner_;
  rmm::device_buffer data_;
  rmm::device_vector<uint64_t> rec_starts_;

  size_t byte_range_offset_ = 0;
  size_t byte_range_size_   = 0;
  bool load_whole_file_     = true;

  table_metadata metadata;
  std::vector<data_type> dtypes_;

  // the map is only used for files with rows in object format; initialize to a dummy value so the
  // map object can be passed to the kernel in any case
  col_map_ptr_type key_to_col_idx_map;
  std::unique_ptr<rmm::device_scalar<col_map_type>> d_key_col_map;

  // parsing options
  const bool allow_newlines_in_strings_ = false;
  ParseOptions opts_{',', '\n', '\"', '.'};
  rmm::device_vector<SerialTrieNode> d_true_trie_;
  rmm::device_vector<SerialTrieNode> d_false_trie_;
  rmm::device_vector<SerialTrieNode> d_na_trie_;

  /**
   * @brief Sets the column map data member and makes a device copy to be used as a kernel
   * parameter.
   */
  void set_column_map(col_map_ptr_type &&map)
  {
    key_to_col_idx_map = std::move(map);
    d_key_col_map      = std::make_unique<rmm::device_scalar<col_map_type>>(*key_to_col_idx_map);
  }
  /**
   * @brief Gets the pointer to the column hash map in the device memory.
   *
   * Returns `nullptr` if the map is not created.
   */
  auto get_column_map_device_ptr() { return key_to_col_idx_map ? d_key_col_map->data() : nullptr; }

  /**
   * @brief Ingest input JSON file/buffer, without decompression
   *
   * Sets the source_, byte_range_offset_, and byte_range_size_ data members
   *
   * @param[in] range_offset Number of bytes offset from the start
   * @param[in] range_size Bytes to read; use `0` for all remaining data
   */
  void ingest_raw_input(size_t range_offset, size_t range_size);

  /**
   * @brief Extract the JSON objects keys from the input file with object rows.
   *
   * @return Array of keys and a map that maps their hash values to column indices
   */
  std::pair<std::vector<std::string>, col_map_ptr_type> get_json_object_keys_hashes(
    cudaStream_t stream);

  /**
   * @brief Decompress the input data, if needed
   *
   * Sets the uncomp_data_ and uncomp_size_ data members
   */
  void decompress_input();

  /**
   * @brief Finds all record starts in the file and stores them in rec_starts_
   *
   * Does not upload the entire file to the GPU
   *
   * @param[in] stream CUDA stream used for device memory operations and kernel launches.
   */
  void set_record_starts(cudaStream_t stream);

  /**
   * @brief Uploads the relevant segment of the input json data onto the GPU.
   *
   * Sets the d_data_ data member.
   * Only rows that need to be parsed are copied, based on the byte range
   * Also updates the array of record starts to match the device data offset.
   */
  void upload_data_to_device();

  /**
   * @brief Parse the first row to set the column name
   *
   * Sets the column_names_ data member
   *
   * @param[in] stream CUDA stream used for device memory operations and kernel launches.
   */
  void set_column_names(cudaStream_t stream);

  /**
   * @brief Set the data type array data member
   *
   * If user does not pass the data types, deduces types from the file content
   *
   * @param[in] stream CUDA stream used for device memory operations and kernel launches.
   */
  void set_data_types(cudaStream_t stream);

  /**
   * @brief Parse the input data and store results a table
   *
   * @param[in] stream CUDA stream used for device memory operations and kernel launches.
   *
   * @return Table and its metadata
   */
  table_with_metadata convert_data_to_table(cudaStream_t stream);

 public:
  /**
   * @brief Constructor from a dataset source with reader options.
   */
  explicit impl(std::unique_ptr<datasource> source,
                std::string filepath,
                reader_options const &args,
                rmm::mr::device_memory_resource *mr);

  /**
   * @brief Read an entire set or a subset of data from the source
   *
   * @param[in] range_offset Number of bytes offset from the start
   * @param[in] range_size Bytes to read; use `0` for all remaining data
   * @param[in] stream CUDA stream used for device memory operations and kernel launches.
   *
   * @return Table and its metadata
   */
  table_with_metadata read(size_t range_offset, size_t range_size, cudaStream_t stream);
};

}  // namespace json
}  // namespace detail
}  // namespace io
}  // namespace cudf
