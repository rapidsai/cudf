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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cudf/cudf.h>
#include <cudf/legacy/table.hpp>

#include <io/utilities/datasource.hpp>
#include <io/utilities/wrapper_utils.hpp>

namespace cudf {
namespace io {
namespace avro {

// Forward declare Avro metadata parser
class avro_metadata;

/**
 * @brief Implementation for Avro reader
 **/
class reader::Impl {
 public:
  /**
   * @brief Constructor from a dataset source with reader options.
   **/
  explicit Impl(std::unique_ptr<datasource> source,
                reader_options const &options);

  /**
   * @brief Read an entire set or a subset of data from the source and returns
   * an array of gdf_columns.
   *
   * @param[in] skip_rows Number of rows to skip from the start
   * @param[in] num_rows Number of rows to read; use `0` for all remaining data
   *
   * @return cudf::table Object that contains the array of gdf_columns
   **/
  table read(int skip_rows, int num_rows);

 private:
  /**
   * @brief Decompresses the block data.
   *
   * @param[in] comp_block_data Compressed block data
   *
   * @return device_buffer<uint8_t> Device buffer to decompressed block data
   **/
  device_buffer<uint8_t> decompress_data(
      const device_buffer<uint8_t> &comp_block_data);

  /**
   * @brief Convert the avro row-based block data and outputs to gdf_columns
   *
   * @param[in] block_data Uncompressed block data
   * @param[in] dict Dictionary entries
   * @param[in] global_dictionary Dictionary allocation
   * @param[in] total_dictionary_entries Number of dictionary entries
   * @param[in, out] columns List of gdf_columns
   **/
  void decode_data(const device_buffer<uint8_t> &block_data,
                   const std::vector<std::pair<uint32_t, uint32_t>> &dict,
                   const hostdevice_vector<uint8_t> &global_dictionary,
                   size_t total_dictionary_entries,
                   const std::vector<gdf_column_wrapper> &columns);

 private:
  std::unique_ptr<datasource> source_;
  std::unique_ptr<avro_metadata> md_;

  std::vector<std::string> columns_;
  std::vector<std::pair<int, std::string>> selected_cols_;
};

}  // namespace avro
}  // namespace io
}  // namespace cudf
