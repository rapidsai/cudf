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

/**
 * @file reader_impl.hpp
 * @brief cuDF-IO CSV reader class implementation header
 */

#pragma once

#include "csv.h"
#include "csv_gpu.h"

#include <io/utilities/column_buffer.hpp>
#include <io/utilities/datasource.hpp>
#include <io/utilities/hostdevice_vector.hpp>
#include <cudf/detail/utilities/trie.cuh>

#include <cudf/io/readers.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace cudf {
namespace experimental {
namespace io {
namespace detail {
namespace csv {

using namespace cudf::io::csv;
using namespace cudf::io;

/**
 * @brief Implementation for CSV reader
 */
class reader::impl {
 public:
  /**
   * @brief Constructor from a dataset source with reader options.
   *
   * @param source Dataset source
   * @param filepath Filepath if reading dataset from a file
   * @param options Settings for controlling reading behavior
   * @param mr Resource to use for device memory allocation
   */
  explicit impl(std::unique_ptr<datasource> source, std::string filepath,
                reader_options const &options,
                rmm::mr::device_memory_resource *mr);

  /**
   * @brief Read an entire set or a subset of data and returns a set of columns.
   *
   * @param range_offset Number of bytes offset from the start
   * @param range_size Bytes to read; use `0` for all remaining data
   * @param skip_rows Number of rows to skip from the start
   * @param skip_rows_end Number of rows to skip from the end
   * @param num_rows Number of rows to read
   * @param metadata Optional location to return table metadata
   * @param stream Stream to use for memory allocation and kernels
   *
   * @return The set of columns along with metadata
   */
  table_with_metadata read(size_t range_offset, size_t range_size,
                           int skip_rows, int skip_end_rows, int num_rows,
                           cudaStream_t stream);

 private:
  /**
   * @brief Finds row positions within the specified input data.
   *
   * This function scans the input data to record the row offsets (relative to
   * the start of the input data) and the symbol or character that begins that
   * row. A row is actually the data/offset between two termination symbols.
   *
   * @param h_data Uncompressed input data in host memory
   * @param h_size Number of bytes of uncompressed input data
   * @param range_offset Number of bytes offset from the start
   * @param stream Stream to use for memory allocation and kernels
   */
  void gather_row_offsets(const char *h_data, size_t h_size,
                          size_t range_offset, cudaStream_t stream);

  /**
   * @brief Filters and discards row positions that are not used.
   *
   * @param h_data Uncompressed input data in host memory
   * @param h_size Number of bytes of uncompressed input data
   * @param range_size Bytes to read; use `0` for all remaining data
   * @param skip_rows Number of rows to skip from the start
   * @param skip_end_rows Number of rows to skip from the end
   * @param num_rows Number of rows to read; use -1 for all remaining data
   * @param stream Stream to use for memory allocation and kernels
   *
   * @return `std::pair<uint64_t, uint64_t>` First and last row positions
   */
  std::pair<uint64_t, uint64_t> select_rows(const char *h_data, size_t h_size,
                                            size_t range_size,
                                            cudf::size_type skip_rows,
                                            cudf::size_type skip_end_rows,
                                            cudf::size_type num_rows,
                                            cudaStream_t stream);

  /**
   * @brief Returns a detected or parsed list of column dtypes.
   *
   * @param stream Stream to use for memory allocation and kernels
   *
   * @return `std::vector<data_type>` List of column types
   */
  std::vector<data_type> gather_column_types(cudaStream_t stream);

  /**
   * @brief Converts the row-column data and outputs to columns.
   *
   * @param column_types Column types
   * @param out_buffers Output columns' device buffers
   * @param stream Stream to use for memory allocation and kernels
   */
  void decode_data(std::vector<data_type> const &column_types,
                   std::vector<column_buffer> &out_buffers,
                   cudaStream_t stream);

 private:
  rmm::mr::device_memory_resource *mr_ = nullptr;
  std::unique_ptr<datasource> source_;
  std::string filepath_;
  std::string compression_type_;
  const reader_options args_;

  rmm::device_buffer data_;
  rmm::device_vector<uint64_t> row_offsets;
  size_t num_records = 0;   // Number of rows with actual data
  long num_bits = 0;        // Numer of 64-bit bitmaps (different than valid)
  int num_active_cols = 0;  // Number of columns to read
  int num_actual_cols = 0;  // Number of columns in the dataset

  // Parsing options
  ParseOptions opts{};
  thrust::host_vector<column_parse::flags> h_column_flags;
  rmm::device_vector<column_parse::flags> d_column_flags;
  rmm::device_vector<SerialTrieNode> d_trueTrie;
  rmm::device_vector<SerialTrieNode> d_falseTrie;
  rmm::device_vector<SerialTrieNode> d_naTrie;

  // Intermediate data
  std::vector<std::string> col_names;
  std::vector<char> header;
};

}  // namespace csv
}  // namespace detail
}  // namespace io
}  // namespace experimental
}  // namespace cudf
