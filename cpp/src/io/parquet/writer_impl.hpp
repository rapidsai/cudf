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
 * @file writer_impl.hpp
 * @brief cuDF-IO Parquet writer class implementation header
 */

#pragma once

#include "parquet.h"
#include "parquet_gpu.h"

#include <io/utilities/hostdevice_vector.hpp>

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/writers.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>

#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace cudf {
namespace experimental {
namespace io {
namespace detail {
namespace parquet {

// Forward internal classes
class parquet_column_view;

using namespace cudf::io::parquet;
using namespace cudf::io;

/**
 * @brief Implementation for parquet writer
 **/
class writer::impl {
  // Parquet datasets are divided into fixed-size, independent rowgroups
  static constexpr uint32_t DEFAULT_ROWGROUP_MAXSIZE = 128 * 1024 * 1024; // 128MB
  static constexpr uint32_t DEFAULT_ROWGROUP_MAXROWS = 1000000; // Or at most 1M rows

  // rowgroups are divided into pages
  static constexpr uint32_t DEFAULT_TARGET_PAGE_SIZE = 512 * 1024;

 public:
  /**
   * @brief Constructor with writer options.
   *
   * @param filepath Filepath if storing dataset to a file
   * @param options Settings for controlling behavior
   * @param mr Resource to use for device memory allocation
   **/
  explicit impl(std::string filepath, writer_options const& options,
                rmm::mr::device_memory_resource* mr);

  /**
   * @brief Write an entire dataset to parquet format.
   *
   * @param table The set of columns
   * @param stream Stream to use for memory allocation and kernels
   **/
  void write(table_view const& table, cudaStream_t stream);

 private:
  /**
   * @brief Gather page fragments
   *
   * @param frag Destination page fragments
   * @param col_desc column description array
   * @param num_columns Total number of columns
   * @param num_fragments Total number of fragments per column
   * @param num_rows Total number of rows
   * @param fragment_size Number of rows per fragment
   * @param stream Stream to use for memory allocation and kernels
   **/
  void init_page_fragments(hostdevice_vector<gpu::PageFragment>& frag,
                           hostdevice_vector<gpu::EncColumnDesc>& col_desc,
                           uint32_t num_columns, uint32_t num_fragments,
                           uint32_t num_rows, uint32_t fragment_size,
                           cudaStream_t stream);

 private:
  rmm::mr::device_memory_resource* _mr = nullptr;

  size_t max_rowgroup_size_ = DEFAULT_ROWGROUP_MAXSIZE;
  size_t max_rowgroup_rows_ = DEFAULT_ROWGROUP_MAXROWS;
  size_t target_page_size_ = DEFAULT_TARGET_PAGE_SIZE;
  Compression compression_kind_ = Compression::UNCOMPRESSED;

  std::vector<uint8_t> buffer_;
  std::ofstream outfile_;
};

}  // namespace parquet
}  // namespace detail
}  // namespace io
}  // namespace experimental
}  // namespace cudf
