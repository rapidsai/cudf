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
   * @param metadata The metadata associated with the table
   * @param stream Stream to use for memory allocation and kernels
   **/
  void write(table_view const& table, const table_metadata *metadata, cudaStream_t stream);

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
  /**
   * @brief Gather per-fragment statistics
   *
   * @param dst_stats output statistics
   * @param frag Input page fragments
   * @param col_desc column description array
   * @param num_columns Total number of columns
   * @param num_fragments Total number of fragments per column
   * @param fragment_size Number of rows per fragment
   * @param stream Stream to use for memory allocation and kernels
   **/
  void gather_fragment_statistics(statistics_chunk *dst_stats,
                           hostdevice_vector<gpu::PageFragment>& frag,
                           hostdevice_vector<gpu::EncColumnDesc>& col_desc,
                           uint32_t num_columns, uint32_t num_fragments,
                           uint32_t fragment_size, cudaStream_t stream);
  /**
   * @brief Build per-chunk dictionaries and count data pages
   *
   * @param chunks column chunk array
   * @param col_desc column description array
   * @param num_rowgroups Total number of rowgroups
   * @param num_columns Total number of columns
   * @param num_dictionaries Total number of dictionaries
   * @param stream Stream to use for memory allocation and kernels
   **/
  void build_chunk_dictionaries(hostdevice_vector<gpu::EncColumnChunk>& chunks,
                                hostdevice_vector<gpu::EncColumnDesc>& col_desc,
                                uint32_t num_rowgroups, uint32_t num_columns,
                                uint32_t num_dictionaries, cudaStream_t stream);
  /**
   * @brief Initialize encoder pages
   *
   * @param chunks column chunk array
   * @param col_desc column description array
   * @param pages encoder pages array
   * @param num_rowgroups Total number of rowgroups
   * @param num_columns Total number of columns
   * @param num_pages Total number of pages
   * @param num_stats_bfr Number of statistics buffers
   * @param stream Stream to use for memory allocation and kernels
   **/
  void init_encoder_pages(hostdevice_vector<gpu::EncColumnChunk>& chunks,
                          hostdevice_vector<gpu::EncColumnDesc>& col_desc,
                          gpu::EncPage *pages,
                          statistics_chunk *page_stats,
                          statistics_chunk *frag_stats,
                          uint32_t num_rowgroups, uint32_t num_columns,
                          uint32_t num_pages, uint32_t num_stats_bfr,
                          cudaStream_t stream);
  /**
   * @brief Encode a batch pages
   *
   * @param chunks column chunk array
   * @param pages encoder pages array
   * @param num_columns Total number of columns
   * @param pages_in_batch number of pages in this batch
   * @param first_page_in_batch first page in batch
   * @param rowgroups_in_batch number of rowgroups in this batch
   * @param first_rowgroup first rowgroup in batch
   * @param comp_in compressor input array
   * @param comp_out compressor status array
   * @param page_stats optional page-level statistics (nullptr if none)
   * @param chunk_stats optional chunk-level statistics (nullptr if none)
   * @param stream Stream to use for memory allocation and kernels
   **/
  void encode_pages(hostdevice_vector<gpu::EncColumnChunk>& chunks,
                    gpu::EncPage *pages, uint32_t num_columns,
                    uint32_t pages_in_batch, uint32_t first_page_in_batch,
                    uint32_t rowgroups_in_batch, uint32_t first_rowgroup,
                    gpu_inflate_input_s *comp_in,
                    gpu_inflate_status_s *comp_out,
                    const statistics_chunk *page_stats,
                    const statistics_chunk *chunk_stats,
                    cudaStream_t stream);

 private:
  rmm::mr::device_memory_resource* _mr = nullptr;

  size_t max_rowgroup_size_ = DEFAULT_ROWGROUP_MAXSIZE;
  size_t max_rowgroup_rows_ = DEFAULT_ROWGROUP_MAXROWS;
  size_t target_page_size_ = DEFAULT_TARGET_PAGE_SIZE;
  Compression compression_ = Compression::UNCOMPRESSED;
  statistics_freq stats_granularity_ = statistics_freq::STATISTICS_NONE;

  std::vector<uint8_t> buffer_;
  std::ofstream outfile_;
};

}  // namespace parquet
}  // namespace detail
}  // namespace io
}  // namespace experimental
}  // namespace cudf
