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
 * @brief cuDF-IO ORC writer class implementation header
 */

#pragma once

#include "orc.h"
#include "orc_gpu.h"

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
namespace orc {

// Forward internal classes
class orc_column_view;

using namespace cudf::io::orc;
using namespace cudf::io;

/**
 * @brief Implementation for ORC writer
 **/
class writer::impl {
  // ORC datasets start with a 3 byte header
  static constexpr const char* MAGIC = "ORC";

  // ORC datasets are divided into fixed-size, independent stripes
  static constexpr uint32_t DEFAULT_STRIPE_SIZE = 64 * 1024 * 1024;

  // ORC rows are divided into groups and assigned indexes for faster seeking
  static constexpr uint32_t DEFAULT_ROW_INDEX_STRIDE = 10000;

  // ORC compresses streams into independent chunks
  static constexpr uint32_t DEFAULT_COMPRESSION_BLOCKSIZE = 256 * 1024;

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
   * @brief Write an entire dataset to ORC format.
   *
   * @param table The set of columns
   * @param metadata The metadata associated with the table
   * @param stream Stream to use for memory allocation and kernels
   **/
  void write(table_view const& table, const table_metadata *metadata, cudaStream_t stream);

 private:
  /**
   * @brief Builds up column dictionaries indices
   *
   * @param columns List of columns
   * @param num_rows Total number of rows
   * @param str_col_ids List of columns that are strings type
   * @param dict_data Dictionary data memory
   * @param dict_index Dictionary index memory
   * @param dict List of dictionary chunks
   * @param stream Stream to use for memory allocation and kernels
   **/
  void init_dictionaries(orc_column_view* columns, size_t num_rows,
                         std::vector<int> const& str_col_ids,
                         uint32_t* dict_data, uint32_t* dict_index,
                         hostdevice_vector<gpu::DictionaryChunk>& dict,
                         cudaStream_t stream);

  /**
   * @brief Builds up per-stripe dictionaries for string columns
   *
   * @param columns List of columns
   * @param num_rows Total number of rows
   * @param str_col_ids List of columns that are strings type
   * @param stripe_list List of stripe boundaries
   * @param dict List of dictionary chunks
   * @param dict_index List of dictionary indices
   * @param stripe_dict List of stripe dictionaries
   * @param stream Stream to use for memory allocation and kernels
   **/
  void build_dictionaries(orc_column_view* columns, size_t num_rows,
                          std::vector<int> const& str_col_ids,
                          std::vector<uint32_t> const& stripe_list,
                          hostdevice_vector<gpu::DictionaryChunk> const& dict,
                          uint32_t* dict_index,
                          hostdevice_vector<gpu::StripeDictionary>& stripe_dict,
                          cudaStream_t stream);

  /**
   * @brief Returns stream information for each column
   *
   * @param columns List of columns
   * @param num_columns Total number of columns
   * @param num_rows Total number of rows
   * @param stripe_list List of stripe boundaries
   * @param strm_ids List of unique stream identifiers
   *
   * @return The streams
   **/
  std::vector<Stream> gather_streams(orc_column_view* columns,
                                     size_t num_columns, size_t num_rows,
                                     std::vector<uint32_t> const& stripe_list,
                                     std::vector<int32_t>& strm_ids);

  /**
   * @brief Encodes the streams as a series of column data chunks
   *
   * @param num_columns Total number of columns
   * @param num_rows Total number of rows
   * @param num_rowgroups Total number of row groups
   * @param str_col_ids List of columns that are strings type
   * @param stripe_list List of stripe boundaries
   * @param streams List of columns' index and data streams
   * @param strm_ids List of unique stream identifiers
   * @param chunks List of column data chunks
   * @param stream Stream to use for memory allocation and kernels
   *
   * @return Device buffer containing encoded data
   **/
  rmm::device_buffer encode_columns(
      orc_column_view* columns, size_t num_columns, size_t num_rows,
      size_t num_rowgroups, std::vector<int> const& str_col_ids,
      std::vector<uint32_t> const& stripe_list,
      std::vector<Stream> const& streams, std::vector<int32_t> const& strm_ids,
      hostdevice_vector<gpu::EncChunk>& chunks, cudaStream_t stream);

  /**
   * @brief Returns stripe information after compacting columns' individual data
   * chunks into contiguous data streams
   *
   * @param num_columns Total number of columns
   * @param num_rows Total number of rows
   * @param num_index_streams Total number of index streams
   * @param num_data_streams Total number of data streams
   * @param stripe_list List of stripe boundaries
   * @param chunks List of column data chunks
   * @param strm_desc List of stream descriptors
   * @param stream Stream to use for memory allocation and kernels
   *
   * @return The stripes' information
   **/
  std::vector<StripeInformation> gather_stripes(
      size_t num_columns, size_t num_rows, size_t num_index_streams,
      size_t num_data_streams, std::vector<uint32_t> const& stripe_list,
      hostdevice_vector<gpu::EncChunk>& chunks,
      hostdevice_vector<gpu::StripeStream>& strm_desc, cudaStream_t stream);

  /**
   * @brief Write the specified column's row index stream
   *
   * @param stripe_id Stripe's identifier
   * @param stream_id Stream's identifier
   * @param columns List of columns
   * @param num_columns Total number of columns
   * @param num_data_streams Total number of data streams
   * @param group Starting row group in the stripe
   * @param groups_in_stripe Number of row groups in the stripe
   * @param chunks List of all column chunks
   * @param strm_desc List of stream descriptors
   * @param comp_out Output status for compressed streams
   * @param streams List of all streams
   * @param pbw Protobuf writer
   **/
  void write_index_stream(
      int32_t stripe_id, int32_t stream_id, orc_column_view* columns,
      size_t num_columns, size_t num_data_streams, size_t group,
      size_t groups_in_stripe, hostdevice_vector<gpu::EncChunk> const& chunks,
      hostdevice_vector<gpu::StripeStream> const& strm_desc,
      hostdevice_vector<gpu_inflate_status_s> const& comp_out,
      StripeInformation& stripe, std::vector<Stream>& streams,
      ProtobufWriter* pbw);

  /**
   * @brief Write the specified column's data streams
   *
   * @param strm_desc Stream's descriptor
   * @param chunk First column chunk of the stream
   * @param compressed_data Compressed stream data
   * @param stream_out Temporary host output buffer
   * @param stripe Stream's parent stripe
   * @param streams List of all streams
   * @param stream Stream to use for memory allocation and kernels
   **/
  void write_data_stream(gpu::StripeStream const& strm_desc,
                         gpu::EncChunk const& chunk,
                         uint8_t const* compressed_data, uint8_t* stream_out,
                         StripeInformation& stripe,
                         std::vector<Stream>& streams, cudaStream_t stream);

  /**
   * @brief Returns the number of row groups for holding the specified rows
   *
   * @tparam T Optional type
   * @param num_rows Number of rows
   **/
  template <typename T = size_t>
  constexpr inline auto div_by_rowgroups(T num_rows) const {
    return cudf::util::div_rounding_up_unsafe<T, T>(num_rows,
                                                    row_index_stride_);
  }

  /**
   * @brief Returns the row index stride divided by the specified number
   *
   * @tparam T Optional type
   * @param modulus Number to use for division
   **/
  template <typename T = size_t>
  constexpr inline auto div_rowgroups_by(T modulus) const {
    return cudf::util::div_rounding_up_unsafe<T, T>(row_index_stride_, modulus);
  }

 private:
  rmm::mr::device_memory_resource* _mr = nullptr;

  size_t max_stripe_size_ = DEFAULT_STRIPE_SIZE;
  size_t row_index_stride_ = DEFAULT_ROW_INDEX_STRIDE;
  size_t compression_blocksize_ = DEFAULT_COMPRESSION_BLOCKSIZE;
  CompressionKind compression_kind_ = CompressionKind::NONE;

  bool enable_dictionary_ = true;

  std::vector<uint8_t> buffer_;
  std::ofstream outfile_;
};

}  // namespace orc
}  // namespace detail
}  // namespace io
}  // namespace experimental
}  // namespace cudf
