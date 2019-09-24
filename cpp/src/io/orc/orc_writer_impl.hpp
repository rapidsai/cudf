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

#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "orc.h"
#include "orc_gpu.h"

#include <io/utilities/wrapper_utils.hpp>

#include <cudf/cudf.h>
#include <cudf/legacy/table.hpp>
#include <utilities/error_utils.hpp>
#include <utilities/integer_utils.hpp>

namespace cudf {
namespace io {
namespace orc {

// Forward internal classes
class ProtobufWriter;
class orc_column;

/**
 * @brief Implementation for ORC writer
 **/
class writer::Impl {
  // ORC datasets start with a 3 byte header
  static constexpr const char* MAGIC = "ORC";

  // ORC datasets are divided into fixed size, independent stripes
  static constexpr uint32_t DEFAULT_STRIPE_SIZE = 64 * 1024 * 64;

  // ORC rows are divided into groups and assigned indexes for faster seeking
  static constexpr uint32_t DEFAULT_ROW_INDEX_STRIDE = 10000;

  // ORC compresses streams into independent chunks
  static constexpr uint32_t DEFAULT_COMPRESSION_BLOCKSIZE = 256 * 1024;

 public:
  /**
   * @brief Constructor with writer options.
   **/
  explicit Impl(std::string filepath, writer_options const& options);

  /**
   * @brief Write an entire dataset to ORC format
   *
   * @param[in] table Table of columns
   **/
  void write(const cudf::table& table);

 private:
  /**
   * @brief Builds up column dictionaries indices
   *
   * @param[in] columns List of columns
   * @param[in] num_rows Total number of rows
   * @param[in] str_col_ids List of columns that are strings type
   * @param[in] dict_data Dictionary data memory
   * @param[in] dict_index Dictionary index memory
   * @param[in,out] dict List of dictionary chunks
   **/
  void init_dictionaries(orc_column* orc_columns, size_t num_rows,
                         const std::vector<int>& str_col_ids,
                         const device_buffer<uint32_t>& dict_data,
                         const device_buffer<uint32_t>& dict_index,
                         hostdevice_vector<gpu::DictionaryChunk>& dict);

  /**
   * @brief Builds up per-stripe dictionaries for string columns
   *
   * @param[in] columns List of columns
   * @param[in] num_rows Total number of rows
   * @param[in] str_col_ids List of columns that are strings type
   * @param[in] stripe_list List of stripe boundaries
   * @param[in] dict List of dictionary chunks
   * @param[in] dict_index List of dictionary indices
   * @param[in,out] stripe_dict List of stripe dictionaries
   **/
  void build_dictionaries(
      orc_column* columns, size_t num_rows, const std::vector<int>& str_col_ids,
      const std::vector<uint32_t>& stripe_list,
      const hostdevice_vector<gpu::DictionaryChunk>& dict,
      const device_buffer<uint32_t>& dict_index,
      hostdevice_vector<gpu::StripeDictionary>& stripe_dict);

  /**
   * @brief Returns stream information for each column
   *
   * @param[in] columns List of columns
   * @param[in] num_columns Total number of columns
   * @param[in] num_rows Total number of rows
   * @param[in] stripe_list List of stripe boundaries
   * @param[in,out] strm_ids List of unique stream identifiers
   *
   * @return List of streams
   **/
  std::vector<Stream> gather_streams(orc_column* columns, size_t num_columns,
                                     size_t num_rows,
                                     const std::vector<uint32_t>& stripe_list,
                                     std::vector<int32_t>& strm_ids);

  /**
   * @brief Encodes the streams as a series of column data chunks
   *
   * @param[in] num_columns Total number of columns
   * @param[in] num_rows Total number of rows
   * @param[in] num_rowgroups Total number of row groups
   * @param[in] str_col_ids List of columns that are strings type
   * @param[in] stripe_list List of stripe boundaries
   * @param[in] streams List of columns' index and data streams
   * @param[in] strm_ids List of unique stream identifiers
   * @param[in,out] chunks List of column data chunks
   *
   * @return Device buffer containing encoded RLE and string data
   **/
  device_buffer<uint8_t> encode_columns(
      orc_column* columns, size_t num_columns, size_t num_rows,
      size_t num_rowgroups, const std::vector<int>& str_col_ids,
      const std::vector<uint32_t>& stripe_list,
      const std::vector<Stream>& streams, const std::vector<int32_t>& strm_ids,
      hostdevice_vector<gpu::EncChunk>& chunks);

  /**
   * @brief Returns stripe information after compacting columns' individual data
   * chunks into contiguous data streams
   *
   * @param[in] num_columns Total number of columns
   * @param[in] num_rows Total number of rows
   * @param[in] num_index_streams Total number of index streams
   * @param[in] num_data_streams Total number of data streams
   * @param[in] stripe_list List of stripe boundaries
   * @param[in,out] chunks List of column data chunks
   * @param[in,out] strm_desc List of stream descriptors
   *
   * @return List of stripes
   **/
  std::vector<StripeInformation> gather_stripes(
      size_t num_columns, size_t num_rows, size_t num_index_streams,
      size_t num_data_streams, const std::vector<uint32_t>& stripe_list,
      hostdevice_vector<gpu::EncChunk>& chunks,
      hostdevice_vector<gpu::StripeStream>& strm_desc);

  /**
   * @brief Write the specified column's row index stream
   *
   * @param[in] stripe_id Stripe's identifier
   * @param[in] stream_id Stream's identifier
   * @param[in] columns List of columns
   * @param[in] num_columns Total number of columns
   * @param[in] num_data_streams Total number of data streams
   * @param[in] group Starting row group in the stripe
   * @param[in] groups_in_stripe Number of row groups in the stripe
   * @param[in] chunks List of all column chunks
   * @param[in] strm_desc List of stream descriptors
   * @param[in] comp_out Output status for compressed streams
   * @param[in] streams List of all streams
   * @param[in] pbw_ Protobuf writer
   **/
  void write_index_stream(
      int32_t stripe_id, int32_t stream_id, orc_column* columns,
      size_t num_columns, size_t num_data_streams, size_t group,
      size_t groups_in_stripe, const hostdevice_vector<gpu::EncChunk>& chunks,
      const hostdevice_vector<gpu::StripeStream>& strm_desc,
      const hostdevice_vector<gpu_inflate_status_s>& comp_out,
      StripeInformation& stripe, std::vector<Stream>& streams,
      ProtobufWriter& pbw_);

  /**
   * @brief Write the specified column's data streams
   *
   * @param[in] strm_desc Stream's descriptor
   * @param[in] chunk First column chunk of the stream
   * @param[in] compressed_data Compressed stream data
   * @param[in] stream_out Temporary host output buffer
   * @param[in] stripe Stream's parent stripe
   * @param[in] streams List of all streams
   **/
  void write_data_stream(const gpu::StripeStream& strm_desc,
                         const gpu::EncChunk& chunk,
                         const uint8_t* compressed_data, uint8_t* stream_out,
                         StripeInformation& stripe,
                         std::vector<Stream>& streams);

  /**
   * @brief Returns the number of row groups for holding the specified rows
   *
   * @param[in] num_rows Number of rows
   **/
  template <typename T = size_t>
  constexpr inline auto div_by_rowgroups(T num_rows) const {
    return util::div_rounding_up_unsafe<T, T>(num_rows, row_index_stride_);
  }

  /**
   * @brief Returns the row index stride divided by the specified number
   *
   * @param[in] modulus Number to use for division
   **/
  template <typename T = size_t>
  constexpr inline auto div_rowgroups_by(T modulus) const {
    return util::div_rounding_up_unsafe<T, T>(row_index_stride_, modulus);
  }

 private:
  size_t max_stripe_size_ = DEFAULT_STRIPE_SIZE;
  size_t row_index_stride_ = DEFAULT_ROW_INDEX_STRIDE;
  size_t compression_blocksize_ = DEFAULT_COMPRESSION_BLOCKSIZE;
  CompressionKind compression_kind_ = CompressionKind::NONE;

  bool enable_dictionary_ = true;

  std::vector<uint8_t> buffer_;
  std::ofstream outfile_;
};

}  // namespace orc
}  // namespace io
}  // namespace cudf
