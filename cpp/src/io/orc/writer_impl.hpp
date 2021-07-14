/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include "orc.h"
#include "orc_gpu.h"

#include <io/utilities/hostdevice_vector.hpp>

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/detail/orc.hpp>
#include <cudf/io/orc.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/error.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <memory>
#include <string>
#include <vector>

namespace cudf {
namespace io {
namespace detail {
namespace orc {
// Forward internal classes
class orc_column_view;

using namespace cudf::io::orc;
using namespace cudf::io;
using cudf::detail::device_2dspan;
using cudf::detail::host_2dspan;
using cudf::detail::hostdevice_2dvector;

/**
 * Non-owning view of a cuDF table that includes ORC-related information.
 *
 * Columns hierarchy is flattened and stored in pre-order.
 */
struct orc_table_view {
  std::vector<orc_column_view> columns;
  rmm::device_uvector<orc_column_device_view> d_columns;
  std::vector<uint32_t> string_column_indices;
  rmm::device_uvector<uint32_t> d_string_column_indices;

  auto num_columns() const noexcept { return columns.size(); }
  size_type num_rows() const noexcept;
  auto num_string_columns() const noexcept { return string_column_indices.size(); }

  auto& column(uint32_t idx) { return columns.at(idx); }
  auto const& column(uint32_t idx) const { return columns.at(idx); }

  auto& string_column(uint32_t idx) { return columns.at(string_column_indices.at(idx)); }
  auto const& string_column(uint32_t idx) const
  {
    return columns.at(string_column_indices.at(idx));
  }
};

/**
 * @brief Indices of rowgroups contained in a stripe.
 *
 * Provides a container-like interface to iterate over rowgroup indices.
 */
struct stripe_rowgroups {
  uint32_t id;     // stripe id
  uint32_t first;  // first rowgroup in the stripe
  uint32_t size;   // number of rowgroups in the stripe
  stripe_rowgroups(uint32_t id, uint32_t first, uint32_t size) : id{id}, first{first}, size{size} {}
  auto cbegin() const { return thrust::make_counting_iterator(first); }
  auto cend() const { return thrust::make_counting_iterator(first + size); }
};

/**
 * @brief Holds the sizes of encoded elements of decimal columns.
 */
struct encoder_decimal_info {
  std::map<uint32_t, rmm::device_uvector<uint32_t>>
    elem_sizes;                                        ///< Column index -> per-element size map
  std::map<uint32_t, std::vector<uint32_t>> rg_sizes;  ///< Column index -> per-rowgroup size map
};

/**
 * @brief List of per-column ORC streams.
 *
 * Provides interface to calculate their offsets.
 */
class orc_streams {
 public:
  orc_streams(std::vector<Stream> streams, std::vector<int32_t> ids)
    : streams{std::move(streams)}, ids{std::move(ids)}
  {
  }
  Stream const& operator[](int idx) const { return streams[idx]; }
  Stream& operator[](int idx) { return streams[idx]; }
  auto id(int idx) const { return ids[idx]; }
  auto& id(int idx) { return ids[idx]; }
  auto size() const { return streams.size(); }

  /**
   * @brief List of ORC stream offsets and their total size.
   */
  struct orc_stream_offsets {
    std::vector<size_t> offsets;
    size_t non_rle_data_size = 0;
    size_t rle_data_size     = 0;
    auto data_size() const { return non_rle_data_size + rle_data_size; }
  };
  orc_stream_offsets compute_offsets(host_span<orc_column_view const> columns,
                                     size_t num_rowgroups) const;

  operator std::vector<Stream> const &() const { return streams; }

 private:
  std::vector<Stream> streams;
  std::vector<int32_t> ids;
};
/**
 * @brief Description of how the ORC file is segmented into stripes and rowgroups.
 */
struct file_segmentation {
  hostdevice_2dvector<rowgroup_rows> rowgroups;
  std::vector<stripe_rowgroups> stripes;

  auto num_rowgroups() const noexcept { return rowgroups.size().first; }
  auto num_stripes() const noexcept { return stripes.size(); }
};

/**
 * @brief ORC per-chunk streams of encoded data.
 */
struct encoded_data {
  rmm::device_uvector<uint8_t> data;                        // Owning array of the encoded data
  hostdevice_2dvector<gpu::encoder_chunk_streams> streams;  // streams of encoded data, per chunk
};

/**
 * @brief Dictionary data for string columns and their device views, per column.
 */
struct string_dictionaries {
  std::vector<rmm::device_uvector<uint32_t>> data;
  std::vector<rmm::device_uvector<uint32_t>> index;
  rmm::device_uvector<device_span<uint32_t>> d_data_view;
  rmm::device_uvector<device_span<uint32_t>> d_index_view;
  // Dictionaries are currently disabled for columns with a rowgroup larger than 2^15
  thrust::host_vector<bool> dictionary_enabled;
};

/**
 * @brief Implementation for ORC writer
 */
class writer::impl {
  // ORC datasets start with a 3 byte header
  static constexpr const char* MAGIC = "ORC";

  // ORC datasets are divided into fixed-size, independent stripes
  static constexpr uint32_t DEFAULT_STRIPE_SIZE = 64 * 1024 * 1024;

  // ORC compresses streams into independent chunks
  static constexpr uint32_t DEFAULT_COMPRESSION_BLOCKSIZE = 256 * 1024;

 public:
  /**
   * @brief Constructor with writer options.
   *
   * @param sink Output sink
   * @param options Settings for controlling behavior
   * @param mode Option to write at once or in chunks
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit impl(std::unique_ptr<data_sink> sink,
                orc_writer_options const& options,
                SingleWriteMode mode,
                rmm::cuda_stream_view stream,
                rmm::mr::device_memory_resource* mr);

  /**
   * @brief Constructor with chunked writer options.
   *
   * @param sink Output sink
   * @param options Settings for controlling behavior
   * @param mode Option to write at once or in chunks
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @param mr Device memory resource to use for device memory allocation
   */
  explicit impl(std::unique_ptr<data_sink> sink,
                chunked_orc_writer_options const& options,
                SingleWriteMode mode,
                rmm::cuda_stream_view stream,
                rmm::mr::device_memory_resource* mr);

  /**
   * @brief Destructor to complete any incomplete write and release resources.
   */
  ~impl();

  /**
   * @brief Begins the chunked/streamed write process.
   */
  void init_state();

  /**
   * @brief Writes a single subtable as part of a larger ORC file/table write.
   *
   * @param[in] table The table information to be written
   */
  void write(table_view const& table);

  /**
   * @brief Finishes the chunked/streamed write process.
   */
  void close();

 private:
  /**
   * @brief Builds up per-stripe dictionaries for string columns.
   *
   * @param orc_table Non-owning view of a cuDF table w/ ORC-related info
   * @param stripe_bounds List of stripe boundaries
   * @param dict List of dictionary chunks
   * @param dict_index List of dictionary indices
   * @param dictionary_enabled Whether dictionary encoding is enabled for a given column
   * @param stripe_dict List of stripe dictionaries
   */
  void build_dictionaries(orc_table_view& orc_table,
                          host_span<stripe_rowgroups const> stripe_bounds,
                          hostdevice_vector<gpu::DictionaryChunk> const& dict,
                          host_span<rmm::device_uvector<uint32_t>> dict_index,
                          host_span<bool const> dictionary_enabled,
                          hostdevice_vector<gpu::StripeDictionary>& stripe_dict);

  /**
   * @brief Builds up per-column streams.
   *
   * @param[in,out] columns List of columns
   * @param[in] segmentation stripe and rowgroup ranges
   * @param[in] decimal_column_sizes Sizes of encoded decimal columns
   * @return List of stream descriptors
   */
  orc_streams create_streams(host_span<orc_column_view> columns,
                             file_segmentation const& segmentation,
                             std::map<uint32_t, size_t> const& decimal_column_sizes);

  /**
   * @brief Encodes the input columns into streams.
   *
   * @param orc_table Non-owning view of a cuDF table w/ ORC-related info
   * @param dict_data Dictionary data memory
   * @param dict_index Dictionary index memory
   * @param dec_chunk_sizes Information about size of encoded decimal columns
   * @param segmentation stripe and rowgroup ranges
   * @param stream CUDA stream used for device memory operations and kernel launches
   * @return Encoded data and per-chunk stream descriptors
   */
  encoded_data encode_columns(orc_table_view const& orc_table,
                              string_dictionaries&& dictionaries,
                              encoder_decimal_info&& dec_chunk_sizes,
                              file_segmentation const& segmentation,
                              orc_streams const& streams);

  /**
   * @brief Returns stripe information after compacting columns' individual data
   * chunks into contiguous data streams.
   *
   * @param[in] num_index_streams Total number of index streams
   * @param[in] segmentation stripe and rowgroup ranges
   * @param[in,out] enc_streams List of encoder chunk streams [column][rowgroup]
   * @param[in,out] strm_desc List of stream descriptors [stripe][data_stream]
   *
   * @return The stripes' information
   */
  std::vector<StripeInformation> gather_stripes(
    size_t num_index_streams,
    file_segmentation const& segmentation,
    hostdevice_2dvector<gpu::encoder_chunk_streams>* enc_streams,
    hostdevice_2dvector<gpu::StripeStream>* strm_desc);

  /**
   * @brief Returns per-stripe and per-file column statistics encoded
   * in ORC protobuf format.
   *
   * @param orc_table Table information to be written
   * @param columns List of columns
   * @param segmentation stripe and rowgroup ranges
   * @return The statistic blobs
   */
  std::vector<std::vector<uint8_t>> gather_statistic_blobs(orc_table_view const& orc_table,
                                                           file_segmentation const& segmentation);

  /**
   * @brief Writes the specified column's row index stream.
   *
   * @param[in] stripe_id Stripe's identifier
   * @param[in] stream_id Stream identifier (column id + 1)
   * @param[in] columns List of columns
   * @param[in] rowgroups_range Indexes of rowgroups in the stripe
   * @param[in] enc_streams List of encoder chunk streams [column][rowgroup]
   * @param[in] strm_desc List of stream descriptors
   * @param[in] comp_out Output status for compressed streams
   * @param[in,out] stripe Stream's parent stripe
   * @param[in,out] streams List of all streams
   * @param[in,out] pbw Protobuf writer
   */
  void write_index_stream(int32_t stripe_id,
                          int32_t stream_id,
                          host_span<orc_column_view const> columns,
                          stripe_rowgroups const& rowgroups_range,
                          host_2dspan<gpu::encoder_chunk_streams const> enc_streams,
                          host_2dspan<gpu::StripeStream const> strm_desc,
                          host_span<gpu_inflate_status_s const> comp_out,
                          StripeInformation* stripe,
                          orc_streams* streams,
                          ProtobufWriter* pbw);

  /**
   * @brief Write the specified column's data streams
   *
   * @param[in] strm_desc Stream's descriptor
   * @param[in] enc_stream Chunk's streams
   * @param[in] compressed_data Compressed stream data
   * @param[in,out] stream_out Temporary host output buffer
   * @param[in,out] stripe Stream's parent stripe
   * @param[in,out] streams List of all streams
   */
  void write_data_stream(gpu::StripeStream const& strm_desc,
                         gpu::encoder_chunk_streams const& enc_stream,
                         uint8_t const* compressed_data,
                         uint8_t* stream_out,
                         StripeInformation* stripe,
                         orc_streams* streams);

  /**
   * @brief Insert 3-byte uncompressed block headers in a byte vector
   *
   * @param byte_vector Raw data (must include initial 3-byte header)
   */
  void add_uncompressed_block_headers(std::vector<uint8_t>& byte_vector);

 private:
  rmm::mr::device_memory_resource* _mr = nullptr;
  // Cuda stream to be used
  rmm::cuda_stream_view stream = rmm::cuda_stream_default;

  size_t max_stripe_size_           = DEFAULT_STRIPE_SIZE;
  size_t row_index_stride_          = default_row_index_stride;
  size_t compression_blocksize_     = DEFAULT_COMPRESSION_BLOCKSIZE;
  CompressionKind compression_kind_ = CompressionKind::NONE;

  bool enable_dictionary_ = true;
  bool enable_statistics_ = true;

  // Overall file metadata.  Filled in during the process and written during write_chunked_end()
  cudf::io::orc::FileFooter ff;
  cudf::io::orc::Metadata md;
  // current write position for rowgroups/chunks
  size_t current_chunk_offset;
  // optional user metadata
  table_metadata const* user_metadata = nullptr;
  // only used in the write_chunked() case. copied from the (optionally) user supplied
  // argument to write_chunked_begin()
  table_metadata_with_nullability user_metadata_with_nullability;
  // special parameter only used by detail::write() to indicate that we are guaranteeing
  // a single table write.  this enables some internal optimizations.
  bool const single_write_mode;
  // to track if the output has been written to sink
  bool closed = false;

  std::vector<uint8_t> buffer_;
  std::unique_ptr<data_sink> out_sink_;
};

}  // namespace orc
}  // namespace detail
}  // namespace io
}  // namespace cudf
