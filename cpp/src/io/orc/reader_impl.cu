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
 * @file reader_impl.cu
 * @brief cuDF-IO ORC reader class implementation
 **/

#include "reader_impl.hpp"
#include "timezone.h"

#include <io/comp/gpuinflate.h>

#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/device_buffer.hpp>

#include <algorithm>
#include <array>

namespace cudf {
namespace experimental {
namespace io {
namespace detail {
namespace orc {

// Import functionality that's independent of legacy code
using namespace cudf::io::orc;
using namespace cudf::io;

namespace {

/**
 * @brief Function that translates ORC data kind to cuDF type enum
 **/
constexpr type_id to_type_id(const orc::SchemaType &schema, bool use_np_dtypes,
                             type_id timestamp_type_id, bool decimals_as_float) {
  switch (schema.kind) {
    case orc::BOOLEAN:
      return type_id::BOOL8;
    case orc::BYTE:
      return type_id::INT8;
    case orc::SHORT:
      return type_id::INT16;
    case orc::INT:
      return type_id::INT32;
    case orc::LONG:
      return type_id::INT64;
    case orc::FLOAT:
      return type_id::FLOAT32;
    case orc::DOUBLE:
      return type_id::FLOAT64;
    case orc::STRING:
    case orc::BINARY:
    case orc::VARCHAR:
    case orc::CHAR:
      // Variable-length types can all be mapped to STRING
      return type_id::STRING;
    case orc::TIMESTAMP:
      return (timestamp_type_id != type_id::EMPTY)
                 ? timestamp_type_id
                 : type_id::TIMESTAMP_NANOSECONDS;
    case orc::DATE:
      // There isn't a (DAYS -> np.dtype) mapping
      return (use_np_dtypes) ? type_id::TIMESTAMP_MILLISECONDS
                             : type_id::TIMESTAMP_DAYS;
    case orc::DECIMAL:
      // There isn't an arbitrary-precision type in cuDF, so map as float or int
      return (decimals_as_float) ? type_id::FLOAT64
                                 : type_id::INT64;
    default:
      break;
  }

  return type_id::EMPTY;
}

/**
 * @brief Function that translates cuDF time unit to ORC clock frequency
 **/
constexpr int32_t to_clockrate(type_id timestamp_type_id) {
  switch (timestamp_type_id) {
    case type_id::TIMESTAMP_SECONDS:
      return 1;
    case type_id::TIMESTAMP_MICROSECONDS:
      return 1000;
    case type_id::TIMESTAMP_MILLISECONDS:
      return 1000000;
    case type_id::TIMESTAMP_NANOSECONDS:
      return 1000000000;
    default:
      return 0;
  }
}

constexpr std::pair<gpu::StreamIndexType, uint32_t> get_index_type_and_pos(
    const orc::StreamKind kind, uint32_t skip_count, bool non_child) {
  switch (kind) {
    case orc::DATA:
      skip_count += 1;
      skip_count |= (skip_count & 0xff) << 8;
      return std::make_pair(gpu::CI_DATA, skip_count);
    case orc::LENGTH:
    case orc::SECONDARY:
      skip_count += 1;
      skip_count |= (skip_count & 0xff) << 16;
      return std::make_pair(gpu::CI_DATA2, skip_count);
    case orc::DICTIONARY_DATA:
      return std::make_pair(gpu::CI_DICTIONARY, skip_count);
    case orc::PRESENT:
      skip_count += (non_child ? 1 : 0);
      return std::make_pair(gpu::CI_PRESENT, skip_count);
    case orc::ROW_INDEX:
      return std::make_pair(gpu::CI_INDEX, skip_count);
    default:
      // Skip this stream as it's not strictly required
      return std::make_pair(gpu::CI_NUM_STREAMS, 0);
  }
}

}  // namespace

/**
 * @brief A helper class for ORC file metadata. Provides some additional
 * convenience methods for initializing and accessing metadata.
 **/
class metadata {
  using OrcStripeInfo =
      std::pair<const StripeInformation *, const StripeFooter *>;

 public:
  explicit metadata(datasource *const src) : source(src) {
    const auto len = source->size();
    const auto max_ps_size = std::min(len, static_cast<size_t>(256));

    // Read uncompressed postscript section (max 255 bytes + 1 byte for length)
    auto buffer = source->get_buffer(len - max_ps_size, max_ps_size);
    const size_t ps_length = buffer->data()[max_ps_size - 1];
    const uint8_t *ps_data = &buffer->data()[max_ps_size - ps_length - 1];
    ProtobufReader pb;
    pb.init(ps_data, ps_length);
    CUDF_EXPECTS(pb.read(&ps, ps_length), "Cannot read postscript");
    CUDF_EXPECTS(ps.footerLength + ps_length < len, "Invalid footer length");

    // If compression is used, all the rest of the metadata is compressed
    // If no compressed is used, the decompressor is simply a pass-through
    decompressor = std::make_unique<OrcDecompressor>(ps.compression,
                                                     ps.compressionBlockSize);

    // Read compressed filefooter section
    buffer = source->get_buffer(len - ps_length - 1 - ps.footerLength,
                                ps.footerLength);
    size_t ff_length = 0;
    auto ff_data =
        decompressor->Decompress(buffer->data(), ps.footerLength, &ff_length);
    pb.init(ff_data, ff_length);
    CUDF_EXPECTS(pb.read(&ff, ff_length), "Cannot read filefooter");
    CUDF_EXPECTS(get_num_columns() > 0, "No columns found");
  }

  /**
   * @brief Filters and reads the info of only a selection of stripes
   *
   * @param[in] stripe Index of the stripe to select
   * @param[in] row_start Starting row of the selection
   * @param[in,out] row_count Total number of rows selected
   *
   * @return List of stripe info and total number of selected rows
   **/
  auto select_stripes(int stripe, int &row_start, int &row_count) {
    std::vector<OrcStripeInfo> selection;

    if (stripe != -1) {
      CUDF_EXPECTS(stripe < get_num_stripes(), "Non-existent stripe");
      selection.emplace_back(&ff.stripes[stripe], nullptr);
      if (row_count < 0) {
        row_count = ff.stripes[stripe].numberOfRows;
      } else {
        row_count = std::min(row_count, (int)ff.stripes[stripe].numberOfRows);
      }
    } else {
      row_start = std::max(row_start, 0);
      if (row_count == -1) {
        row_count = get_total_rows();
      }
      CUDF_EXPECTS(row_count >= 0, "Invalid row count");
      CUDF_EXPECTS(row_start <= get_total_rows(), "Invalid row start");

      int stripe_skip_rows = 0;
      for (int i = 0, count = 0; i < (int)ff.stripes.size(); ++i) {
        count += ff.stripes[i].numberOfRows;
        if (count > row_start) {
          if (selection.size() == 0) {
            stripe_skip_rows = row_start - (count - ff.stripes[i].numberOfRows);
          }
          selection.emplace_back(&ff.stripes[i], nullptr);
        }
        if (count >= (row_start + row_count)) {
          break;
        }
      }
      row_start = stripe_skip_rows;
    }

    // Read each stripe's stripefooter metadata
    if (not selection.empty()) {
      orc::ProtobufReader pb;

      stripefooters.resize(selection.size());
      for (size_t i = 0; i < selection.size(); ++i) {
        const auto stripe = selection[i].first;
        const auto sf_comp_offset =
            stripe->offset + stripe->indexLength + stripe->dataLength;
        const auto sf_comp_length = stripe->footerLength;
        CUDF_EXPECTS(sf_comp_offset + sf_comp_length < source->size(),
                     "Invalid stripe information");

        const auto buffer = source->get_buffer(sf_comp_offset, sf_comp_length);
        size_t sf_length = 0;
        auto sf_data = decompressor->Decompress(buffer->data(), sf_comp_length,
                                                &sf_length);
        pb.init(sf_data, sf_length);
        CUDF_EXPECTS(pb.read(&stripefooters[i], sf_length),
                     "Cannot read stripefooter");
        selection[i].second = &stripefooters[i];
      }
    }

    return selection;
  }

  /**
   * @brief Filters and reduces down to a selection of columns
   *
   * @param[in] use_names List of column names to select
   * @param[out] has_timestamp_column Whether there is a orc::TIMESTAMP column
   *
   * @return List of ORC column indexes
   **/
  auto select_columns(std::vector<std::string> use_names,
                      bool &has_timestamp_column) {
    std::vector<int> selection;

    if (not use_names.empty()) {
      int index = 0;
      for (const auto &use_name : use_names) {
        for (int i = 0; i < get_num_columns(); ++i, ++index) {
          if (index >= get_num_columns()) {
            index = 0;
          }
          if (ff.GetColumnName(index) == use_name) {
            selection.emplace_back(index);
            index++;
            if (ff.types[i].kind == orc::TIMESTAMP) {
              has_timestamp_column = true;
            }
            break;
          }
        }
      }
    } else {
      // For now, only select all leaf nodes
      for (int i = 0; i < get_num_columns(); ++i) {
        if (ff.types[i].subtypes.size() == 0) {
          selection.emplace_back(i);
          if (ff.types[i].kind == orc::TIMESTAMP) {
            has_timestamp_column = true;
          }
        }
      }
    }
    CUDF_EXPECTS(selection.size() > 0, "Filtered out all columns");

    return selection;
  }

  inline int get_total_rows() const { return ff.numberOfRows; }
  inline int get_num_stripes() const { return ff.stripes.size(); }
  inline int get_num_columns() const { return ff.types.size(); }
  inline int get_row_index_stride() const { return ff.rowIndexStride; }

 public:
  PostScript ps;
  FileFooter ff;
  std::vector<StripeFooter> stripefooters;
  std::unique_ptr<OrcDecompressor> decompressor;

 private:
  datasource *const source;
};

namespace {

/**
 * @brief Struct that maps ORC streams to columns
 **/
struct orc_stream_info {
  orc_stream_info() = default;
  explicit orc_stream_info(uint64_t offset_, size_t dst_pos_, uint32_t length_,
                           uint32_t gdf_idx_, uint32_t stripe_idx_)
      : offset(offset_),
        dst_pos(dst_pos_),
        length(length_),
        gdf_idx(gdf_idx_),
        stripe_idx(stripe_idx_) {}
  uint64_t offset;  // offset in file
  size_t
      dst_pos;  // offset in memory relative to start of compressed stripe data
  uint32_t length;      // length in file
  uint32_t gdf_idx;     // gdf column index
  uint32_t stripe_idx;  // stripe index
};

/**
 * @brief Function that populates column descriptors stream/chunk
 **/
size_t gather_stream_info(const size_t stripe_index,
                          const orc::StripeInformation *stripeinfo,
                          const orc::StripeFooter *stripefooter,
                          const std::vector<int> &orc2gdf,
                          const std::vector<int> &gdf2orc,
                          const std::vector<orc::SchemaType> types,
                          bool use_index, size_t *num_dictionary_entries,
                          hostdevice_vector<gpu::ColumnDesc> &chunks,
                          std::vector<orc_stream_info> &stream_info) {
  const auto num_columns = gdf2orc.size();
  uint64_t src_offset = 0;
  uint64_t dst_offset = 0;
  for (const auto &stream : stripefooter->streams) {
    if (stream.column >= orc2gdf.size()) {
      dst_offset += stream.length;
      continue;
    }

    auto col = orc2gdf[stream.column];
    if (col == -1) {
      // A struct-type column has no data itself, but rather child columns
      // for each of its fields. There is only a PRESENT stream, which
      // needs to be included for the reader.
      const auto schema_type = types[stream.column];
      if (schema_type.subtypes.size() != 0) {
        if (schema_type.kind == orc::STRUCT && stream.kind == orc::PRESENT) {
          for (const auto &idx : schema_type.subtypes) {
            auto child_idx = (idx < orc2gdf.size()) ? orc2gdf[idx] : -1;
            if (child_idx >= 0) {
              col = child_idx;
              auto &chunk = chunks[stripe_index * num_columns + col];
              chunk.strm_id[gpu::CI_PRESENT] = stream_info.size();
              chunk.strm_len[gpu::CI_PRESENT] = stream.length;
            }
          }
        }
      }
    }
    if (col != -1) {
      if (src_offset >= stripeinfo->indexLength || use_index) {
        // NOTE: skip_count field is temporarily used to track index ordering
        auto &chunk = chunks[stripe_index * num_columns + col];
        const auto idx = get_index_type_and_pos(stream.kind, chunk.skip_count,
                                                col == orc2gdf[stream.column]);
        if (idx.first < gpu::CI_NUM_STREAMS) {
          chunk.strm_id[idx.first] = stream_info.size();
          chunk.strm_len[idx.first] = stream.length;
          chunk.skip_count = idx.second;

          if (idx.first == gpu::CI_DICTIONARY) {
            chunk.dictionary_start = *num_dictionary_entries;
            chunk.dict_len =
                stripefooter->columns[stream.column].dictionarySize;
            *num_dictionary_entries +=
                stripefooter->columns[stream.column].dictionarySize;
          }
        }
      }
      stream_info.emplace_back(stripeinfo->offset + src_offset, dst_offset,
                               stream.length, col, stripe_index);
      dst_offset += stream.length;
    }
    src_offset += stream.length;
  }

  return dst_offset;
}

}  // namespace

rmm::device_buffer reader::impl::decompress_stripe_data(
    const hostdevice_vector<gpu::ColumnDesc> &chunks,
    const std::vector<rmm::device_buffer> &stripe_data,
    const OrcDecompressor *decompressor,
    std::vector<orc_stream_info> &stream_info, size_t num_stripes,
    rmm::device_vector<gpu::RowGroup> &row_groups, size_t row_index_stride,
    cudaStream_t stream) {
  // Parse the columns' compressed info
  hostdevice_vector<gpu::CompressedStreamInfo> compinfo(0, stream_info.size(),
                                                        stream);
  for (const auto &info : stream_info) {
    compinfo.insert(gpu::CompressedStreamInfo(
        static_cast<const uint8_t *>(stripe_data[info.stripe_idx].data()) +
            info.dst_pos,
        info.length));
  }
  CUDA_TRY(cudaMemcpyAsync(compinfo.device_ptr(), compinfo.host_ptr(),
                           compinfo.memory_size(), cudaMemcpyHostToDevice,
                           stream));
  CUDA_TRY(gpu::ParseCompressedStripeData(
      compinfo.device_ptr(), compinfo.size(), decompressor->GetBlockSize(),
      decompressor->GetLog2MaxCompressionRatio(), stream));
  CUDA_TRY(cudaMemcpyAsync(compinfo.host_ptr(), compinfo.device_ptr(),
                           compinfo.memory_size(), cudaMemcpyDeviceToHost,
                           stream));
  CUDA_TRY(cudaStreamSynchronize(stream));

  // Count the exact number of compressed blocks
  size_t num_compressed_blocks = 0;
  size_t num_uncompressed_blocks = 0;
  size_t total_decomp_size = 0;
  for (size_t i = 0; i < compinfo.size(); ++i) {
    num_compressed_blocks += compinfo[i].num_compressed_blocks;
    num_uncompressed_blocks += compinfo[i].num_uncompressed_blocks;
    total_decomp_size += compinfo[i].max_uncompressed_size;
  }
  CUDF_EXPECTS(total_decomp_size > 0, "No decompressible data found");

  rmm::device_buffer decomp_data(total_decomp_size, stream);
  rmm::device_vector<gpu_inflate_input_s> inflate_in(num_compressed_blocks +
                                                     num_uncompressed_blocks);
  rmm::device_vector<gpu_inflate_status_s> inflate_out(num_compressed_blocks);

  // Parse again to populate the decompression input/output buffers
  size_t decomp_offset = 0;
  uint32_t start_pos = 0;
  uint32_t start_pos_uncomp = (uint32_t)num_compressed_blocks;
  for (size_t i = 0; i < compinfo.size(); ++i) {
    auto dst_base = static_cast<uint8_t *>(decomp_data.data());
    compinfo[i].uncompressed_data = dst_base + decomp_offset;
    compinfo[i].decctl = inflate_in.data().get() + start_pos;
    compinfo[i].decstatus = inflate_out.data().get() + start_pos;
    compinfo[i].copyctl = inflate_in.data().get() + start_pos_uncomp;

    stream_info[i].dst_pos = decomp_offset;
    decomp_offset += compinfo[i].max_uncompressed_size;
    start_pos += compinfo[i].num_compressed_blocks;
    start_pos_uncomp += compinfo[i].num_uncompressed_blocks;
  }
  CUDA_TRY(cudaMemcpyAsync(compinfo.device_ptr(), compinfo.host_ptr(),
                           compinfo.memory_size(), cudaMemcpyHostToDevice,
                           stream));
  CUDA_TRY(gpu::ParseCompressedStripeData(
      compinfo.device_ptr(), compinfo.size(), decompressor->GetBlockSize(),
      decompressor->GetLog2MaxCompressionRatio(), stream));

  // Dispatch batches of blocks to decompress
  if (num_compressed_blocks > 0) {
    switch (decompressor->GetKind()) {
      case orc::ZLIB:
        CUDA_TRY(gpuinflate(inflate_in.data().get(), inflate_out.data().get(),
                            num_compressed_blocks, 0, stream));
        break;
      case orc::SNAPPY:
        CUDA_TRY(gpu_unsnap(inflate_in.data().get(), inflate_out.data().get(),
                            num_compressed_blocks, stream));
        break;
      default:
        CUDF_EXPECTS(false, "Unexpected decompression dispatch");
        break;
    }
  }
  if (num_uncompressed_blocks > 0) {
    CUDA_TRY(gpu_copy_uncompressed_blocks(
        inflate_in.data().get() + num_compressed_blocks,
        num_uncompressed_blocks, stream));
  }
  CUDA_TRY(gpu::PostDecompressionReassemble(compinfo.device_ptr(),
                                            compinfo.size(), stream));

  // Update the stream information with the updated uncompressed info
  // TBD: We could update the value from the information we already
  // have in stream_info[], but using the gpu results also updates
  // max_uncompressed_size to the actual uncompressed size, or zero if
  // decompression failed.
  CUDA_TRY(cudaMemcpyAsync(compinfo.host_ptr(), compinfo.device_ptr(),
                           compinfo.memory_size(), cudaMemcpyDeviceToHost,
                           stream));
  CUDA_TRY(cudaStreamSynchronize(stream));

  const size_t num_columns = chunks.size() / num_stripes;

  for (size_t i = 0; i < num_stripes; ++i) {
    for (size_t j = 0; j < num_columns; ++j) {
      auto &chunk = chunks[i * num_columns + j];
      for (int k = 0; k < gpu::CI_NUM_STREAMS; ++k) {
        if (chunk.strm_len[k] > 0 && chunk.strm_id[k] < compinfo.size()) {
          chunk.streams[k] = compinfo[chunk.strm_id[k]].uncompressed_data;
          chunk.strm_len[k] = compinfo[chunk.strm_id[k]].max_uncompressed_size;
        }
      }
    }
  }

  if (not row_groups.empty()) {
    CUDA_TRY(cudaMemcpyAsync(chunks.device_ptr(), chunks.host_ptr(),
                             chunks.memory_size(), cudaMemcpyHostToDevice,
                             stream));
    CUDA_TRY(gpu::ParseRowGroupIndex(
        row_groups.data().get(), compinfo.device_ptr(), chunks.device_ptr(),
        num_columns, num_stripes, row_groups.size() / num_columns,
        row_index_stride, stream));
  }

  return decomp_data;
}

void reader::impl::decode_stream_data(
    const hostdevice_vector<gpu::ColumnDesc> &chunks, size_t num_dicts,
    size_t skip_rows, size_t num_rows,
    const std::vector<int64_t> &timezone_table,
    const rmm::device_vector<gpu::RowGroup> &row_groups,
    size_t row_index_stride, std::vector<column_buffer> &out_buffers,
    cudaStream_t stream) {
  const auto num_columns = out_buffers.size();
  const auto num_stripes = chunks.size() / out_buffers.size();

  // Update chunks with pointers to column data
  for (size_t i = 0; i < num_stripes; ++i) {
    for (size_t j = 0; j < num_columns; ++j) {
      auto &chunk = chunks[i * num_columns + j];
      chunk.column_data_base = out_buffers[j].data();
      chunk.valid_map_base = out_buffers[j].null_mask();
    }
  }

  // Allocate global dictionary for deserializing
  rmm::device_vector<gpu::DictionaryEntry> global_dict(num_dicts);

  // Allocate timezone transition table timestamp conversion
  rmm::device_vector<int64_t> tz_table = timezone_table;

  CUDA_TRY(cudaMemcpyAsync(chunks.device_ptr(), chunks.host_ptr(),
                           chunks.memory_size(), cudaMemcpyHostToDevice,
                           stream));
  CUDA_TRY(gpu::DecodeNullsAndStringDictionaries(
      chunks.device_ptr(), global_dict.data().get(), num_columns, num_stripes,
      num_rows, skip_rows, stream));
  CUDA_TRY(gpu::DecodeOrcColumnData(
      chunks.device_ptr(), global_dict.data().get(), num_columns, num_stripes,
      num_rows, skip_rows, tz_table.data().get(), tz_table.size(),
      row_groups.data().get(), row_groups.size() / num_columns,
      row_index_stride, stream));
  CUDA_TRY(cudaMemcpyAsync(chunks.host_ptr(), chunks.device_ptr(),
                           chunks.memory_size(), cudaMemcpyDeviceToHost,
                           stream));
  CUDA_TRY(cudaStreamSynchronize(stream));

  for (size_t i = 0; i < num_stripes; ++i) {
    for (size_t j = 0; j < num_columns; ++j) {
      out_buffers[j].null_count() += chunks[i * num_columns + j].null_count;
    }
  }
}

reader::impl::impl(std::unique_ptr<datasource> source,
                   reader_options const &options,
                   rmm::mr::device_memory_resource *mr)
    : _source(std::move(source)), _mr(mr) {
  // Open and parse the source dataset metadata
  _metadata = std::make_unique<metadata>(_source.get());

  // Select only columns required by the options
  _selected_columns =
      _metadata->select_columns(options.columns, _has_timestamp_column);

  // Override output timestamp resolution if requested
  if (options.timestamp_type.id() != EMPTY) {
    _timestamp_type = options.timestamp_type;
  }

  // Enable or disable attempt to use row index for parsing
  _use_index = options.use_index;

  // Enable or disable the conversion to numpy-compatible dtypes
  _use_np_dtypes = options.use_np_dtypes;

  // Control decimals conversion (float64 or int64 with optional scale)
  _decimals_as_float = options.decimals_as_float;
  _decimals_as_int_scale = options.forced_decimals_scale;
}

table_with_metadata reader::impl::read(int skip_rows, int num_rows, int stripe,
                                       cudaStream_t stream) {
  std::vector<std::unique_ptr<column>> out_columns;
  table_metadata out_metadata;

  // Select only stripes required (aka row groups)
  const auto selected_stripes =
      _metadata->select_stripes(stripe, skip_rows, num_rows);

  // Association between each ORC column and its gdf_column
  std::vector<int32_t> orc_col_map(_metadata->get_num_columns(), -1);

  // Get a list of column data types
  std::vector<data_type> column_types;
  for (const auto &col : _selected_columns) {
    auto col_type = to_type_id(_metadata->ff.types[col], _use_np_dtypes,
                               _timestamp_type.id(), _decimals_as_float);
    CUDF_EXPECTS(col_type != type_id::EMPTY, "Unknown type");
    column_types.emplace_back(col_type);

    // Map each ORC column to its column
    orc_col_map[col] = column_types.size() - 1;
  }

  if (num_rows > 0 && selected_stripes.size() != 0) {
    const auto num_columns = _selected_columns.size();
    const auto num_chunks = selected_stripes.size() * num_columns;
    hostdevice_vector<gpu::ColumnDesc> chunks(num_chunks, stream);
    memset(chunks.host_ptr(), 0, chunks.memory_size());

    const bool use_index =
        (_use_index == true) &&
        // Only use if we don't have much work with complete columns & stripes
        // TODO: Consider nrows, gpu, and tune the threshold
        (num_rows > _metadata->get_row_index_stride() &&
         !(_metadata->get_row_index_stride() & 7) &&
         _metadata->get_row_index_stride() > 0 &&
         num_columns * selected_stripes.size() < 8 * 128) &&
        // Only use if first row is aligned to a stripe boundary
        // TODO: Fix logic to handle unaligned rows
        (skip_rows == 0);

    // Logically view streams as columns
    std::vector<orc_stream_info> stream_info;

    // Tracker for eventually deallocating compressed and uncompressed data
    std::vector<rmm::device_buffer> stripe_data;

    size_t stripe_start_row = 0;
    size_t num_dict_entries = 0;
    size_t num_rowgroups = 0;
    for (size_t i = 0; i < selected_stripes.size(); ++i) {
      const auto stripe_info = selected_stripes[i].first;
      const auto stripe_footer = selected_stripes[i].second;

      auto stream_count = stream_info.size();
      const auto total_data_size =
          gather_stream_info(i, stripe_info, stripe_footer, orc_col_map,
                             _selected_columns, _metadata->ff.types, use_index,
                             &num_dict_entries, chunks, stream_info);
      CUDF_EXPECTS(total_data_size > 0, "Expected streams data within stripe");

      stripe_data.emplace_back(total_data_size, stream);
      auto dst_base = static_cast<uint8_t *>(stripe_data.back().data());

      // Coalesce consecutive streams into one read
      while (stream_count < stream_info.size()) {
        const auto d_dst = dst_base + stream_info[stream_count].dst_pos;
        const auto offset = stream_info[stream_count].offset;
        auto len = stream_info[stream_count].length;
        stream_count++;

        while (stream_count < stream_info.size() &&
               stream_info[stream_count].offset == offset + len) {
          len += stream_info[stream_count].length;
          stream_count++;
        }
        const auto buffer = _source->get_buffer(offset, len);
        CUDA_TRY(cudaMemcpyAsync(d_dst, buffer->data(), len,
                                 cudaMemcpyHostToDevice, stream));
        CUDA_TRY(cudaStreamSynchronize(stream));
      }

      // Update chunks to reference streams pointers
      for (size_t j = 0; j < num_columns; j++) {
        auto &chunk = chunks[i * num_columns + j];
        chunk.start_row = stripe_start_row;
        chunk.num_rows = stripe_info->numberOfRows;
        chunk.encoding_kind = stripe_footer->columns[_selected_columns[j]].kind;
        chunk.type_kind = _metadata->ff.types[_selected_columns[j]].kind;
        if (_decimals_as_float) {
          chunk.decimal_scale = _metadata->ff.types[_selected_columns[j]].scale | ORC_DECIMAL2FLOAT64_SCALE;
        }
        else if (_decimals_as_int_scale < 0) {
          chunk.decimal_scale = _metadata->ff.types[_selected_columns[j]].scale;
        }
        else {
          chunk.decimal_scale = _decimals_as_int_scale;
        }
        chunk.rowgroup_id = num_rowgroups;
        chunk.dtype_len = (column_types[j].id() == type_id::STRING)
                              ? sizeof(std::pair<const char *, size_t>)
                              : cudf::size_of(column_types[j]);
        if (chunk.type_kind == orc::TIMESTAMP) {
          chunk.ts_clock_rate = to_clockrate(_timestamp_type.id());
        }
        for (int k = 0; k < gpu::CI_NUM_STREAMS; k++) {
          if (chunk.strm_len[k] > 0) {
            chunk.streams[k] = dst_base + stream_info[chunk.strm_id[k]].dst_pos;
          }
        }
      }
      stripe_start_row += stripe_info->numberOfRows;
      if (use_index) {
        num_rowgroups += (stripe_info->numberOfRows +
                          _metadata->get_row_index_stride() - 1) /
                         _metadata->get_row_index_stride();
      }
    }

    // Process dataset chunk pages into output columns
    if (stripe_data.size() != 0) {
      // Setup row group descriptors if using indexes
      rmm::device_vector<gpu::RowGroup> row_groups(num_rowgroups * num_columns);
      if (_metadata->ps.compression != orc::NONE) {
        auto decomp_data = decompress_stripe_data(
            chunks, stripe_data, _metadata->decompressor.get(), stream_info,
            selected_stripes.size(), row_groups,
            _metadata->get_row_index_stride(), stream);
        stripe_data.clear();
        stripe_data.push_back(std::move(decomp_data));
      } else {
        if (not row_groups.empty()) {
          CUDA_TRY(cudaMemcpyAsync(chunks.device_ptr(), chunks.host_ptr(),
                                   chunks.memory_size(), cudaMemcpyHostToDevice,
                                   stream));
          CUDA_TRY(gpu::ParseRowGroupIndex(
              row_groups.data().get(), nullptr, chunks.device_ptr(),
              num_columns, selected_stripes.size(), num_rowgroups,
              _metadata->get_row_index_stride(), stream));
        }
      }

      // Setup table for converting timestamp columns from local to UTC time
      std::vector<int64_t> tz_table;
      if (_has_timestamp_column) {
        CUDF_EXPECTS(BuildTimezoneTransitionTable(
                         tz_table, selected_stripes[0].second->writerTimezone),
                     "Cannot setup timezone LUT");
      }

      std::vector<column_buffer> out_buffers;
      for (size_t i = 0; i < column_types.size(); ++i) {
        out_buffers.emplace_back(column_types[i], num_rows, stream, _mr);
      }

      decode_stream_data(
          chunks, num_dict_entries, skip_rows, num_rows, tz_table, row_groups,
          _metadata->get_row_index_stride(), out_buffers, stream);

      for (size_t i = 0; i < column_types.size(); ++i) {
        out_columns.emplace_back(make_column(column_types[i], num_rows,
                                             out_buffers[i], stream, _mr));
      }
    }
  }

  // Return column names (must match order of returned columns)
  out_metadata.column_names.resize(_selected_columns.size());
  for (size_t i = 0; i < _selected_columns.size(); i++) {
    out_metadata.column_names[i] = _metadata->ff.GetColumnName(_selected_columns[i]);
  }
  // Return user metadata
  for (const auto& kv : _metadata->ff.metadata) {
    out_metadata.user_data.insert({kv.name, kv.value});
  }

  return { std::make_unique<table>(std::move(out_columns)), std::move(out_metadata) };
}

// Forward to implementation
reader::reader(std::string filepath, reader_options const &options,
               rmm::mr::device_memory_resource *mr)
    : _impl(std::make_unique<impl>(datasource::create(filepath), options, mr)) {
}

// Forward to implementation
reader::reader(const char *buffer, size_t length, reader_options const &options,
               rmm::mr::device_memory_resource *mr)
    : _impl(std::make_unique<impl>(datasource::create(buffer, length), options,
                                   mr)) {}

// Forward to implementation
reader::reader(std::shared_ptr<arrow::io::RandomAccessFile> file,
               reader_options const &options,
               rmm::mr::device_memory_resource *mr)
    : _impl(std::make_unique<impl>(datasource::create(file), options, mr)) {}

// Destructor within this translation unit
reader::~reader() = default;

// Forward to implementation
table_with_metadata reader::read_all(cudaStream_t stream) {
  return _impl->read(0, -1, -1, stream);
}

// Forward to implementation
table_with_metadata reader::read_stripe(size_type stripe,
                                        cudaStream_t stream) {
  return _impl->read(0, -1, stripe, stream);
}

// Forward to implementation
table_with_metadata reader::read_rows(size_type skip_rows,
                                      size_type num_rows,
                                      cudaStream_t stream) {
  return _impl->read(skip_rows, (num_rows != 0) ? num_rows : -1, -1, stream);
}

}  // namespace orc
}  // namespace detail
}  // namespace io
}  // namespace experimental
}  // namespace cudf
