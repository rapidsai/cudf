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

/**
 * @file reader_impl.cu
 * @brief cuDF-IO ORC reader class implementation
 */

#include "io/orc/orc_gpu.h"
#include "reader_impl.hpp"
#include "timezone.cuh"

#include <io/comp/gpuinflate.h>
#include "orc.h"

#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <iterator>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <array>

namespace cudf {
namespace io {
namespace detail {
namespace orc {
// Import functionality that's independent of legacy code
using namespace cudf::io::orc;
using namespace cudf::io;

namespace {
/**
 * @brief Function that translates ORC data kind to cuDF type enum
 */
constexpr type_id to_type_id(const orc::SchemaType &schema,
                             bool use_np_dtypes,
                             type_id timestamp_type_id,
                             bool decimals_as_float64)
{
  switch (schema.kind) {
    case orc::BOOLEAN: return type_id::BOOL8;
    case orc::BYTE: return type_id::INT8;
    case orc::SHORT: return type_id::INT16;
    case orc::INT: return type_id::INT32;
    case orc::LONG: return type_id::INT64;
    case orc::FLOAT: return type_id::FLOAT32;
    case orc::DOUBLE: return type_id::FLOAT64;
    case orc::STRING:
    case orc::BINARY:
    case orc::VARCHAR:
    case orc::CHAR:
      // Variable-length types can all be mapped to STRING
      return type_id::STRING;
    case orc::TIMESTAMP:
      return (timestamp_type_id != type_id::EMPTY) ? timestamp_type_id
                                                   : type_id::TIMESTAMP_NANOSECONDS;
    case orc::DATE:
      // There isn't a (DAYS -> np.dtype) mapping
      return (use_np_dtypes) ? type_id::TIMESTAMP_MILLISECONDS : type_id::TIMESTAMP_DAYS;
    case orc::DECIMAL: return (decimals_as_float64) ? type_id::FLOAT64 : type_id::DECIMAL64;
    default: break;
  }

  return type_id::EMPTY;
}

/**
 * @brief Function that translates cuDF time unit to ORC clock frequency
 */
constexpr int32_t to_clockrate(type_id timestamp_type_id)
{
  switch (timestamp_type_id) {
    case type_id::TIMESTAMP_SECONDS: return 1;
    case type_id::TIMESTAMP_MILLISECONDS: return 1000;
    case type_id::TIMESTAMP_MICROSECONDS: return 1000000;
    case type_id::TIMESTAMP_NANOSECONDS: return 1000000000;
    default: return 0;
  }
}

constexpr std::pair<gpu::StreamIndexType, uint32_t> get_index_type_and_pos(
  const orc::StreamKind kind, uint32_t skip_count, bool non_child)
{
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
    case orc::DICTIONARY_DATA: return std::make_pair(gpu::CI_DICTIONARY, skip_count);
    case orc::PRESENT:
      skip_count += (non_child ? 1 : 0);
      return std::make_pair(gpu::CI_PRESENT, skip_count);
    case orc::ROW_INDEX: return std::make_pair(gpu::CI_INDEX, skip_count);
    default:
      // Skip this stream as it's not strictly required
      return std::make_pair(gpu::CI_NUM_STREAMS, 0);
  }
}

}  // namespace

namespace {
/**
 * @brief Struct that maps ORC streams to columns
 */
struct orc_stream_info {
  orc_stream_info() = default;
  explicit orc_stream_info(
    uint64_t offset_, size_t dst_pos_, uint32_t length_, uint32_t gdf_idx_, uint32_t stripe_idx_)
    : offset(offset_),
      dst_pos(dst_pos_),
      length(length_),
      gdf_idx(gdf_idx_),
      stripe_idx(stripe_idx_)
  {
  }
  uint64_t offset;      // offset in file
  size_t dst_pos;       // offset in memory relative to start of compressed stripe data
  size_t length;        // length in file
  uint32_t gdf_idx;     // column index
  uint32_t stripe_idx;  // stripe index
};

/**
 * @brief Function that populates column descriptors stream/chunk
 */
size_t gather_stream_info(const size_t stripe_index,
                          const orc::StripeInformation *stripeinfo,
                          const orc::StripeFooter *stripefooter,
                          const std::vector<int> &orc2gdf,
                          const std::vector<int> &gdf2orc,
                          const std::vector<orc::SchemaType> types,
                          bool use_index,
                          size_t *num_dictionary_entries,
                          hostdevice_vector<gpu::ColumnDesc> &chunks,
                          std::vector<orc_stream_info> &stream_info)
{
  const auto num_columns = gdf2orc.size();
  uint64_t src_offset    = 0;
  uint64_t dst_offset    = 0;
  for (const auto &stream : stripefooter->streams) {
    if (!stream.column_id || *stream.column_id >= orc2gdf.size()) {
      dst_offset += stream.length;
      continue;
    }

    auto const column_id = *stream.column_id;
    auto col             = orc2gdf[column_id];

    if (col == -1) {
      // A struct-type column has no data itself, but rather child columns
      // for each of its fields. There is only a PRESENT stream, which
      // needs to be included for the reader.
      const auto schema_type = types[column_id];
      if (schema_type.subtypes.size() != 0) {
        if (schema_type.kind == orc::STRUCT && stream.kind == orc::PRESENT) {
          for (const auto &idx : schema_type.subtypes) {
            auto child_idx = (idx < orc2gdf.size()) ? orc2gdf[idx] : -1;
            if (child_idx >= 0) {
              col                             = child_idx;
              auto &chunk                     = chunks[stripe_index * num_columns + col];
              chunk.strm_id[gpu::CI_PRESENT]  = stream_info.size();
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
        const auto idx =
          get_index_type_and_pos(stream.kind, chunk.skip_count, col == orc2gdf[column_id]);
        if (idx.first < gpu::CI_NUM_STREAMS) {
          chunk.strm_id[idx.first]  = stream_info.size();
          chunk.strm_len[idx.first] = stream.length;
          chunk.skip_count          = idx.second;

          if (idx.first == gpu::CI_DICTIONARY) {
            chunk.dictionary_start = *num_dictionary_entries;
            chunk.dict_len         = stripefooter->columns[column_id].dictionarySize;
            *num_dictionary_entries += stripefooter->columns[column_id].dictionarySize;
          }
        }
      }
      stream_info.emplace_back(
        stripeinfo->offset + src_offset, dst_offset, stream.length, col, stripe_index);
      dst_offset += stream.length;
    }
    src_offset += stream.length;
  }

  return dst_offset;
}

/**
 * @brief Determines if a column should be converted from decimal to float
 */
bool should_convert_decimal_column_to_float(const std::vector<std::string> &columns_to_convert,
                                            cudf::io::orc::metadata &metadata,
                                            int column_index)
{
  return (std::find(columns_to_convert.begin(),
                    columns_to_convert.end(),
                    metadata.get_column_name(column_index)) != columns_to_convert.end());
}

}  // namespace

/**
 * @brief In order to support multiple input files/buffers we need to gather
 * the metadata across all of those input(s). This class provides a place
 * to aggregate that metadata from all the files.
 */
class aggregate_orc_metadata {
  using OrcStripeInfo = std::pair<const StripeInformation *, const StripeFooter *>;

 public:
  mutable std::vector<cudf::io::orc::metadata> per_file_metadata;
  size_type const num_rows;
  size_type const num_columns;
  size_type const num_stripes;

  /**
   * @brief Create a metadata object from each element in the source vector
   */
  auto metadatas_from_sources(std::vector<std::unique_ptr<datasource>> const &sources)
  {
    std::vector<cudf::io::orc::metadata> metadatas;
    std::transform(
      sources.cbegin(), sources.cend(), std::back_inserter(metadatas), [](auto const &source) {
        return cudf::io::orc::metadata(source.get());
      });
    return metadatas;
  }

  /**
   * @brief Sums up the number of rows of each source
   */
  size_type calc_num_rows() const
  {
    return std::accumulate(
      per_file_metadata.begin(), per_file_metadata.end(), 0, [](auto &sum, auto &pfm) {
        return sum + pfm.get_total_rows();
      });
  }

  /**
   * @brief Number of columns in a ORC file.
   */
  size_type calc_num_cols() const
  {
    if (not per_file_metadata.empty()) { return per_file_metadata[0].get_num_columns(); }
    return 0;
  }

  /**
   * @brief Sums up the number of stripes of each source
   */
  size_type calc_num_stripes() const
  {
    return std::accumulate(
      per_file_metadata.begin(), per_file_metadata.end(), 0, [](auto &sum, auto &pfm) {
        return sum + pfm.get_num_stripes();
      });
  }

  aggregate_orc_metadata(std::vector<std::unique_ptr<datasource>> const &sources)
    : per_file_metadata(metadatas_from_sources(sources)),
      num_rows(calc_num_rows()),
      num_columns(calc_num_cols()),
      num_stripes(calc_num_stripes())
  {
    // Verify that the input files have the same number of columns,
    // as well as matching types, compression, and names
    for (auto const &pfm : per_file_metadata) {
      CUDF_EXPECTS(per_file_metadata[0].get_num_columns() == pfm.get_num_columns(),
                   "All sources must have the same number of columns");
      CUDF_EXPECTS(per_file_metadata[0].ps.compression == pfm.ps.compression,
                   "All sources must have the same compression type");

      // Check the types, column names, and decimal scale
      for (size_t i = 0; i < pfm.ff.types.size(); i++) {
        CUDF_EXPECTS(pfm.ff.types[i].kind == per_file_metadata[0].ff.types[i].kind,
                     "Column types across all input sources must be the same");
        CUDF_EXPECTS(std::equal(pfm.ff.types[i].fieldNames.begin(),
                                pfm.ff.types[i].fieldNames.end(),
                                per_file_metadata[0].ff.types[i].fieldNames.begin()),
                     "All source column names must be the same");
        CUDF_EXPECTS(
          pfm.ff.types[i].scale.value_or(0) == per_file_metadata[0].ff.types[i].scale.value_or(0),
          "All scale values must be the same");
      }
    }
  }

  auto const &get_schema(int schema_idx) const { return per_file_metadata[0].ff.types[schema_idx]; }

  auto get_col_type(int col_idx) const { return per_file_metadata[0].ff.types[col_idx]; }

  auto get_num_rows() const { return num_rows; }

  auto get_num_cols() const { return per_file_metadata[0].get_num_columns(); }

  auto get_num_stripes() const { return num_stripes; }

  auto get_num_source_files() const { return per_file_metadata.size(); }

  auto const &get_types() const { return per_file_metadata[0].ff.types; }

  int get_row_index_stride() const { return per_file_metadata[0].ff.rowIndexStride; }

  auto get_column_name(const int source_idx, const int column_idx) const
  {
    CUDF_EXPECTS(source_idx <= static_cast<int>(per_file_metadata.size()),
                 "Out of range source_idx provided");
    CUDF_EXPECTS(column_idx <= per_file_metadata[source_idx].get_num_columns(),
                 "Out of range column_idx provided");
    return per_file_metadata[source_idx].get_column_name(column_idx);
  }

  std::vector<cudf::io::orc::metadata::stripe_source_mapping> select_stripes(
    std::vector<std::vector<size_type>> const &user_specified_stripes,
    size_type &row_start,
    size_type &row_count)
  {
    std::vector<cudf::io::orc::metadata::stripe_source_mapping> selected_stripes_mapping;

    if (!user_specified_stripes.empty()) {
      CUDF_EXPECTS(user_specified_stripes.size() == get_num_source_files(),
                   "Must specify stripes for each source");
      // row_start is 0 if stripes are set. If this is not true anymore, then
      // row_start needs to be subtracted to get the correct row_count
      CUDF_EXPECTS(row_start == 0, "Start row index should be 0");

      row_count = 0;
      // Each vector entry represents a source file; each nested vector represents the
      // user_defined_stripes to get from that source file
      for (size_t src_file_idx = 0; src_file_idx < user_specified_stripes.size(); ++src_file_idx) {
        std::vector<OrcStripeInfo> stripe_infos;

        // Coalesce stripe info at the source file later since that makes downstream processing much
        // easier in impl::read
        for (const size_t &stripe_idx : user_specified_stripes[src_file_idx]) {
          CUDF_EXPECTS(stripe_idx < per_file_metadata[src_file_idx].ff.stripes.size(),
                       "Invalid stripe index");
          stripe_infos.push_back(
            std::make_pair(&per_file_metadata[src_file_idx].ff.stripes[stripe_idx], nullptr));
          row_count += per_file_metadata[src_file_idx].ff.stripes[stripe_idx].numberOfRows;
        }
        selected_stripes_mapping.push_back({static_cast<int>(src_file_idx), stripe_infos});
      }
    } else {
      row_start = std::max(row_start, 0);
      if (row_count < 0) {
        row_count = static_cast<size_type>(
          std::min<int64_t>(get_num_rows(), std::numeric_limits<size_type>::max()));
      }
      row_count = std::min(row_count, get_num_rows() - row_start);
      CUDF_EXPECTS(row_count >= 0, "Invalid row count");
      CUDF_EXPECTS(row_start <= get_num_rows(), "Invalid row start");

      size_type count = 0;
      // Iterate all source files, each source file has corelating metadata
      for (size_t src_file_idx = 0;
           src_file_idx < per_file_metadata.size() && count < row_start + row_count;
           ++src_file_idx) {
        std::vector<OrcStripeInfo> stripe_infos;

        for (size_t stripe_idx = 0;
             stripe_idx < per_file_metadata[src_file_idx].ff.stripes.size() &&
             count < row_start + row_count;
             ++stripe_idx) {
          count += per_file_metadata[src_file_idx].ff.stripes[stripe_idx].numberOfRows;
          if (count > row_start || count == 0) {
            stripe_infos.push_back(
              std::make_pair(&per_file_metadata[src_file_idx].ff.stripes[stripe_idx], nullptr));
          }
        }

        selected_stripes_mapping.push_back({static_cast<int>(src_file_idx), stripe_infos});
      }
    }

    // Read each stripe's stripefooter metadata
    if (not selected_stripes_mapping.empty()) {
      for (auto &mapping : selected_stripes_mapping) {
        // Resize to all stripe_info for the source level
        per_file_metadata[mapping.source_idx].stripefooters.resize(mapping.stripe_info.size());

        for (size_t i = 0; i < mapping.stripe_info.size(); i++) {
          const auto stripe         = mapping.stripe_info[i].first;
          const auto sf_comp_offset = stripe->offset + stripe->indexLength + stripe->dataLength;
          const auto sf_comp_length = stripe->footerLength;
          CUDF_EXPECTS(
            sf_comp_offset + sf_comp_length < per_file_metadata[mapping.source_idx].source->size(),
            "Invalid stripe information");
          const auto buffer =
            per_file_metadata[mapping.source_idx].source->host_read(sf_comp_offset, sf_comp_length);
          size_t sf_length = 0;
          auto sf_data     = per_file_metadata[mapping.source_idx].decompressor->Decompress(
            buffer->data(), sf_comp_length, &sf_length);
          ProtobufReader(sf_data, sf_length)
            .read(per_file_metadata[mapping.source_idx].stripefooters[i]);
          mapping.stripe_info[i].second = &per_file_metadata[mapping.source_idx].stripefooters[i];
        }
      }
    }

    return selected_stripes_mapping;
  }

  /**
   * @brief Filters and reduces down to a selection of columns
   *
   * @param use_names List of column names to select
   * @param has_timestamp_column True if timestamp column present and false otherwise
   *
   * @return input column information, output column information, list of output column schema
   * indices
   */
  std::vector<int> select_columns(std::vector<std::string> const &use_names,
                                  bool &has_timestamp_column) const
  {
    auto const &pfm = per_file_metadata[0];

    std::vector<int> output_column_schema_idxs;
    if (not use_names.empty()) {
      int index = 0;
      for (auto const &use_name : use_names) {
        bool name_found = false;
        for (int i = 0; i < pfm.get_num_columns(); ++i, ++index) {
          if (index >= pfm.get_num_columns()) { index = 0; }
          if (pfm.get_column_name(index).compare(use_name) == 0) {
            name_found = true;
            output_column_schema_idxs.emplace_back(index);
            if (pfm.ff.types[index].kind == orc::TIMESTAMP) { has_timestamp_column = true; }
            index++;
            break;
          }
        }
        CUDF_EXPECTS(name_found, "Unknown column name : " + std::string(use_name));
      }
    } else {
      // For now, only select all leaf nodes
      for (int i = 1; i < pfm.get_num_columns(); ++i) {
        if (pfm.ff.types[i].subtypes.empty()) {
          output_column_schema_idxs.emplace_back(i);
          if (pfm.ff.types[i].kind == orc::TIMESTAMP) { has_timestamp_column = true; }
        }
      }
    }

    return output_column_schema_idxs;
  }
};

rmm::device_buffer reader::impl::decompress_stripe_data(
  hostdevice_vector<gpu::ColumnDesc> &chunks,
  const std::vector<rmm::device_buffer> &stripe_data,
  const OrcDecompressor *decompressor,
  std::vector<orc_stream_info> &stream_info,
  size_t num_stripes,
  device_span<gpu::RowGroup> row_groups,
  size_t row_index_stride,
  rmm::cuda_stream_view stream)
{
  // Parse the columns' compressed info
  hostdevice_vector<gpu::CompressedStreamInfo> compinfo(0, stream_info.size(), stream);
  for (const auto &info : stream_info) {
    compinfo.insert(gpu::CompressedStreamInfo(
      static_cast<const uint8_t *>(stripe_data[info.stripe_idx].data()) + info.dst_pos,
      info.length));
  }
  compinfo.host_to_device(stream);
  gpu::ParseCompressedStripeData(compinfo.device_ptr(),
                                 compinfo.size(),
                                 decompressor->GetBlockSize(),
                                 decompressor->GetLog2MaxCompressionRatio(),
                                 stream);
  compinfo.device_to_host(stream, true);

  // Count the exact number of compressed blocks
  size_t num_compressed_blocks   = 0;
  size_t num_uncompressed_blocks = 0;
  size_t total_decomp_size       = 0;
  for (size_t i = 0; i < compinfo.size(); ++i) {
    num_compressed_blocks += compinfo[i].num_compressed_blocks;
    num_uncompressed_blocks += compinfo[i].num_uncompressed_blocks;
    total_decomp_size += compinfo[i].max_uncompressed_size;
  }
  CUDF_EXPECTS(total_decomp_size > 0, "No decompressible data found");

  rmm::device_buffer decomp_data(total_decomp_size, stream);
  rmm::device_uvector<gpu_inflate_input_s> inflate_in(
    num_compressed_blocks + num_uncompressed_blocks, stream);
  rmm::device_uvector<gpu_inflate_status_s> inflate_out(num_compressed_blocks, stream);

  // Parse again to populate the decompression input/output buffers
  size_t decomp_offset      = 0;
  uint32_t start_pos        = 0;
  uint32_t start_pos_uncomp = (uint32_t)num_compressed_blocks;
  for (size_t i = 0; i < compinfo.size(); ++i) {
    auto dst_base                 = static_cast<uint8_t *>(decomp_data.data());
    compinfo[i].uncompressed_data = dst_base + decomp_offset;
    compinfo[i].decctl            = inflate_in.data() + start_pos;
    compinfo[i].decstatus         = inflate_out.data() + start_pos;
    compinfo[i].copyctl           = inflate_in.data() + start_pos_uncomp;

    stream_info[i].dst_pos = decomp_offset;
    decomp_offset += compinfo[i].max_uncompressed_size;
    start_pos += compinfo[i].num_compressed_blocks;
    start_pos_uncomp += compinfo[i].num_uncompressed_blocks;
  }
  compinfo.host_to_device(stream);
  gpu::ParseCompressedStripeData(compinfo.device_ptr(),
                                 compinfo.size(),
                                 decompressor->GetBlockSize(),
                                 decompressor->GetLog2MaxCompressionRatio(),
                                 stream);

  // Dispatch batches of blocks to decompress
  if (num_compressed_blocks > 0) {
    switch (decompressor->GetKind()) {
      case orc::ZLIB:
        CUDA_TRY(
          gpuinflate(inflate_in.data(), inflate_out.data(), num_compressed_blocks, 0, stream));
        break;
      case orc::SNAPPY:
        CUDA_TRY(gpu_unsnap(inflate_in.data(), inflate_out.data(), num_compressed_blocks, stream));
        break;
      default: CUDF_EXPECTS(false, "Unexpected decompression dispatch"); break;
    }
  }
  if (num_uncompressed_blocks > 0) {
    CUDA_TRY(gpu_copy_uncompressed_blocks(
      inflate_in.data() + num_compressed_blocks, num_uncompressed_blocks, stream));
  }
  gpu::PostDecompressionReassemble(compinfo.device_ptr(), compinfo.size(), stream);

  // Update the stream information with the updated uncompressed info
  // TBD: We could update the value from the information we already
  // have in stream_info[], but using the gpu results also updates
  // max_uncompressed_size to the actual uncompressed size, or zero if
  // decompression failed.
  compinfo.device_to_host(stream, true);

  const size_t num_columns = chunks.size() / num_stripes;

  for (size_t i = 0; i < num_stripes; ++i) {
    for (size_t j = 0; j < num_columns; ++j) {
      auto &chunk = chunks[i * num_columns + j];
      for (int k = 0; k < gpu::CI_NUM_STREAMS; ++k) {
        if (chunk.strm_len[k] > 0 && chunk.strm_id[k] < compinfo.size()) {
          chunk.streams[k]  = compinfo[chunk.strm_id[k]].uncompressed_data;
          chunk.strm_len[k] = compinfo[chunk.strm_id[k]].max_uncompressed_size;
        }
      }
    }
  }

  if (not row_groups.empty()) {
    chunks.host_to_device(stream);
    gpu::ParseRowGroupIndex(row_groups.data(),
                            compinfo.device_ptr(),
                            chunks.device_ptr(),
                            num_columns,
                            num_stripes,
                            row_groups.size() / num_columns,
                            row_index_stride,
                            stream);
  }

  return decomp_data;
}

void reader::impl::decode_stream_data(hostdevice_vector<gpu::ColumnDesc> &chunks,
                                      size_t num_dicts,
                                      size_t skip_rows,
                                      size_t num_rows,
                                      timezone_table_view tz_table,
                                      device_span<gpu::RowGroup const> row_groups,
                                      size_t row_index_stride,
                                      std::vector<column_buffer> &out_buffers,
                                      rmm::cuda_stream_view stream)
{
  const auto num_columns = out_buffers.size();
  const auto num_stripes = chunks.size() / out_buffers.size();

  // Update chunks with pointers to column data
  for (size_t i = 0; i < num_stripes; ++i) {
    for (size_t j = 0; j < num_columns; ++j) {
      auto &chunk            = chunks[i * num_columns + j];
      chunk.column_data_base = out_buffers[j].data();
      chunk.valid_map_base   = out_buffers[j].null_mask();
    }
  }

  // Allocate global dictionary for deserializing
  rmm::device_uvector<gpu::DictionaryEntry> global_dict(num_dicts, stream);

  chunks.host_to_device(stream);
  gpu::DecodeNullsAndStringDictionaries(
    chunks.device_ptr(), global_dict.data(), num_columns, num_stripes, num_rows, skip_rows, stream);
  gpu::DecodeOrcColumnData(chunks.device_ptr(),
                           global_dict.data(),
                           num_columns,
                           num_stripes,
                           num_rows,
                           skip_rows,
                           tz_table,
                           row_groups.data(),
                           row_groups.size() / num_columns,
                           row_index_stride,
                           stream);
  chunks.device_to_host(stream, true);

  for (size_t i = 0; i < num_stripes; ++i) {
    for (size_t j = 0; j < num_columns; ++j) {
      out_buffers[j].null_count() += chunks[i * num_columns + j].null_count;
    }
  }
}

reader::impl::impl(std::vector<std::unique_ptr<datasource>> &&sources,
                   orc_reader_options const &options,
                   rmm::mr::device_memory_resource *mr)
  : _mr(mr), _sources(std::move(sources))
{
  // Open and parse the source(s) dataset metadata
  _metadata = std::make_unique<aggregate_orc_metadata>(_sources);

  // Select only columns required by the options
  _selected_columns = _metadata->select_columns(options.get_columns(), _has_timestamp_column);

  // Override output timestamp resolution if requested
  if (options.get_timestamp_type().id() != type_id::EMPTY) {
    _timestamp_type = options.get_timestamp_type();
  }

  // Enable or disable attempt to use row index for parsing
  _use_index = options.is_enabled_use_index();

  // Enable or disable the conversion to numpy-compatible dtypes
  _use_np_dtypes = options.is_enabled_use_np_dtypes();

  // Control decimals conversion (float64 or int64 with optional scale)
  _decimal_cols_as_float = options.get_decimal_cols_as_float();
}

table_with_metadata reader::impl::read(size_type skip_rows,
                                       size_type num_rows,
                                       const std::vector<std::vector<size_type>> &stripes,
                                       rmm::cuda_stream_view stream)
{
  std::vector<std::unique_ptr<column>> out_columns;
  table_metadata out_metadata;

  // There are no columns in table
  if (_selected_columns.size() == 0) return {std::make_unique<table>(), std::move(out_metadata)};

  // Select only stripes required (aka row groups)
  const auto selected_stripes = _metadata->select_stripes(stripes, skip_rows, num_rows);

  // Association between each ORC column and its cudf::column
  std::vector<int32_t> orc_col_map(_metadata->get_num_cols(), -1);

  // Get a list of column data types
  std::vector<data_type> column_types;
  for (const auto &col : _selected_columns) {
    // If the column type is orc::DECIMAL see if the user
    // desires it to be converted to float64 or not
    auto const decimal_as_float64 = should_convert_decimal_column_to_float(
      _decimal_cols_as_float, _metadata->per_file_metadata[0], col);

    auto col_type = to_type_id(
      _metadata->get_col_type(col), _use_np_dtypes, _timestamp_type.id(), decimal_as_float64);
    CUDF_EXPECTS(col_type != type_id::EMPTY, "Unknown type");
    // Remove this once we support Decimal128 data type
    CUDF_EXPECTS((col_type != type_id::DECIMAL64) or (_metadata->get_col_type(col).precision <= 18),
                 "Decimal data has precision > 18, Decimal64 data type doesn't support it.");
    if (col_type == type_id::DECIMAL64) {
      // sign of the scale is changed since cuDF follows c++ libraries like CNL
      // which uses negative scaling, but liborc and other libraries
      // follow positive scaling.
      auto const scale = -static_cast<int32_t>(_metadata->get_col_type(col).scale.value_or(0));
      column_types.emplace_back(col_type, scale);
    } else {
      column_types.emplace_back(col_type);
    }

    // Map each ORC column to its column
    orc_col_map[col] = column_types.size() - 1;
  }

  // If no rows or stripes to read, return empty columns
  if (num_rows <= 0 || selected_stripes.empty()) {
    std::transform(column_types.cbegin(),
                   column_types.cend(),
                   std::back_inserter(out_columns),
                   [](auto const &dtype) { return make_empty_column(dtype); });
  } else {
    // Get the total number of stripes across all input files.
    size_t total_num_stripes =
      std::accumulate(selected_stripes.begin(),
                      selected_stripes.end(),
                      0,
                      [](size_t sum, auto &stripe_source_mapping) {
                        return sum + stripe_source_mapping.stripe_info.size();
                      });

    const auto num_columns = _selected_columns.size();
    const auto num_chunks  = total_num_stripes * num_columns;
    hostdevice_vector<gpu::ColumnDesc> chunks(num_chunks, stream);
    memset(chunks.host_ptr(), 0, chunks.memory_size());

    const bool use_index =
      (_use_index == true) &&
      // Only use if we don't have much work with complete columns & stripes
      // TODO: Consider nrows, gpu, and tune the threshold
      (num_rows > _metadata->get_row_index_stride() && !(_metadata->get_row_index_stride() & 7) &&
       _metadata->get_row_index_stride() > 0 && num_columns * total_num_stripes < 8 * 128) &&
      // Only use if first row is aligned to a stripe boundary
      // TODO: Fix logic to handle unaligned rows
      (skip_rows == 0);

    // Logically view streams as columns
    std::vector<orc_stream_info> stream_info;

    // Tracker for eventually deallocating compressed and uncompressed data
    std::vector<rmm::device_buffer> stripe_data;

    size_t stripe_start_row   = 0;
    size_t num_dict_entries   = 0;
    size_t num_rowgroups      = 0;
    size_t stripe_chunk_index = 0;

    for (auto &stripe_source_mapping : selected_stripes) {
      // Iterate through the source files selected stripes
      for (size_t stripe_pos_index = 0; stripe_pos_index < stripe_source_mapping.stripe_info.size();
           stripe_pos_index++) {
        auto &stripe_pair        = stripe_source_mapping.stripe_info[stripe_pos_index];
        const auto stripe_info   = stripe_pair.first;
        const auto stripe_footer = stripe_pair.second;

        auto stream_count          = stream_info.size();
        const auto total_data_size = gather_stream_info(stripe_chunk_index,
                                                        stripe_info,
                                                        stripe_footer,
                                                        orc_col_map,
                                                        _selected_columns,
                                                        _metadata->get_types(),
                                                        use_index,
                                                        &num_dict_entries,
                                                        chunks,
                                                        stream_info);

        CUDF_EXPECTS(total_data_size > 0, "Expected streams data within stripe");

        stripe_data.emplace_back(total_data_size, stream);
        auto dst_base = static_cast<uint8_t *>(stripe_data.back().data());

        // Coalesce consecutive streams into one read
        while (stream_count < stream_info.size()) {
          const auto d_dst  = dst_base + stream_info[stream_count].dst_pos;
          const auto offset = stream_info[stream_count].offset;
          auto len          = stream_info[stream_count].length;
          stream_count++;

          while (stream_count < stream_info.size() &&
                 stream_info[stream_count].offset == offset + len) {
            len += stream_info[stream_count].length;
            stream_count++;
          }
          if (_metadata->per_file_metadata[stripe_source_mapping.source_idx]
                .source->is_device_read_preferred(len)) {
            CUDF_EXPECTS(
              _metadata->per_file_metadata[stripe_source_mapping.source_idx].source->device_read(
                offset, len, d_dst, stream) == len,
              "Unexpected discrepancy in bytes read.");
          } else {
            const auto buffer =
              _metadata->per_file_metadata[stripe_source_mapping.source_idx].source->host_read(
                offset, len);
            CUDF_EXPECTS(buffer->size() == len, "Unexpected discrepancy in bytes read.");
            CUDA_TRY(
              cudaMemcpyAsync(d_dst, buffer->data(), len, cudaMemcpyHostToDevice, stream.value()));
            stream.synchronize();
          }
        }

        // Update chunks to reference streams pointers
        for (size_t col_idx = 0; col_idx < num_columns; col_idx++) {
          auto &chunk         = chunks[stripe_chunk_index * num_columns + col_idx];
          chunk.start_row     = stripe_start_row;
          chunk.num_rows      = stripe_info->numberOfRows;
          chunk.encoding_kind = stripe_footer->columns[_selected_columns[col_idx]].kind;
          chunk.type_kind     = _metadata->per_file_metadata[stripe_source_mapping.source_idx]
                              .ff.types[_selected_columns[col_idx]]
                              .kind;
          auto const decimal_as_float64 = should_convert_decimal_column_to_float(
            _decimal_cols_as_float, _metadata->per_file_metadata[0], _selected_columns[col_idx]);
          chunk.decimal_scale = _metadata->per_file_metadata[stripe_source_mapping.source_idx]
                                  .ff.types[_selected_columns[col_idx]]
                                  .scale.value_or(0) |
                                (decimal_as_float64 ? orc::gpu::orc_decimal2float64_scale : 0);
          chunk.rowgroup_id = num_rowgroups;
          chunk.dtype_len   = (column_types[col_idx].id() == type_id::STRING)
                              ? sizeof(string_index_pair)
                              : cudf::size_of(column_types[col_idx]);
          if (chunk.type_kind == orc::TIMESTAMP) {
            chunk.ts_clock_rate = to_clockrate(_timestamp_type.id());
          }
          for (int k = 0; k < gpu::CI_NUM_STREAMS; k++) {
            chunk.streams[k] = dst_base + stream_info[chunk.strm_id[k]].dst_pos;
          }
        }

        stripe_start_row += stripe_info->numberOfRows;
        if (use_index) {
          num_rowgroups += (stripe_info->numberOfRows + _metadata->get_row_index_stride() - 1) /
                           _metadata->get_row_index_stride();
        }
        stripe_chunk_index++;
      }
    }

    // Process dataset chunk pages into output columns
    if (stripe_data.size() != 0) {
      // Setup row group descriptors if using indexes
      rmm::device_uvector<gpu::RowGroup> row_groups(num_rowgroups * num_columns, stream);
      if (_metadata->per_file_metadata[0].ps.compression != orc::NONE) {
        auto decomp_data =
          decompress_stripe_data(chunks,
                                 stripe_data,
                                 _metadata->per_file_metadata[0].decompressor.get(),
                                 stream_info,
                                 total_num_stripes,
                                 row_groups,
                                 _metadata->get_row_index_stride(),
                                 stream);
        stripe_data.clear();
        stripe_data.push_back(std::move(decomp_data));
      } else {
        if (not row_groups.is_empty()) {
          chunks.host_to_device(stream);
          gpu::ParseRowGroupIndex(row_groups.data(),
                                  nullptr,
                                  chunks.device_ptr(),
                                  num_columns,
                                  total_num_stripes,
                                  num_rowgroups,
                                  _metadata->get_row_index_stride(),
                                  stream);
        }
      }

      // Setup table for converting timestamp columns from local to UTC time
      auto const tz_table = _has_timestamp_column
                              ? build_timezone_transition_table(
                                  selected_stripes[0].stripe_info[0].second->writerTimezone, stream)
                              : timezone_table{};

      std::vector<column_buffer> out_buffers;
      for (size_t i = 0; i < column_types.size(); ++i) {
        bool is_nullable = false;
        for (size_t j = 0; j < total_num_stripes; ++j) {
          if (chunks[j * num_columns + i].strm_len[gpu::CI_PRESENT] != 0) {
            is_nullable = true;
            break;
          }
        }
        out_buffers.emplace_back(column_types[i], num_rows, is_nullable, stream, _mr);
      }

      decode_stream_data(chunks,
                         num_dict_entries,
                         skip_rows,
                         num_rows,
                         tz_table.view(),
                         row_groups,
                         _metadata->get_row_index_stride(),
                         out_buffers,
                         stream);

      for (size_t i = 0; i < column_types.size(); ++i) {
        out_columns.emplace_back(make_column(out_buffers[i], nullptr, stream, _mr));
      }
    }
  }

  // Return column names (must match order of returned columns)
  out_metadata.column_names.resize(_selected_columns.size());
  for (size_t i = 0; i < _selected_columns.size(); i++) {
    out_metadata.column_names[i] = _metadata->get_column_name(0, _selected_columns[i]);
  }

  for (const auto &meta : _metadata->per_file_metadata) {
    for (const auto &kv : meta.ff.metadata) { out_metadata.user_data.insert({kv.name, kv.value}); }
  }

  return {std::make_unique<table>(std::move(out_columns)), std::move(out_metadata)};
}

// Forward to implementation
reader::reader(std::vector<std::string> const &filepaths,
               orc_reader_options const &options,
               rmm::cuda_stream_view stream,
               rmm::mr::device_memory_resource *mr)
{
  _impl = std::make_unique<impl>(datasource::create(filepaths), options, mr);
}

// Forward to implementation
reader::reader(std::vector<std::unique_ptr<cudf::io::datasource>> &&sources,
               orc_reader_options const &options,
               rmm::cuda_stream_view stream,
               rmm::mr::device_memory_resource *mr)
{
  _impl = std::make_unique<impl>(std::move(sources), options, mr);
}

// Destructor within this translation unit
reader::~reader() = default;

// Forward to implementation
table_with_metadata reader::read(orc_reader_options const &options, rmm::cuda_stream_view stream)
{
  return _impl->read(
    options.get_skip_rows(), options.get_num_rows(), options.get_stripes(), stream);
}
}  // namespace orc
}  // namespace detail
}  // namespace io
}  // namespace cudf
