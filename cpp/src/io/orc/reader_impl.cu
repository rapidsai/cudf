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

#include "reader_impl.hpp"
#include "timezone.cuh"

#include <io/comp/gpuinflate.h>
#include "orc.h"

#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <array>
#include <regex>

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
                             type_id timestamp_type_id)
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
    case orc::DECIMAL: return type_id::DECIMAL64;
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

}  // namespace

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

// /**
//  * @brief Class for parsing dataset metadata
//  */
// struct metadata {
//   explicit metadata(datasource *source)
//   {
//     constexpr auto header_len = sizeof(file_header_s);
//     constexpr auto ender_len  = sizeof(file_ender_s);

//     const auto len           = source->size();
//     const auto header_buffer = source->host_read(0, header_len);
//     const auto header        = reinterpret_cast<const file_header_s *>(header_buffer->data());
//     const auto ender_buffer  = source->host_read(len - ender_len, ender_len);
//     const auto ender         = reinterpret_cast<const file_ender_s *>(ender_buffer->data());
//     CUDF_EXPECTS(len > header_len + ender_len, "Incorrect data source");
//     CUDF_EXPECTS(header->magic == parquet_magic && ender->magic == parquet_magic,
//                  "Corrupted header or footer");
//     CUDF_EXPECTS(ender->footer_len != 0 && ender->footer_len <= (len - header_len - ender_len),
//                  "Incorrect footer length");

//     const auto buffer = source->host_read(len - ender->footer_len - ender_len,
//     ender->footer_len); CompactProtocolReader cp(buffer->data(), ender->footer_len);
//     CUDF_EXPECTS(cp.read(this), "Cannot parse metadata");
//     CUDF_EXPECTS(cp.InitSchema(this), "Cannot initialize schema");
//   }
// };

/**
 * @brief In order to support multiple input files/buffers we need to gather
 * the metadata across all of those input(s). This class provides a place
 * to aggregate that metadata from all the files.
 */
class aggregate_orc_metadata {
  using OrcStripeInfo = std::pair<const StripeInformation *, const StripeFooter *>;

  std::vector<cudf::io::orc::metadata> const per_file_metadata;
  std::map<std::string, std::string> const agg_keyval_map;
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

 public:
  aggregate_orc_metadata(std::vector<std::unique_ptr<datasource>> const &sources)
    : per_file_metadata(metadatas_from_sources(sources)),
      num_rows(calc_num_rows()),
      num_columns(calc_num_cols()),
      num_stripes(calc_num_stripes())
  {
    // Verify that the input files have matching numbers of columns
    int num_cols = -1;
    for (auto const &pfm : per_file_metadata) {
      if (num_cols == -1) { num_cols = pfm.get_num_columns(); }
      if (pfm.get_num_columns() != num_cols) {
        CUDF_EXPECTS(num_cols == static_cast<int>(pfm.get_num_columns()),
                     "All sources must have the same number of columns");
      }
    }

    // XXX: Need to talk with Vukasin about the best way to compare this schema ....
    // // Verify that the input files have matching schemas
    // for (auto const &pfm : per_file_metadata) {
    //   CUDF_EXPECTS(per_file_metadata[0].schema == pfm.schema,
    //                "All sources must have the same schemas");
    // }
  }

  auto get_col_type(int col_idx) const { return per_file_metadata[0].ff.types[col_idx]; }

  auto get_num_rows() const { return num_rows; }

  auto get_num_cols() const { return num_columns; }

  auto get_num_stripes() const { return num_stripes; }

  auto get_num_source_files() const { return per_file_metadata.size(); }

  auto get_types() const { return per_file_metadata[0].ff.types; }

  auto get_row_index_stride() const { return per_file_metadata[0].ff.rowIndexStride; }

  auto const &get_key_value_metadata() const { return agg_keyval_map; }

  auto get_post_script_for_metadata(int metadata_idx) { return per_file_metadata[metadata_idx].ps; }

  /**
   * @brief Extracts the pandas "index_columns" section
   *
   * PANDAS adds its own metadata to the key_value section when writing out the
   * dataframe to a file to aid in exact reconstruction. The JSON-formatted
   * metadata contains the index column(s) and PANDA-specific datatypes.
   *
   * @return comma-separated index column names in quotes
   */
  std::string get_pandas_index() const
  {
    auto it = agg_keyval_map.find("pandas");
    if (it != agg_keyval_map.end()) {
      // Captures a list of quoted strings found inside square brackets after `"index_columns":`
      // Inside quotes supports newlines, brackets, escaped quotes, etc.
      // One-liner regex:
      // "index_columns"\s*:\s*\[\s*((?:"(?:|(?:.*?(?![^\\]")).?)[^\\]?",?\s*)*)\]
      // Documented below.
      std::regex index_columns_expr{
        R"("index_columns"\s*:\s*\[\s*)"  // match preamble, opening square bracket, whitespace
        R"(()"                            // Open first capturing group
        R"((?:")"                         // Open non-capturing group match opening quote
        R"((?:|(?:.*?(?![^\\]")).?))"     // match empty string or anything between quotes
        R"([^\\]?")"                      // Match closing non-escaped quote
        R"(,?\s*)"                        // Match optional comma and whitespace
        R"()*)"                           // Close non-capturing group and repeat 0 or more times
        R"())"                            // Close first capturing group
        R"(\])"                           // Match closing square brackets
      };
      std::smatch sm;
      if (std::regex_search(it->second, sm, index_columns_expr)) { return sm[1].str(); }
    }
    return "";
  }

  /**
   * @brief Extracts the column name(s) used for the row indexes in a dataframe
   *
   * @param names List of column names to load, where index column name(s) will be added
   */
  void add_pandas_index_names(std::vector<std::string> &names) const
  {
    auto str = get_pandas_index();
    if (str.length() != 0) {
      std::regex index_name_expr{R"(\"((?:\\.|[^\"])*)\")"};
      std::smatch sm;
      while (std::regex_search(str, sm, index_name_expr)) {
        if (sm.size() == 2) {  // 2 = whole match, first item
          if (std::find(names.begin(), names.end(), sm[1].str()) == names.end()) {
            std::regex esc_quote{R"(\\")"};
            names.emplace_back(std::regex_replace(sm[1].str(), esc_quote, R"(")"));
          }
        }
        str = sm.suffix();
      }
    }
  }

  struct row_group_info {
    size_type const index;
    size_t const start_row;  // TODO source index
    size_type const source_index;
    row_group_info(size_type index, size_t start_row, size_type source_index)
      : index(index), start_row(start_row), source_index(source_index)
    {
    }
  };

  std::vector<OrcStripeInfo> select_stripes(const std::vector<size_type> &stripes,
                                            size_type &row_start,
                                            size_type &row_count)
  {
    auto const &pfm = per_file_metadata[0];
    std::vector<OrcStripeInfo> selection;

    if (!stripes.empty()) {
      size_t stripe_rows = 0;
      for (const auto &stripe_idx : stripes) {
        CUDF_EXPECTS(stripe_idx >= 0 && stripe_idx < pfm.get_num_stripes(), "Invalid stripe index");
        selection.emplace_back(&pfm.ff.stripes[stripe_idx], nullptr);
        stripe_rows += pfm.ff.stripes[stripe_idx].numberOfRows;
      }
      // row_start is 0 if stripes are set. If this is not true anymore, then
      // row_start needs to be subtracted to get the correct row_count
      CUDF_EXPECTS(row_start == 0, "Start row index should be 0");
      row_count = static_cast<size_type>(stripe_rows);
    } else {
      row_start = std::max(row_start, 0);
      if (row_count < 0) {
        row_count = static_cast<size_type>(
          std::min<size_t>(get_num_rows() - row_start, std::numeric_limits<size_type>::max()));
      } else {
        row_count = static_cast<size_type>(std::min<size_t>(get_num_rows() - row_start, row_count));
      }
      CUDF_EXPECTS(row_count >= 0 && row_start >= 0, "Negative row count or starting row");
      CUDF_EXPECTS(
        !(row_start > 0 && (row_count > (std::numeric_limits<size_type>::max() - row_start))),
        "Summation of starting row index and number of rows would cause overflow");

      size_type stripe_skip_rows = 0;
      for (size_t i = 0, count = 0; i < pfm.ff.stripes.size(); ++i) {
        count += pfm.ff.stripes[i].numberOfRows;
        if (count > static_cast<size_t>(row_start)) {
          if (selection.empty()) {
            stripe_skip_rows =
              static_cast<size_type>(row_start - (count - pfm.ff.stripes[i].numberOfRows));
          }
          selection.emplace_back(&pfm.ff.stripes[i], nullptr);
        }
        if (count >= static_cast<size_t>(row_start) + static_cast<size_t>(row_count)) { break; }
      }
      row_start = stripe_skip_rows;
    }

    // Read each stripe's stripefooter metadata
    if (not selection.empty()) {
      pfm.stripefooters.resize(selection.size());
      for (size_t i = 0; i < selection.size(); ++i) {
        const auto stripe         = selection[i].first;
        const auto sf_comp_offset = stripe->offset + stripe->indexLength + stripe->dataLength;
        const auto sf_comp_length = stripe->footerLength;
        CUDF_EXPECTS(sf_comp_offset + sf_comp_length < pfm.source->size(),
                     "Invalid stripe information");

        const auto buffer = pfm.source->host_read(sf_comp_offset, sf_comp_length);
        size_t sf_length  = 0;
        auto sf_data = pfm.decompressor->Decompress(buffer->data(), sf_comp_length, &sf_length);
        ProtobufReader(sf_data, sf_length).read(pfm.stripefooters[i]);
        selection[i].second = &pfm.stripefooters[i];
      }
    }

    return selection;
  }

  /**
   * @brief Filters and reduces down to a selection of columns
   *
   * @param use_names List of column names to select
   * @param has_timestamp_column Whether the ORC file contains a timestamp column
   *
   * @return vector<int> of indexes that should be used in the resulting Dataframe.
   */
  auto select_columns(std::vector<std::string> const &use_names, bool &has_timestamp_column) const
  {
    // XXX: Do we really need to use a single metadata object here? We already know all schemas
    // "match" by this point BUT are they in the same order so is returning a vector<int> going
    // to produce un-expected results?
    auto const &pfm = per_file_metadata[0];

    // Indexes of columns that should be included in resulting Dataframe
    std::vector<int> selection;

    if (not use_names.empty()) {
      int index = 0;
      for (const auto &use_name : use_names) {
        bool name_found = false;
        for (int i = 0; i < pfm.get_num_columns(); ++i, ++index) {
          if (index >= pfm.get_num_columns()) { index = 0; }
          if (pfm.get_column_name(index) == use_name) {
            name_found = true;
            selection.emplace_back(index);
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
          selection.emplace_back(i);
          if (pfm.ff.types[i].kind == orc::TIMESTAMP) { has_timestamp_column = true; }
        }
      }
    }

    return selection;
  }
};

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
}

table_with_metadata reader::impl::read(size_type skip_rows,
                                       size_type num_rows,
                                       const std::vector<size_type> &stripes,
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
    auto col_type = to_type_id(_metadata->get_col_type(col), _use_np_dtypes, _timestamp_type.id());
    CUDF_EXPECTS(col_type != type_id::EMPTY, "Unknown type");
    // Remove this once we support Decimal128 data type
    CUDF_EXPECTS((col_type != type_id::DECIMAL64) or (_metadata->get_col_type(col).precision <= 18),
                 "Decimal data has precision > 18, Decimal64 data type doesn't support it.");
    // sign of the scale is changed since cuDF follows c++ libraries like CNL
    // which uses negative scaling, but liborc and other libraries
    // follow positive scaling.
    auto scale = (col_type == type_id::DECIMAL64)
                   ? -static_cast<int32_t>(_metadata->get_col_type(col).scale)
                   : 0;
    column_types.emplace_back(col_type, scale);

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
    for (int file_idx = 0; file_idx < _metadata.get_num_source_files(); file_idx++) {
      read_individual_file(file_idx, _metadata skip_rows, num_rows, selected_stripes, stream);
    }
  }

  // Return column names (must match order of returned columns)
  out_metadata.column_names.resize(_selected_columns.size());
  for (size_t i = 0; i < _selected_columns.size(); i++) {
    out_metadata.column_names[i] = _metadata->get_column_name(_selected_columns[i]);
  }
  // Return user metadata
  for (const auto &kv : _metadata->ff.metadata) {
    out_metadata.user_data.insert({kv.name, kv.value});
  }

  return {std::make_unique<table>(std::move(out_columns)), std::move(out_metadata)};
}

void reader::impl::read_individual_file(int file_idx,
                                        std::vector<data_type> column_types,
                                        cudf::io::orc::metadata file_metadata,
                                        size_type skip_rows,
                                        size_type num_rows,
                                        std::vector<OrcStripeInfo> const &selected_stripes,
                                        rmm::cuda_stream_view stream,
                                        std::vector<int32_t> orc_col_map)
{
  const auto num_columns = _selected_columns.size();
  const auto num_chunks  = selected_stripes.size() * num_columns;
  hostdevice_vector<gpu::ColumnDesc> chunks(num_chunks, stream);
  memset(chunks.host_ptr(), 0, chunks.memory_size());

  const bool use_index =
    (_use_index == true) &&
    // Only use if we don't have much work with complete columns & stripes
    // TODO: Consider nrows, gpu, and tune the threshold
    (num_rows > _metadata->get_row_index_stride() && !(_metadata->get_row_index_stride() & 7) &&
     _metadata->get_row_index_stride() > 0 && num_columns * selected_stripes.size() < 8 * 128) &&
    // Only use if first row is aligned to a stripe boundary
    // TODO: Fix logic to handle unaligned rows
    (skip_rows == 0);

  // Logically view streams as columns
  std::vector<orc_stream_info> stream_info;

  // Tracker for eventually deallocating compressed and uncompressed data
  std::vector<rmm::device_buffer> stripe_data;

  size_t stripe_start_row = 0;
  size_t num_dict_entries = 0;
  size_t num_rowgroups    = 0;

  for (size_t i = 0; i < selected_stripes.size(); ++i) {
    const auto stripe_info   = selected_stripes[i].first;
    const auto stripe_footer = selected_stripes[i].second;

    auto stream_count          = stream_info.size();
    const auto total_data_size = gather_stream_info(i,
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
      const auto buffer = _sources[file_idx]->host_read(offset, len);
      CUDA_TRY(cudaMemcpyAsync(d_dst, buffer->data(), len, cudaMemcpyHostToDevice, stream.value()));
      stream.synchronize();
    }

    // Update chunks to reference streams pointers
    for (size_t j = 0; j < num_columns; j++) {
      auto &chunk         = chunks[i * num_columns + j];
      chunk.start_row     = stripe_start_row;
      chunk.num_rows      = stripe_info->numberOfRows;
      chunk.encoding_kind = stripe_footer->columns[_selected_columns[j]].kind;
      chunk.type_kind     = _metadata->get_types()[_selected_columns[j]].kind;
      chunk.decimal_scale = _metadata->get_types()[_selected_columns[j]].scale;
      chunk.rowgroup_id   = num_rowgroups;
      chunk.dtype_len     = (column_types[j].id() == type_id::STRING)
                          ? sizeof(std::pair<const char *, size_t>)
                          : cudf::size_of(column_types[j]);
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
  }

  // Process dataset chunk pages into output columns
  if (stripe_data.size() != 0) {
    // Setup row group descriptors if using indexes
    rmm::device_uvector<gpu::RowGroup> row_groups(num_rowgroups * num_columns, stream);
    if (file_metadata.ps.compression != orc::NONE) {
      auto decomp_data = decompress_stripe_data(chunks,
                                                stripe_data,
                                                file_metadata.decompressor.get(),
                                                stream_info,
                                                selected_stripes.size(),
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
                                selected_stripes.size(),
                                num_rowgroups,
                                _metadata->get_row_index_stride(),
                                stream);
      }
    }

    // Setup table for converting timestamp columns from local to UTC time
    auto const tz_table =
      _has_timestamp_column
        ? build_timezone_transition_table(selected_stripes[0].second->writerTimezone, stream)
        : timezone_table{};

    std::vector<column_buffer> out_buffers;
    for (size_t i = 0; i < column_types.size(); ++i) {
      bool is_nullable = false;
      for (size_t j = 0; j < selected_stripes.size(); ++j) {
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

// Forward to implementation
reader::reader(std::vector<std::string> const &filepaths,
               orc_reader_options const &options,
               rmm::mr::device_memory_resource *mr)
{
  _impl = std::make_unique<impl>(datasource::create(filepaths), options, mr);
}

// Forward to implementation
reader::reader(std::vector<std::unique_ptr<cudf::io::datasource>> &&sources,
               orc_reader_options const &options,
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
