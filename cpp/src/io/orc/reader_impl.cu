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

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <iterator>
#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

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
constexpr type_id to_type_id(const orc::SchemaType& schema,
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
    case orc::LIST: return type_id::LIST;
    case orc::STRUCT: return type_id::STRUCT;
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
 * @brief struct to store buffer data and size of list buffer
 */
struct list_buffer_data {
  size_type* data;
  size_type size;
};

// Generates offsets for list buffer from number of elements in a row.
void generate_offsets_for_list(rmm::device_uvector<list_buffer_data> const& buff_data,
                               rmm::cuda_stream_view stream)
{
  auto transformer = [] __device__(list_buffer_data list_data) {
    thrust::exclusive_scan(
      thrust::seq, list_data.data, list_data.data + list_data.size, list_data.data);
  };
  thrust::for_each(rmm::exec_policy(stream), buff_data.begin(), buff_data.end(), transformer);
  stream.synchronize();
}

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
                          const orc::StripeInformation* stripeinfo,
                          const orc::StripeFooter* stripefooter,
                          const std::vector<int>& orc2gdf,
                          const std::vector<orc_column_meta>& gdf2orc,
                          const std::vector<orc::SchemaType> types,
                          bool use_index,
                          size_t* num_dictionary_entries,
                          cudf::detail::hostdevice_2dvector<gpu::ColumnDesc>& chunks,
                          std::vector<orc_stream_info>& stream_info)
{
  uint64_t src_offset = 0;
  uint64_t dst_offset = 0;
  for (const auto& stream : stripefooter->streams) {
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
          for (const auto& idx : schema_type.subtypes) {
            auto child_idx = (idx < orc2gdf.size()) ? orc2gdf[idx] : -1;
            if (child_idx >= 0) {
              col                             = child_idx;
              auto& chunk                     = chunks[stripe_index][col];
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
        auto& chunk = chunks[stripe_index][col];
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
bool should_convert_decimal_column_to_float(const std::vector<std::string>& columns_to_convert,
                                            cudf::io::orc::metadata& metadata,
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
  using OrcStripeInfo = std::pair<const StripeInformation*, const StripeFooter*>;

 public:
  mutable std::vector<cudf::io::orc::metadata> per_file_metadata;
  size_type const num_rows;
  size_type const num_columns;
  size_type const num_stripes;

  /**
   * @brief Create a metadata object from each element in the source vector
   */
  auto metadatas_from_sources(std::vector<std::unique_ptr<datasource>> const& sources)
  {
    std::vector<cudf::io::orc::metadata> metadatas;
    std::transform(
      sources.cbegin(), sources.cend(), std::back_inserter(metadatas), [](auto const& source) {
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
      per_file_metadata.begin(), per_file_metadata.end(), 0, [](auto& sum, auto& pfm) {
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
      per_file_metadata.begin(), per_file_metadata.end(), 0, [](auto& sum, auto& pfm) {
        return sum + pfm.get_num_stripes();
      });
  }

  aggregate_orc_metadata(std::vector<std::unique_ptr<datasource>> const& sources)
    : per_file_metadata(metadatas_from_sources(sources)),
      num_rows(calc_num_rows()),
      num_columns(calc_num_cols()),
      num_stripes(calc_num_stripes())
  {
    // Verify that the input files have the same number of columns,
    // as well as matching types, compression, and names
    for (auto const& pfm : per_file_metadata) {
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

  auto const& get_schema(int schema_idx) const { return per_file_metadata[0].ff.types[schema_idx]; }

  auto get_col_type(int col_idx) const { return per_file_metadata[0].ff.types[col_idx]; }

  auto get_num_rows() const { return num_rows; }

  auto get_num_cols() const { return per_file_metadata[0].get_num_columns(); }

  auto get_num_stripes() const { return num_stripes; }

  auto get_num_source_files() const { return per_file_metadata.size(); }

  auto const& get_types() const { return per_file_metadata[0].ff.types; }

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
    std::vector<std::vector<size_type>> const& user_specified_stripes,
    size_type& row_start,
    size_type& row_count)
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
        for (const size_t& stripe_idx : user_specified_stripes[src_file_idx]) {
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
      for (auto& mapping : selected_stripes_mapping) {
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
   * @brief Adds column as per the request and saves metadata about children.
   *        Struct children are in the same level as struct, only list column
   *        children are pushed to next level.
   *
   * @param selection A vector that saves list of columns as per levels of nesting.
   * @param types A vector of schema types of columns.
   * @param level current level of nesting.
   * @param id current column id that needs to be added.
   * @param has_timestamp_column True if timestamp column present and false otherwise.
   *
   * @return returns number of child columns at same level in case of struct and next level in case
   * of list
   */
  uint32_t add_column(std::vector<std::vector<orc_column_meta>>& selection,
                      std::vector<SchemaType> const& types,
                      const size_t level,
                      const uint32_t id,
                      bool& has_timestamp_column,
                      bool& has_list_column)
  {
    uint32_t num_lvl_child_columns = 0;
    if (level == selection.size()) { selection.emplace_back(); }
    selection[level].push_back({id, 0});
    const int col_id = selection[level].size() - 1;
    if (types[id].kind == orc::TIMESTAMP) { has_timestamp_column = true; }

    switch (types[id].kind) {
      case orc::LIST: {
        uint32_t lvl_cols = 0;
        if (not types[id].subtypes.empty()) {
          has_list_column = true;
          // Since list column needs to be processed before its child can be processed,
          // child column is being added to next level
          lvl_cols =
            add_column(selection, types, level + 1, id + 1, has_timestamp_column, has_list_column);
        }
        // The list child column may be a struct in which case lvl_cols will be > 1
        selection[level][col_id].num_children = lvl_cols;
      } break;

      case orc::STRUCT:
        for (const auto child_id : types[id].subtypes) {
          num_lvl_child_columns +=
            add_column(selection, types, level, child_id, has_timestamp_column, has_list_column);
        }
        selection[level][col_id].num_children = num_lvl_child_columns;
        break;

      default: break;
    }

    return num_lvl_child_columns + 1;
  }

  /**
   * @brief Filters and reduces down to a selection of columns
   *
   * @param use_names List of column names to select
   * @param has_timestamp_column True if timestamp column present and false otherwise
   *
   * @return Vector of list of ORC column meta-data
   */
  std::vector<std::vector<orc_column_meta>> select_columns(
    std::vector<std::string> const& use_names, bool& has_timestamp_column, bool& has_list_column)
  {
    auto const& pfm = per_file_metadata[0];
    std::vector<std::vector<orc_column_meta>> selection;

    if (not use_names.empty()) {
      uint32_t index = 0;
      // Have to check only parent columns
      auto const num_columns = pfm.ff.types[0].subtypes.size();

      for (const auto& use_name : use_names) {
        bool name_found = false;
        for (uint32_t i = 0; i < num_columns; ++i, ++index) {
          if (index >= num_columns) { index = 0; }
          auto col_id = pfm.ff.types[0].subtypes[index];
          if (pfm.get_column_name(col_id) == use_name) {
            name_found = true;
            add_column(selection, pfm.ff.types, 0, col_id, has_timestamp_column, has_list_column);
            // Should start with next index
            index = i + 1;
            break;
          }
        }
        CUDF_EXPECTS(name_found, "Unknown column name : " + std::string(use_name));
      }
    } else {
      for (auto const& col_id : pfm.ff.types[0].subtypes) {
        add_column(selection, pfm.ff.types, 0, col_id, has_timestamp_column, has_list_column);
      }
    }

    return selection;
  }
};

rmm::device_buffer reader::impl::decompress_stripe_data(
  cudf::detail::hostdevice_2dvector<gpu::ColumnDesc>& chunks,
  const std::vector<rmm::device_buffer>& stripe_data,
  const OrcDecompressor* decompressor,
  std::vector<orc_stream_info>& stream_info,
  size_t num_stripes,
  cudf::detail::hostdevice_2dvector<gpu::RowGroup>& row_groups,
  size_t row_index_stride,
  bool use_base_stride,
  rmm::cuda_stream_view stream)
{
  // Parse the columns' compressed info
  hostdevice_vector<gpu::CompressedStreamInfo> compinfo(0, stream_info.size(), stream);
  for (const auto& info : stream_info) {
    compinfo.insert(gpu::CompressedStreamInfo(
      static_cast<const uint8_t*>(stripe_data[info.stripe_idx].data()) + info.dst_pos,
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
    auto dst_base                 = static_cast<uint8_t*>(decomp_data.data());
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

  const size_t num_columns = chunks.size().second;

  for (size_t i = 0; i < num_stripes; ++i) {
    for (size_t j = 0; j < num_columns; ++j) {
      auto& chunk = chunks[i][j];
      for (int k = 0; k < gpu::CI_NUM_STREAMS; ++k) {
        if (chunk.strm_len[k] > 0 && chunk.strm_id[k] < compinfo.size()) {
          chunk.streams[k]  = compinfo[chunk.strm_id[k]].uncompressed_data;
          chunk.strm_len[k] = compinfo[chunk.strm_id[k]].max_uncompressed_size;
        }
      }
    }
  }

  if (row_groups.size().first) {
    chunks.host_to_device(stream);
    row_groups.host_to_device(stream);
    gpu::ParseRowGroupIndex(row_groups.base_device_ptr(),
                            compinfo.device_ptr(),
                            chunks.base_device_ptr(),
                            num_columns,
                            num_stripes,
                            row_groups.size().first,
                            row_index_stride,
                            use_base_stride,
                            stream);
  }

  return decomp_data;
}

void reader::impl::decode_stream_data(cudf::detail::hostdevice_2dvector<gpu::ColumnDesc>& chunks,
                                      size_t num_dicts,
                                      size_t skip_rows,
                                      timezone_table_view tz_table,
                                      cudf::detail::hostdevice_2dvector<gpu::RowGroup>& row_groups,
                                      size_t row_index_stride,
                                      std::vector<column_buffer>& out_buffers,
                                      size_t level,
                                      rmm::cuda_stream_view stream)
{
  const auto num_stripes = chunks.size().first;
  const auto num_columns = chunks.size().second;

  // Update chunks with pointers to column data
  for (size_t i = 0; i < num_stripes; ++i) {
    for (size_t j = 0; j < num_columns; ++j) {
      auto& chunk            = chunks[i][j];
      chunk.column_data_base = out_buffers[j].data();
      chunk.valid_map_base   = out_buffers[j].null_mask();
    }
  }

  // Allocate global dictionary for deserializing
  rmm::device_uvector<gpu::DictionaryEntry> global_dict(num_dicts, stream);

  chunks.host_to_device(stream);
  gpu::DecodeNullsAndStringDictionaries(
    chunks.base_device_ptr(), global_dict.data(), num_columns, num_stripes, skip_rows, stream);
  gpu::DecodeOrcColumnData(chunks.base_device_ptr(),
                           global_dict.data(),
                           row_groups,
                           num_columns,
                           num_stripes,
                           skip_rows,
                           tz_table,
                           row_groups.size().first,
                           row_index_stride,
                           level,
                           stream);
  chunks.device_to_host(stream, true);

  for (size_t i = 0; i < num_stripes; ++i) {
    for (size_t j = 0; j < num_columns; ++j) {
      out_buffers[j].null_count() += chunks[i][j].null_count;
    }
  }
}

// Aggregate child column metadata per stripe and per column
void reader::impl::aggregate_child_meta(cudf::detail::host_2dspan<gpu::ColumnDesc> chunks,
                                        cudf::detail::host_2dspan<gpu::RowGroup> row_groups,
                                        std::vector<orc_column_meta> const& list_col,
                                        const int32_t level)
{
  const auto num_of_stripes         = chunks.size().first;
  const auto num_of_rowgroups       = row_groups.size().first;
  const auto num_parent_cols        = _selected_columns[level].size();
  const auto num_child_cols         = _selected_columns[level + 1].size();
  const auto number_of_child_chunks = num_child_cols * num_of_stripes;
  auto& num_child_rows              = _col_meta.num_child_rows;

  // Reset the meta to store child column details.
  num_child_rows.resize(_selected_columns[level + 1].size());
  std::fill(num_child_rows.begin(), num_child_rows.end(), 0);
  _col_meta.child_start_row.resize(number_of_child_chunks);
  _col_meta.num_child_rows_per_stripe.resize(number_of_child_chunks);
  _col_meta.rwgrp_meta.resize(num_of_rowgroups * num_child_cols);

  auto child_start_row = cudf::detail::host_2dspan<uint32_t>(
    _col_meta.child_start_row.data(), num_of_stripes, num_child_cols);
  auto num_child_rows_per_stripe = cudf::detail::host_2dspan<uint32_t>(
    _col_meta.num_child_rows_per_stripe.data(), num_of_stripes, num_child_cols);
  auto rwgrp_meta = cudf::detail::host_2dspan<reader_column_meta::row_group_meta>(
    _col_meta.rwgrp_meta.data(), num_of_rowgroups, num_child_cols);

  int index = 0;  // number of child column processed

  // For each parent column, update its child column meta for each stripe.
  std::for_each(list_col.cbegin(), list_col.cend(), [&](const auto p_col) {
    const auto parent_col_idx = _col_meta.orc_col_map[level][p_col.id];
    auto start_row            = 0;
    auto processed_row_groups = 0;

    for (size_t stripe_id = 0; stripe_id < num_of_stripes; stripe_id++) {
      // Aggregate num_rows and start_row from processed parent columns per row groups
      if (num_of_rowgroups) {
        auto stripe_num_row_groups = chunks[stripe_id][parent_col_idx].num_rowgroups;
        auto processed_child_rows  = 0;

        for (size_t rowgroup_id = 0; rowgroup_id < stripe_num_row_groups;
             rowgroup_id++, processed_row_groups++) {
          const auto child_rows = row_groups[processed_row_groups][parent_col_idx].num_child_rows;
          for (uint32_t id = 0; id < p_col.num_children; id++) {
            const auto child_col_idx                                  = index + id;
            rwgrp_meta[processed_row_groups][child_col_idx].start_row = processed_child_rows;
            rwgrp_meta[processed_row_groups][child_col_idx].num_rows  = child_rows;
          }
          processed_child_rows += child_rows;
        }
      }

      // Aggregate start row, number of rows per chunk and total number of rows in a column
      const auto child_rows = chunks[stripe_id][parent_col_idx].num_child_rows;
      for (uint32_t id = 0; id < p_col.num_children; id++) {
        const auto child_col_idx = index + id;

        num_child_rows[child_col_idx] += child_rows;
        num_child_rows_per_stripe[stripe_id][child_col_idx] = child_rows;
        // start row could be different for each column when there is nesting at each stripe level
        child_start_row[stripe_id][child_col_idx] = (stripe_id == 0) ? 0 : start_row;
      }
      start_row += child_rows;
    }
    index += p_col.num_children;
  });
}

std::unique_ptr<column> reader::impl::create_empty_column(const int32_t orc_col_id,
                                                          column_name_info& schema_info,
                                                          rmm::cuda_stream_view stream)
{
  schema_info.name = _metadata->get_column_name(0, orc_col_id);
  // If the column type is orc::DECIMAL see if the user
  // desires it to be converted to float64 or not
  auto const decimal_as_float64 = should_convert_decimal_column_to_float(
    _decimal_cols_as_float, _metadata->per_file_metadata[0], orc_col_id);
  auto const type = to_type_id(
    _metadata->get_schema(orc_col_id), _use_np_dtypes, _timestamp_type.id(), decimal_as_float64);
  int32_t scale = 0;
  std::vector<std::unique_ptr<column>> child_columns;
  std::unique_ptr<column> out_col = nullptr;

  switch (type) {
    case type_id::LIST:
      schema_info.children.emplace_back("offsets");
      schema_info.children.emplace_back("");
      out_col = make_lists_column(
        0,
        make_empty_column(data_type(type_id::INT32)),
        create_empty_column(
          _metadata->get_col_type(orc_col_id).subtypes[0], schema_info.children.back(), stream),
        0,
        rmm::device_buffer{0, stream},
        stream);

      break;

    case type_id::STRUCT:
      for (const auto col : _metadata->get_col_type(orc_col_id).subtypes) {
        schema_info.children.emplace_back("");
        child_columns.push_back(create_empty_column(col, schema_info.children.back(), stream));
      }
      out_col =
        make_structs_column(0, std::move(child_columns), 0, rmm::device_buffer{0, stream}, stream);
      break;

    case type_id::DECIMAL64:
      scale = -static_cast<int32_t>(_metadata->get_types()[orc_col_id].scale.value_or(0));
    default: out_col = make_empty_column(data_type(type, scale));
  }

  return out_col;
}

// Adds child column buffers to parent column
column_buffer&& reader::impl::assemble_buffer(const int32_t orc_col_id,
                                              std::vector<std::vector<column_buffer>>& col_buffers,
                                              const size_t level)
{
  auto const col_id = _col_meta.orc_col_map[level][orc_col_id];
  auto& col_buffer  = col_buffers[level][col_id];

  col_buffer.name = _metadata->get_column_name(0, orc_col_id);
  switch (col_buffer.type.id()) {
    case type_id::LIST:
      col_buffer.children.emplace_back(
        assemble_buffer(_metadata->get_col_type(orc_col_id).subtypes[0], col_buffers, level + 1));
      break;

    case type_id::STRUCT:
      for (auto const& col : _metadata->get_col_type(orc_col_id).subtypes) {
        col_buffer.children.emplace_back(assemble_buffer(col, col_buffers, level));
      }

      break;

    default: break;
  }

  return std::move(col_buffer);
}

// creates columns along with schema information for each column
void reader::impl::create_columns(std::vector<std::vector<column_buffer>>&& col_buffers,
                                  std::vector<std::unique_ptr<column>>& out_columns,
                                  std::vector<column_name_info>& schema_info,
                                  rmm::cuda_stream_view stream)
{
  for (size_t i = 0; i < _selected_columns[0].size();) {
    auto const& col_meta = _selected_columns[0][i];
    schema_info.emplace_back("");

    auto col_buffer = assemble_buffer(col_meta.id, col_buffers, 0);
    out_columns.emplace_back(make_column(col_buffer, &schema_info.back(), stream, _mr));

    // Need to skip child columns of struct which are at the same level and have been processed
    i += (col_buffers[0][i].type.id() == type_id::STRUCT) ? col_meta.num_children + 1 : 1;
  }
}

reader::impl::impl(std::vector<std::unique_ptr<datasource>>&& sources,
                   orc_reader_options const& options,
                   rmm::mr::device_memory_resource* mr)
  : _mr(mr), _sources(std::move(sources))
{
  // Open and parse the source(s) dataset metadata
  _metadata = std::make_unique<aggregate_orc_metadata>(_sources);

  // Select only columns required by the options
  _selected_columns =
    _metadata->select_columns(options.get_columns(), _has_timestamp_column, _has_list_column);

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
                                       const std::vector<std::vector<size_type>>& stripes,
                                       rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(skip_rows == 0 or (not _has_list_column),
               "skip_rows is not supported by list column");

  std::vector<std::unique_ptr<column>> out_columns;
  // buffer and stripe data are stored as per nesting level
  std::vector<std::vector<column_buffer>> out_buffers(_selected_columns.size());
  std::vector<column_name_info> schema_info;
  std::vector<std::vector<rmm::device_buffer>> lvl_stripe_data(_selected_columns.size());
  table_metadata out_metadata;

  // There are no columns in the table
  if (_selected_columns.size() == 0) return {std::make_unique<table>(), std::move(out_metadata)};

  // Select only stripes required (aka row groups)
  const auto selected_stripes = _metadata->select_stripes(stripes, skip_rows, num_rows);

  // Iterates through levels of nested columns, struct columns and its children will be
  // in the same level since child column also have same number of rows,
  // list column children will be 1 level down compared to parent.
  for (size_t level = 0; level < _selected_columns.size(); level++) {
    auto& selected_columns = _selected_columns[level];
    // Association between each ORC column and its cudf::column
    _col_meta.orc_col_map.emplace_back(_metadata->get_num_cols(), -1);
    std::vector<orc_column_meta> list_col;

    // Get a list of column data types
    std::vector<data_type> column_types;
    for (auto& col : selected_columns) {
      // If the column type is orc::DECIMAL see if the user
      // desires it to be converted to float64 or not
      auto const decimal_as_float64 = should_convert_decimal_column_to_float(
        _decimal_cols_as_float, _metadata->per_file_metadata[0], col.id);
      auto col_type = to_type_id(
        _metadata->get_col_type(col.id), _use_np_dtypes, _timestamp_type.id(), decimal_as_float64);
      CUDF_EXPECTS(col_type != type_id::EMPTY, "Unknown type");
      // Remove this once we support Decimal128 data type
      CUDF_EXPECTS(
        (col_type != type_id::DECIMAL64) or (_metadata->get_col_type(col.id).precision <= 18),
        "Decimal data has precision > 18, Decimal64 data type doesn't support it.");
      if (col_type == type_id::DECIMAL64) {
        // sign of the scale is changed since cuDF follows c++ libraries like CNL
        // which uses negative scaling, but liborc and other libraries
        // follow positive scaling.
        auto const scale = -static_cast<int32_t>(_metadata->get_col_type(col.id).scale.value_or(0));
        column_types.emplace_back(col_type, scale);
      } else {
        column_types.emplace_back(col_type);
      }

      // Map each ORC column to its column
      _col_meta.orc_col_map[level][col.id] = column_types.size() - 1;
      if (col_type == type_id::LIST) list_col.emplace_back(col);
    }

    // If no rows or stripes to read, return empty columns
    if (num_rows <= 0 || selected_stripes.empty()) {
      for (size_t i = 0; i < _selected_columns[0].size();) {
        auto const& col_meta = _selected_columns[0][i];
        auto const schema    = _metadata->get_schema(col_meta.id);
        schema_info.emplace_back("");
        out_columns.push_back(
          std::move(create_empty_column(col_meta.id, schema_info.back(), stream)));
        // Since struct children will be in the same level, have to skip them.
        i += (schema.kind == orc::STRUCT) ? col_meta.num_children + 1 : 1;
      }
      break;
    } else {
      // Get the total number of stripes across all input files.
      size_t total_num_stripes =
        std::accumulate(selected_stripes.begin(),
                        selected_stripes.end(),
                        0,
                        [](size_t sum, auto& stripe_source_mapping) {
                          return sum + stripe_source_mapping.stripe_info.size();
                        });
      const auto num_columns = selected_columns.size();
      cudf::detail::hostdevice_2dvector<gpu::ColumnDesc> chunks(
        total_num_stripes, num_columns, stream);
      memset(chunks.base_host_ptr(), 0, chunks.memory_size());

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
      auto& stripe_data = lvl_stripe_data[level];

      size_t stripe_start_row = 0;
      size_t num_dict_entries = 0;
      size_t num_rowgroups    = 0;
      int stripe_idx          = 0;

      for (auto const& stripe_source_mapping : selected_stripes) {
        // Iterate through the source files selected stripes
        for (auto const& stripe : stripe_source_mapping.stripe_info) {
          const auto stripe_info   = stripe.first;
          const auto stripe_footer = stripe.second;

          auto stream_count          = stream_info.size();
          const auto total_data_size = gather_stream_info(stripe_idx,
                                                          stripe_info,
                                                          stripe_footer,
                                                          _col_meta.orc_col_map[level],
                                                          selected_columns,
                                                          _metadata->get_types(),
                                                          use_index,
                                                          &num_dict_entries,
                                                          chunks,
                                                          stream_info);

          CUDF_EXPECTS(total_data_size > 0, "Expected streams data within stripe");

          stripe_data.emplace_back(total_data_size, stream);
          auto dst_base = static_cast<uint8_t*>(stripe_data.back().data());

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
              CUDA_TRY(cudaMemcpyAsync(
                d_dst, buffer->data(), len, cudaMemcpyHostToDevice, stream.value()));
              stream.synchronize();
            }
          }

          const auto num_rows_per_stripe = stripe_info->numberOfRows;
          const auto rowgroup_id         = num_rowgroups;
          auto stripe_num_rowgroups      = 0;
          if (use_index) {
            stripe_num_rowgroups = (num_rows_per_stripe + _metadata->get_row_index_stride() - 1) /
                                   _metadata->get_row_index_stride();
          }
          // Update chunks to reference streams pointers
          for (size_t col_idx = 0; col_idx < num_columns; col_idx++) {
            auto& chunk = chunks[stripe_idx][col_idx];
            // start row, number of rows in a each stripe and total number of rows
            // may change in lower levels of nesting
            chunk.start_row = (level == 0)
                                ? stripe_start_row
                                : _col_meta.child_start_row[stripe_idx * num_columns + col_idx];
            chunk.num_rows =
              (level == 0)
                ? stripe_info->numberOfRows
                : _col_meta.num_child_rows_per_stripe[stripe_idx * num_columns + col_idx];
            chunk.column_num_rows = (level == 0) ? num_rows : _col_meta.num_child_rows[col_idx];
            chunk.encoding_kind   = stripe_footer->columns[selected_columns[col_idx].id].kind;
            chunk.type_kind       = _metadata->per_file_metadata[stripe_source_mapping.source_idx]
                                .ff.types[selected_columns[col_idx].id]
                                .kind;
            auto const decimal_as_float64 =
              should_convert_decimal_column_to_float(_decimal_cols_as_float,
                                                     _metadata->per_file_metadata[0],
                                                     selected_columns[col_idx].id);
            chunk.decimal_scale = _metadata->per_file_metadata[stripe_source_mapping.source_idx]
                                    .ff.types[selected_columns[col_idx].id]
                                    .scale.value_or(0) |
                                  (decimal_as_float64 ? orc::gpu::orc_decimal2float64_scale : 0);

            chunk.rowgroup_id   = rowgroup_id;
            chunk.dtype_len     = (column_types[col_idx].id() == type_id::STRING)
                                    ? sizeof(string_index_pair)
                                  : ((column_types[col_idx].id() == type_id::LIST) or
                                 (column_types[col_idx].id() == type_id::STRUCT))
                                    ? sizeof(int32_t)
                                    : cudf::size_of(column_types[col_idx]);
            chunk.num_rowgroups = stripe_num_rowgroups;
            if (chunk.type_kind == orc::TIMESTAMP) {
              chunk.ts_clock_rate = to_clockrate(_timestamp_type.id());
            }
            for (int k = 0; k < gpu::CI_NUM_STREAMS; k++) {
              chunk.streams[k] = dst_base + stream_info[chunk.strm_id[k]].dst_pos;
            }
          }
          stripe_start_row += num_rows_per_stripe;
          num_rowgroups += stripe_num_rowgroups;

          stripe_idx++;
        }
      }

      // Process dataset chunk pages into output columns
      if (stripe_data.size() != 0) {
        auto row_groups =
          cudf::detail::hostdevice_2dvector<gpu::RowGroup>(num_rowgroups, num_columns, stream);
        if (level > 0 and row_groups.size().first) {
          cudf::host_span<gpu::RowGroup> row_groups_span(row_groups.base_host_ptr(),
                                                         num_rowgroups * num_columns);
          auto& rw_grp_meta = _col_meta.rwgrp_meta;

          // Update start row and num rows per row group
          std::transform(rw_grp_meta.begin(),
                         rw_grp_meta.end(),
                         row_groups_span.begin(),
                         rw_grp_meta.begin(),
                         [&](auto meta, auto& row_grp) {
                           row_grp.num_rows  = meta.num_rows;
                           row_grp.start_row = meta.start_row;
                           return meta;
                         });
        }
        // Setup row group descriptors if using indexes
        if (_metadata->per_file_metadata[0].ps.compression != orc::NONE) {
          auto decomp_data =
            decompress_stripe_data(chunks,
                                   stripe_data,
                                   _metadata->per_file_metadata[0].decompressor.get(),
                                   stream_info,
                                   total_num_stripes,
                                   row_groups,
                                   _metadata->get_row_index_stride(),
                                   level == 0,
                                   stream);
          stripe_data.clear();
          stripe_data.push_back(std::move(decomp_data));
        } else {
          if (row_groups.size().first) {
            chunks.host_to_device(stream);
            row_groups.host_to_device(stream);
            gpu::ParseRowGroupIndex(row_groups.base_device_ptr(),
                                    nullptr,
                                    chunks.base_device_ptr(),
                                    num_columns,
                                    total_num_stripes,
                                    num_rowgroups,
                                    _metadata->get_row_index_stride(),
                                    level == 0,
                                    stream);
          }
        }

        // Setup table for converting timestamp columns from local to UTC time
        auto const tz_table =
          _has_timestamp_column
            ? build_timezone_transition_table(
                selected_stripes[0].stripe_info[0].second->writerTimezone, stream)
            : timezone_table{};

        for (size_t i = 0; i < column_types.size(); ++i) {
          bool is_nullable = false;
          for (size_t j = 0; j < total_num_stripes; ++j) {
            if (chunks[j][i].strm_len[gpu::CI_PRESENT] != 0) {
              is_nullable = true;
              break;
            }
          }
          auto is_list_type = (column_types[i].id() == type_id::LIST);
          auto n_rows       = (level == 0) ? num_rows : _col_meta.num_child_rows[i];
          // For list column, offset column will be always size + 1
          if (is_list_type) n_rows++;
          out_buffers[level].emplace_back(column_types[i], n_rows, is_nullable, stream, _mr);
        }

        decode_stream_data(chunks,
                           num_dict_entries,
                           skip_rows,
                           tz_table.view(),
                           row_groups,
                           _metadata->get_row_index_stride(),
                           out_buffers[level],
                           level,
                           stream);

        // Extract information to process list child columns
        if (list_col.size()) {
          row_groups.device_to_host(stream, true);
          aggregate_child_meta(chunks, row_groups, list_col, level);
        }

        // ORC stores number of elements at each row, so we need to generate offsets from that
        if (list_col.size()) {
          std::vector<list_buffer_data> buff_data;
          std::for_each(
            out_buffers[level].begin(), out_buffers[level].end(), [&buff_data](auto& out_buffer) {
              if (out_buffer.type.id() == type_id::LIST) {
                auto data = static_cast<size_type*>(out_buffer.data());
                buff_data.emplace_back(list_buffer_data{data, out_buffer.size});
              }
            });

          auto const dev_buff_data = cudf::detail::make_device_uvector_async(buff_data, stream);
          generate_offsets_for_list(dev_buff_data, stream);
        }
      }
    }
  }

  // If out_columns is empty, then create columns from buffer.
  if (out_columns.empty()) {
    create_columns(std::move(out_buffers), out_columns, schema_info, stream);
  }

  // Return column names (must match order of returned columns)
  out_metadata.column_names.reserve(schema_info.size());
  std::transform(schema_info.cbegin(),
                 schema_info.cend(),
                 std::back_inserter(out_metadata.column_names),
                 [](auto info) { return info.name; });

  out_metadata.schema_info = std::move(schema_info);

  for (const auto& meta : _metadata->per_file_metadata) {
    for (const auto& kv : meta.ff.metadata) {
      out_metadata.user_data.insert({kv.name, kv.value});
    }
  }

  return {std::make_unique<table>(std::move(out_columns)), std::move(out_metadata)};
}

// Forward to implementation
reader::reader(std::vector<std::string> const& filepaths,
               orc_reader_options const& options,
               rmm::cuda_stream_view stream,
               rmm::mr::device_memory_resource* mr)
{
  _impl = std::make_unique<impl>(datasource::create(filepaths), options, mr);
}

// Forward to implementation
reader::reader(std::vector<std::unique_ptr<cudf::io::datasource>>&& sources,
               orc_reader_options const& options,
               rmm::cuda_stream_view stream,
               rmm::mr::device_memory_resource* mr)
{
  _impl = std::make_unique<impl>(std::move(sources), options, mr);
}

// Destructor within this translation unit
reader::~reader() = default;

// Forward to implementation
table_with_metadata reader::read(orc_reader_options const& options, rmm::cuda_stream_view stream)
{
  return _impl->read(
    options.get_skip_rows(), options.get_num_rows(), options.get_stripes(), stream);
}
}  // namespace orc
}  // namespace detail
}  // namespace io
}  // namespace cudf
