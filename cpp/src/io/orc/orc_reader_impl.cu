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

#include "orc_reader_impl.hpp"
#include "timezone.h"

#include <cudf/cudf.h>
#include <io/comp/gpuinflate.h>

#include <cuda_runtime.h>
#include <nvstrings/NVStrings.h>

#include <cstring>
#include <iostream>
#include <numeric>

namespace cudf {
namespace io {
namespace orc {

static_assert(sizeof(orc::gpu::CompressedStreamInfo) <= 256 &&
                  !(sizeof(orc::gpu::CompressedStreamInfo) & 7),
              "Unexpected sizeof(CompressedStreamInfo)");
static_assert(sizeof(orc::gpu::ColumnDesc) <= 256 &&
                  !(sizeof(orc::gpu::ColumnDesc) & 7),
              "Unexpected sizeof(ColumnDesc)");

#if 0
#define LOG_PRINTF(...) std::printf(__VA_ARGS__)
#else
#define LOG_PRINTF(...) (void)0
#endif

/**
 * @brief Function that translates ORC datatype to GDF dtype
 **/
constexpr std::pair<gdf_dtype, gdf_dtype_extra_info> to_dtype(
    const orc::SchemaType &schema, bool use_np_dtypes = true) {
  switch (schema.kind) {
    case orc::BOOLEAN:
      return std::make_pair(GDF_BOOL8, gdf_dtype_extra_info{TIME_UNIT_NONE});
    case orc::BYTE:
      return std::make_pair(GDF_INT8, gdf_dtype_extra_info{TIME_UNIT_NONE});
    case orc::SHORT:
      return std::make_pair(GDF_INT16, gdf_dtype_extra_info{TIME_UNIT_NONE});
    case orc::INT:
      return std::make_pair(GDF_INT32, gdf_dtype_extra_info{TIME_UNIT_NONE});
    case orc::LONG:
      return std::make_pair(GDF_INT64, gdf_dtype_extra_info{TIME_UNIT_NONE});
    case orc::FLOAT:
      return std::make_pair(GDF_FLOAT32, gdf_dtype_extra_info{TIME_UNIT_NONE});
    case orc::DOUBLE:
      return std::make_pair(GDF_FLOAT64, gdf_dtype_extra_info{TIME_UNIT_NONE});
    case orc::STRING:
    case orc::BINARY:
    case orc::VARCHAR:
    case orc::CHAR:
      // Variable-length types can all be mapped to GDF_STRING
      return std::make_pair(GDF_STRING, gdf_dtype_extra_info{TIME_UNIT_NONE});
    case orc::TIMESTAMP:
      // There isn't a GDF_TIMESTAMP -> np.dtype mapping so use np.datetime64
      return (use_np_dtypes) ? std::make_pair(GDF_DATE64, gdf_dtype_extra_info{TIME_UNIT_ms})
                             : std::make_pair(GDF_TIMESTAMP, gdf_dtype_extra_info{TIME_UNIT_ms});
    case orc::DATE:
      // There isn't a GDF_DATE32 -> np.dtype mapping so use np.datetime64
      return (use_np_dtypes) ? std::make_pair(GDF_DATE64, gdf_dtype_extra_info{TIME_UNIT_ms})
                             : std::make_pair(GDF_DATE32, gdf_dtype_extra_info{TIME_UNIT_NONE});
    case orc::DECIMAL:
      // There isn't an arbitrary-precision type in cuDF, so map as float
      static_assert(DECIMALS_AS_FLOAT64 == 1, "Missing decimal->float");
      return std::make_pair(GDF_FLOAT64, gdf_dtype_extra_info{TIME_UNIT_NONE});
    default:
      break;
  }

  return std::make_pair(GDF_invalid, gdf_dtype_extra_info{TIME_UNIT_NONE});
}

constexpr std::pair<orc::gpu::StreamIndexType, uint32_t> get_index_type_and_pos(
    const orc::StreamKind kind, uint32_t skip_count, bool non_child) {
  switch (kind) {
    case orc::DATA:
      skip_count += 1;
      skip_count |= (skip_count & 0xff) << 8;
      return std::make_pair(orc::gpu::CI_DATA, skip_count);
    case orc::LENGTH:
    case orc::SECONDARY:
      skip_count += 1;
      skip_count |= (skip_count & 0xff) << 16;
      return std::make_pair(orc::gpu::CI_DATA2, skip_count);
    case orc::DICTIONARY_DATA:
      return std::make_pair(orc::gpu::CI_DICTIONARY, skip_count);
    case orc::PRESENT:
      skip_count += (non_child ? 1 : 0);
      return std::make_pair(orc::gpu::CI_PRESENT, skip_count);
    case orc::ROW_INDEX:
      return std::make_pair(orc::gpu::CI_INDEX, skip_count);
    default:
      // Skip this stream as it's not strictly required
      return std::make_pair(orc::gpu::CI_NUM_STREAMS, 0);
  }
}

/**
 * @brief A helper class for ORC file metadata. Provides some additional
 * convenience methods for initializing and accessing metadata.
 **/
class OrcMetadata {
  using OrcStripeInfo =
      std::pair<const orc::StripeInformation *, const orc::StripeFooter *>;

 public:
  explicit OrcMetadata(datasource *const src) : source(src) {
    const auto len = source->size();
    const auto max_ps_size = std::min(len, static_cast<size_t>(256));

    // Read uncompressed postscript section (max 255 bytes + 1 byte for length)
    auto buffer = source->get_buffer(len - max_ps_size, max_ps_size);
    const size_t ps_length = buffer->data()[max_ps_size - 1];
    const uint8_t *ps_data = &buffer->data()[max_ps_size - ps_length - 1];
    orc::ProtobufReader pb;
    pb.init(ps_data, ps_length);
    CUDF_EXPECTS(pb.read(&ps, ps_length), "Cannot read postscript");
    CUDF_EXPECTS(ps.footerLength + ps_length < len, "Invalid footer length");
    print_postscript(ps_length);

    // If compression is used, all the rest of the metadata is compressed
    // If no compressed is used, the decompressor is simply a pass-through
    decompressor = std::make_unique<orc::OrcDecompressor>(
        ps.compression, ps.compressionBlockSize);

    // Read compressed filefooter section
    buffer = source->get_buffer(len - ps_length - 1 - ps.footerLength,
                                ps.footerLength);
    size_t ff_length = 0;
    auto ff_data = decompressor->Decompress(buffer->data(), ps.footerLength, &ff_length);
    pb.init(ff_data, ff_length);
    CUDF_EXPECTS(pb.read(&ff, ff_length), "Cannot read filefooter");
    CUDF_EXPECTS(get_num_columns() > 0, "No columns found");
    print_filefooter();
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
  auto select_stripes(int stripe, int row_start, int &row_count) {
    std::vector<OrcStripeInfo> selection;

    if (stripe != -1) {
      CUDF_EXPECTS(stripe < get_num_stripes(), "Non-existent stripe");
      for (int i = 0; i < stripe; ++i) {
        row_start += ff.stripes[i].numberOfRows;
      }
      selection.emplace_back(&ff.stripes[stripe], nullptr);
      row_count = ff.stripes[stripe].numberOfRows;
    } else {
      row_start = std::max(row_start, 0);
      if (row_count == -1) {
        row_count = get_total_rows();
      }
      CUDF_EXPECTS(row_count >= 0, "Invalid row count");
      CUDF_EXPECTS(row_start <= get_total_rows(), "Invalid row start");

      for (int i = 0, count = 0; i < (int)ff.stripes.size(); ++i) {
        count += ff.stripes[i].numberOfRows;
        if (count > row_start || count == 0) {
          selection.emplace_back(&ff.stripes[i], nullptr);
        }
        if (count >= (row_start + row_count)) {
          break;
        }
      }
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

 private:
  void print_postscript(size_t ps_length) const {
    LOG_PRINTF("\n[+] PostScript:\n");
    LOG_PRINTF(" postscriptLength = %zd\n", ps_length);
    LOG_PRINTF(" footerLength = %zd\n", (size_t)ps.footerLength);
    LOG_PRINTF(" compression = %d\n", ps.compression);
    LOG_PRINTF(" compressionBlockSize = %d\n", ps.compressionBlockSize);
    LOG_PRINTF(" version(%zd) = {%d,%d}\n", ps.version.size(),
               (ps.version.size() > 0) ? (int32_t)ps.version[0] : -1,
               (ps.version.size() > 1) ? (int32_t)ps.version[1] : -1);
    LOG_PRINTF(" metadataLength = %zd\n", (size_t)ps.metadataLength);
    LOG_PRINTF(" magic = \"%s\"\n", ps.magic.c_str());
  }

  void print_filefooter() const {
    LOG_PRINTF("\n[+] FileFooter:\n");
    LOG_PRINTF(" headerLength = %zd\n", ff.headerLength);
    LOG_PRINTF(" contentLength = %zd\n", ff.contentLength);
    LOG_PRINTF(" stripes (%zd entries):\n", ff.stripes.size());
    for (size_t i = 0; i < ff.stripes.size(); i++) {
      LOG_PRINTF(
          "  [%zd] @ %zd: %d rows, index+data+footer: %zd+%zd+%d bytes\n", i,
          ff.stripes[i].offset, ff.stripes[i].numberOfRows,
          ff.stripes[i].indexLength, ff.stripes[i].dataLength,
          ff.stripes[i].footerLength);
    }
    LOG_PRINTF(" types (%zd entries):\n", ff.types.size());
    for (size_t i = 0; i < ff.types.size(); i++) {
      LOG_PRINTF("  column [%zd]: kind = %d, parent = %d\n", i,
                 ff.types[i].kind, ff.types[i].parent_idx);
      if (ff.types[i].subtypes.size() > 0) {
        LOG_PRINTF("   subtypes = ");
        for (size_t j = 0; j < ff.types[i].subtypes.size(); j++) {
          LOG_PRINTF("%c%d", (j) ? ',' : '{', ff.types[i].subtypes[j]);
        }
        LOG_PRINTF("}\n");
      }
      if (ff.types[i].fieldNames.size() > 0) {
        LOG_PRINTF("   fieldNames = ");
        for (size_t j = 0; j < ff.types[i].fieldNames.size(); j++) {
          LOG_PRINTF("%c\"%s\"", (j) ? ',' : '{',
                     ff.types[i].fieldNames[j].c_str());
        }
        LOG_PRINTF("}\n");
      }
    }
    if (ff.metadata.size() > 0) {
      LOG_PRINTF(" metadata (%zd entries):\n", ff.metadata.size());
      for (size_t i = 0; i < ff.metadata.size(); i++) {
        LOG_PRINTF("  [%zd] \"%s\" = \"%s\"\n", i, ff.metadata[i].name.c_str(),
                   ff.metadata[i].value.c_str());
      }
    }
    LOG_PRINTF(" numberOfRows = %zd\n", ff.numberOfRows);
    LOG_PRINTF(" rowIndexStride = %d\n", ff.rowIndexStride);
  }

 public:
  orc::PostScript ps;
  orc::FileFooter ff;
  std::vector<orc::StripeFooter> stripefooters;
  std::unique_ptr<orc::OrcDecompressor> decompressor;

 private:
  datasource *const source;
};

/**
 * @brief Struct that maps ORC streams to columns
 **/
struct OrcStreamInfo {
  OrcStreamInfo() = default;
  explicit OrcStreamInfo(uint64_t offset_, size_t dst_pos_, uint32_t length_,
                         uint32_t gdf_idx_, uint32_t stripe_idx_)
      : offset(offset_),
        dst_pos(dst_pos_),
        length(length_),
        gdf_idx(gdf_idx_),
        stripe_idx(stripe_idx_) {}
  uint64_t offset;      // offset in file
  size_t dst_pos;       // offset in memory relative to the beginning of the compressed stripe data
  uint32_t length;      // length in file
  uint32_t gdf_idx;     // gdf column index
  uint32_t stripe_idx;  // stripe index
};

size_t reader::Impl::gather_stream_info(
    const size_t stripe_index, const orc::StripeInformation *stripeinfo,
    const orc::StripeFooter *stripefooter, const std::vector<int> &orc2gdf,
    const std::vector<int> &gdf2orc, const std::vector<orc::SchemaType> types,
    bool use_index,
    size_t *num_dictionary_entries,
    hostdevice_vector<orc::gpu::ColumnDesc> &chunks,
    std::vector<OrcStreamInfo> &stream_info) {

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
              chunk.strm_id[orc::gpu::CI_PRESENT] = stream_info.size();
              chunk.strm_len[orc::gpu::CI_PRESENT] = stream.length;
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
        if (idx.first < orc::gpu::CI_NUM_STREAMS) {
          chunk.strm_id[idx.first] = stream_info.size();
          chunk.strm_len[idx.first] = stream.length;
          chunk.skip_count = idx.second;

          if (idx.first == orc::gpu::CI_DICTIONARY) {
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

device_buffer<uint8_t> reader::Impl::decompress_stripe_data(
    const hostdevice_vector<orc::gpu::ColumnDesc> &chunks,
    const std::vector<device_buffer<uint8_t>> &stripe_data,
    const orc::OrcDecompressor *decompressor,
    std::vector<OrcStreamInfo> &stream_info, size_t num_stripes,
    rmm::device_vector<orc::gpu::RowGroup> &row_groups,
    size_t row_index_stride) {

  // Parse the columns' compressed info
  hostdevice_vector<orc::gpu::CompressedStreamInfo> compinfo(0, stream_info.size());
  for (size_t i = 0; i < compinfo.max_size(); ++i) {
    compinfo.insert(orc::gpu::CompressedStreamInfo(
        stripe_data[stream_info[i].stripe_idx].data() + stream_info[i].dst_pos,
        stream_info[i].length));
  }
  CUDA_TRY(cudaMemcpyAsync(compinfo.device_ptr(), compinfo.host_ptr(),
                           compinfo.memory_size(), cudaMemcpyHostToDevice));
  CUDA_TRY(ParseCompressedStripeData(
      compinfo.device_ptr(), compinfo.size(), decompressor->GetBlockSize(),
      decompressor->GetLog2MaxCompressionRatio()));
  CUDA_TRY(cudaMemcpyAsync(compinfo.host_ptr(), compinfo.device_ptr(),
                           compinfo.memory_size(), cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaStreamSynchronize(0));

  // Count the exact number of compressed blocks
  size_t num_compressed_blocks = 0;
  size_t total_decompressed_size = 0;
  for (size_t i = 0; i < compinfo.size(); ++i) {
    num_compressed_blocks += compinfo[i].num_compressed_blocks;
    total_decompressed_size += compinfo[i].max_uncompressed_size;
  }
  CUDF_EXPECTS(total_decompressed_size > 0, "No decompressible data found");

  LOG_PRINTF(
      "[+] Compression\n Total compressed size: %zd\n Number of "
      "compressed blocks: %zd\n Codec: %d\n",
      total_decompressed_size, num_compressed_blocks, decompressor->GetKind());

  device_buffer<uint8_t> decomp_data(total_decompressed_size);
  rmm::device_vector<gpu_inflate_input_s> inflate_in(num_compressed_blocks);
  rmm::device_vector<gpu_inflate_status_s> inflate_out(num_compressed_blocks);

  // Parse again to populate the decompression input/output buffers
  size_t decomp_offset = 0;
  uint32_t start_pos = 0;
  for (size_t i = 0; i < compinfo.size(); ++i) {
    compinfo[i].uncompressed_data = decomp_data.data() + decomp_offset;
    compinfo[i].decctl = inflate_in.data().get() + start_pos;
    compinfo[i].decstatus = inflate_out.data().get() + start_pos;
    compinfo[i].max_compressed_blocks = compinfo[i].num_compressed_blocks;

    stream_info[i].dst_pos = decomp_offset;
    decomp_offset += compinfo[i].max_uncompressed_size;
    start_pos += compinfo[i].num_compressed_blocks;
  }
  CUDA_TRY(cudaMemcpyAsync(compinfo.device_ptr(), compinfo.host_ptr(),
                           compinfo.memory_size(), cudaMemcpyHostToDevice));
  CUDA_TRY(ParseCompressedStripeData(
      compinfo.device_ptr(), compinfo.size(), decompressor->GetBlockSize(),
      decompressor->GetLog2MaxCompressionRatio()));

  // Dispatch batches of blocks to decompress
  if (num_compressed_blocks > 0) {
    switch (decompressor->GetKind()) {
      case orc::ZLIB:
        CUDA_TRY(gpuinflate(inflate_in.data().get(), inflate_out.data().get(),
                            num_compressed_blocks, 0));
        break;
      case orc::SNAPPY:
        CUDA_TRY(gpu_unsnap(inflate_in.data().get(), inflate_out.data().get(),
                            num_compressed_blocks));
        break;
      default:
        CUDF_EXPECTS(false, "Unexpected decompression dispatch");
        break;
    }
  }
  CUDA_TRY(PostDecompressionReassemble(compinfo.device_ptr(), compinfo.size()));

  // Update the stream information with the updated uncompressed info
  // TBD: We could update the value from the information we already
  // have in stream_info[], but using the gpu results also updates
  // max_uncompressed_size to the actual uncompressed size, or zero if
  // decompression failed.
  CUDA_TRY(cudaMemcpyAsync(compinfo.host_ptr(), compinfo.device_ptr(),
                           compinfo.memory_size(), cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaStreamSynchronize(0));

  const size_t num_columns = chunks.size() / num_stripes;

  for (size_t i = 0; i < num_stripes; ++i) {
    for (size_t j = 0; j < num_columns; ++j) {
      auto &chunk = chunks[i * num_columns + j];
      for (int k = 0; k < orc::gpu::CI_NUM_STREAMS; ++k) {
        if (chunk.strm_len[k] > 0 && chunk.strm_id[k] < compinfo.size()) {
          chunk.streams[k] = compinfo[chunk.strm_id[k]].uncompressed_data;
          chunk.strm_len[k] = compinfo[chunk.strm_id[k]].max_uncompressed_size;
        }
      }
    }
  }

  if (not row_groups.empty()) {
    CUDA_TRY(cudaMemcpyAsync(chunks.device_ptr(), chunks.host_ptr(),
                             chunks.memory_size(), cudaMemcpyHostToDevice));
    CUDA_TRY(ParseRowGroupIndex(row_groups.data().get(), compinfo.device_ptr(),
                                chunks.device_ptr(), num_columns, num_stripes,
                                row_groups.size() / num_columns,
                                row_index_stride));
  }

  return decomp_data;
}

void reader::Impl::decode_stream_data(
    const hostdevice_vector<orc::gpu::ColumnDesc> &chunks, size_t num_dicts,
    size_t skip_rows, const std::vector<int64_t> &timezone_table,
    rmm::device_vector<orc::gpu::RowGroup> &row_groups, size_t row_index_stride,
    const std::vector<gdf_column_wrapper> &columns) {

  const size_t num_columns = columns.size();
  const size_t num_stripes = chunks.size() / columns.size();
  const size_t num_rows = columns[0]->size;

  // Update chunks with pointers to column data
  for (size_t i = 0; i < num_stripes; ++i) {
    for (size_t j = 0; j < num_columns; ++j) {
      auto &chunk = chunks[i * num_columns + j];
      chunk.valid_map_base = reinterpret_cast<uint32_t *>(columns[j]->valid);
      chunk.column_data_base = columns[j]->data;
      chunk.dtype_len = (columns[j]->dtype == GDF_STRING)
                            ? sizeof(std::pair<const char *, size_t>)
                            : cudf::size_of(columns[j]->dtype);
    }
  }

  // Allocate global dictionary for deserializing
  rmm::device_vector<orc::gpu::DictionaryEntry> global_dict(num_dicts);

  // Allocate timezone transition table timestamp conversion
  rmm::device_vector<int64_t> tz_table = timezone_table;

  CUDA_TRY(cudaMemcpyAsync(chunks.device_ptr(), chunks.host_ptr(),
                           chunks.memory_size(), cudaMemcpyHostToDevice));
  CUDA_TRY(DecodeNullsAndStringDictionaries(
      chunks.device_ptr(), global_dict.data().get(), num_columns, num_stripes,
      num_rows, skip_rows));
  CUDA_TRY(DecodeOrcColumnData(
      chunks.device_ptr(), global_dict.data().get(), num_columns, num_stripes,
      num_rows, skip_rows, tz_table.data().get(), tz_table.size(),
      row_groups.data().get(), row_groups.size() / num_columns,
      row_index_stride));
  CUDA_TRY(cudaMemcpyAsync(chunks.host_ptr(), chunks.device_ptr(),
                           chunks.memory_size(), cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaStreamSynchronize(0));

  LOG_PRINTF("[+] Decoded Column Information\n");
  for (size_t i = 0; i < num_columns; ++i) {
    for (size_t j = 0; j < num_stripes; ++j) {
      columns[i]->null_count += chunks[j * num_columns + i].null_count;
    }
    LOG_PRINTF(
        "columns[%zd].null_count = %d/%d (start_row=%d, nrows=%d, "
        "strm_len=%d)\n",
        i, columns[i]->null_count, columns[i]->size, chunks[i].start_row,
        chunks[i].num_rows, chunks[i].strm_len[orc::gpu::CI_PRESENT]);
  }
}

reader::Impl::Impl(std::unique_ptr<datasource> source,
                   reader_options const &options)
    : source_(std::move(source)) {

  // Open and parse the source Parquet dataset metadata
  md_ = std::make_unique<OrcMetadata>(source_.get());

  // Select only columns required by the options
  selected_cols_ = md_->select_columns(options.columns, has_timestamp_column_);

  // Enable or disable attempt to use row index for parsing
  use_index_ = options.use_index;

  // Enable or disable the conversion to numpy-compatible dtypes
  use_np_dtypes_ = options.use_np_dtypes;
}

table reader::Impl::read(int skip_rows, int num_rows, int stripe) {

  // Select only stripes required (aka row groups)
  const auto selected_stripes =
      md_->select_stripes(stripe, skip_rows, num_rows);
  const int num_columns = selected_cols_.size();

  // Association between each ORC column and its gdf_column
  std::vector<int32_t> orc_col_map(md_->get_num_columns(), -1);

  // Initialize gdf_columns, but hold off on allocating storage space
  std::vector<gdf_column_wrapper> columns;
  LOG_PRINTF("[+] Selected columns: %d\n", num_columns);
  for (const auto &col : selected_cols_) {
    auto dtype_info = to_dtype(md_->ff.types[col], use_np_dtypes_);

    // Map each ORC column to its gdf_column
    orc_col_map[col] = columns.size();

    columns.emplace_back(static_cast<gdf_size_type>(num_rows), dtype_info.first,
                         dtype_info.second, md_->ff.GetColumnName(col));

    LOG_PRINTF(" %2zd: name=%s size=%zd type=%d data=%lx valid=%lx\n",
               columns.size() - 1, columns.back()->col_name,
               (size_t)columns.back()->size, columns.back()->dtype,
               (uint64_t)columns.back()->data, (uint64_t)columns.back()->valid);
  }

  // Logically view streams as columns
  std::vector<OrcStreamInfo> stream_info;

  // Tracker for eventually deallocating compressed and uncompressed data
  std::vector<device_buffer<uint8_t>> stripe_data;

  if (num_rows > 0) {
    const auto num_column_chunks = selected_stripes.size() * num_columns;
    hostdevice_vector<orc::gpu::ColumnDesc> chunks(num_column_chunks);
    memset(chunks.host_ptr(), 0, chunks.memory_size());

    const bool use_index =
        (use_index_ == true) &&
        // Only use if we don't have much work with complete columns & stripes
        // TODO: Consider nrows, gpu, and tune the threshold
        (num_rows > md_->get_row_index_stride() &&
         !(md_->get_row_index_stride() & 7) && md_->get_row_index_stride() > 0 &&
         num_columns * selected_stripes.size() < 8 * 128) &&
        // Only use if first row is aligned to a stripe boundary
        // TODO: Fix logic to handle unaligned rows
        (skip_rows == 0);

    size_t stripe_start_row = 0;
    size_t num_dict_entries = 0;
    size_t num_rowgroups = 0;
    for (size_t i = 0; i < selected_stripes.size(); ++i) {
      const auto stripe_info = selected_stripes[i].first;
      const auto stripe_footer = selected_stripes[i].second;

      auto stream_count = stream_info.size();
      const auto total_data_size = gather_stream_info(
          i, stripe_info, stripe_footer, orc_col_map, selected_cols_,
          md_->ff.types, use_index, &num_dict_entries, chunks, stream_info);
      CUDF_EXPECTS(total_data_size > 0, "Expected streams data within stripe");

      stripe_data.emplace_back(total_data_size);
      uint8_t *d_data = stripe_data.back().data();

      // Coalesce consecutive streams into one read
      while (stream_count < stream_info.size()) {
        const auto d_dst = d_data + stream_info[stream_count].dst_pos;
        const auto offset = stream_info[stream_count].offset;
        auto len = stream_info[stream_count].length;
        stream_count++;

        while (stream_count < stream_info.size() &&
               stream_info[stream_count].offset == offset + len) {
          len += stream_info[stream_count].length;
          stream_count++;
        }
        const auto buffer = source_->get_buffer(offset, len);
        CUDA_TRY(cudaMemcpyAsync(d_dst, buffer->data(), len,
                                 cudaMemcpyHostToDevice));
        CUDA_TRY(cudaStreamSynchronize(0));
      }

      // Update chunks to reference streams pointers
      for (int j = 0; j < num_columns; j++) {
        auto &chunk = chunks[i * num_columns + j];
        chunk.start_row = stripe_start_row;
        chunk.num_rows = stripe_info->numberOfRows;
        chunk.encoding_kind = stripe_footer->columns[selected_cols_[j]].kind;
        chunk.type_kind = md_->ff.types[selected_cols_[j]].kind;
        chunk.decimal_scale = md_->ff.types[selected_cols_[j]].scale;
        chunk.rowgroup_id = num_rowgroups;
        for (int k = 0; k < orc::gpu::CI_NUM_STREAMS; k++) {
          if (chunk.strm_len[k] > 0) {
            chunk.streams[k] = d_data + stream_info[chunk.strm_id[k]].dst_pos;
          }
        }
      }
      stripe_start_row += stripe_info->numberOfRows;
      if (use_index) {
        num_rowgroups +=
            (stripe_info->numberOfRows + md_->get_row_index_stride() - 1) /
            md_->get_row_index_stride();
      }
    }

    // Setup table for converting timestamp columns from local to UTC time
    std::vector<int64_t> tz_table;
    if (has_timestamp_column_ && !selected_stripes.empty()) {
      CUDF_EXPECTS(BuildTimezoneTransitionTable(
                       tz_table, selected_stripes[0].second->writerTimezone),
                   "Cannot setup timezone LUT");
    }

    // Setup row group descriptors if using indexes
    rmm::device_vector<orc::gpu::RowGroup> row_groups(num_rowgroups *
                                                      num_columns);

    if (md_->ps.compression != orc::NONE) {
      auto decomp_data = decompress_stripe_data(
          chunks, stripe_data, md_->decompressor.get(), stream_info,
          selected_stripes.size(), row_groups, md_->get_row_index_stride());
      stripe_data.clear();
      stripe_data.push_back(std::move(decomp_data));
    } else {
      if (not row_groups.empty()) {
        CUDA_TRY(cudaMemcpyAsync(chunks.device_ptr(), chunks.host_ptr(),
                                 chunks.memory_size(), cudaMemcpyHostToDevice));
        CUDA_TRY(ParseRowGroupIndex(
            row_groups.data().get(), nullptr, chunks.device_ptr(), num_columns,
            selected_stripes.size(), num_rowgroups, md_->get_row_index_stride()));
      }
    }

    for (auto &column : columns) {
      CUDF_EXPECTS(column.allocate() == GDF_SUCCESS, "Cannot allocate columns");
      if (column->dtype == GDF_STRING) {
        // Kernel doesn't init invalid entries but NvStrings expects zero length
        CUDA_TRY(cudaMemsetAsync(
            column->data, 0,
            column->size * sizeof(std::pair<const char *, size_t>)));
      }
    }
    decode_stream_data(chunks, num_dict_entries, skip_rows, tz_table,
                       row_groups, md_->get_row_index_stride(), columns);
  } else {
    // Columns' data's memory is still expected for an empty dataframe
    for (auto &column : columns) {
      CUDF_EXPECTS(column.allocate() == GDF_SUCCESS, "Cannot allocate columns");
      if (column->dtype == GDF_STRING) {
        // Kernel doesn't init invalid entries but NvStrings expects zero length
        CUDA_TRY(cudaMemsetAsync(
            column->data, 0,
            column->size * sizeof(std::pair<const char *, size_t>)));
      }
    }
  }

  // For string dtype, allocate an NvStrings container instance, deallocating
  // the original string list memory in the process.
  // This container takes a list of string pointers and lengths, and copies
  // into its own memory so the source memory must not be released yet.
  for (auto &column : columns) {
    if (column->dtype == GDF_STRING) {
      using str_pair = std::pair<const char *, size_t>;
      using str_ptr = std::unique_ptr<NVStrings, decltype(&NVStrings::destroy)>;

      auto str_list = static_cast<str_pair *>(column->data);
      str_ptr str_data(NVStrings::create_from_index(str_list, num_rows),
                       &NVStrings::destroy);
      RMM_FREE(std::exchange(column->data, str_data.release()), 0);
    }
  }

  // Transfer ownership to raw pointer output arguments
  std::vector<gdf_column *> out_cols(columns.size());
  for (size_t i = 0; i < columns.size(); ++i) {
    out_cols[i] = columns[i].release();
  }

  return table(out_cols.data(), out_cols.size());
}

reader::reader(std::string filepath, reader_options const &options)
    : impl_(std::make_unique<Impl>(datasource::create(filepath), options)) {}

reader::reader(const char *buffer, size_t length, reader_options const &options)
    : impl_(std::make_unique<Impl>(datasource::create(buffer, length),
                                   options)) {}

reader::reader(std::shared_ptr<arrow::io::RandomAccessFile> file,
               reader_options const &options)
    : impl_(std::make_unique<Impl>(datasource::create(file), options)) {}

table reader::read_all() { return impl_->read(0, -1, -1); }

table reader::read_rows(size_t skip_rows, size_t num_rows) {
  return impl_->read(skip_rows, (num_rows != 0) ? (int)num_rows : -1, -1);
}

table reader::read_stripe(size_t stripe) { return impl_->read(0, -1, stripe); }

reader::~reader() = default;

} // namespace orc
} // namespace io
} // namespace cudf
