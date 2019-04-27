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

#include "orc.h"
#include "orc_gpu.h"
#include "timezone.h"

#include "cudf.h"
#include "io/comp/gpuinflate.h"
#include "io/utilities/wrapper_utils.hpp"
#include "utilities/cudf_utils.h"
#include "utilities/error_utils.hpp"

#include <cuda_runtime.h>
#include <nvstrings/NVStrings.h>
#include <rmm/thrust_rmm_allocator.h>

#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

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
 * @brief Helper class for memory mapping a file source
 **/
class DataSource {
 public:
  explicit DataSource(const char *filepath) {
    fd = open(filepath, O_RDONLY);
    CUDF_EXPECTS(fd > 0, "Cannot open file");

    struct stat st {};
    CUDF_EXPECTS(fstat(fd, &st) == 0, "Cannot query file size");

    mapped_size = st.st_size;
    CUDF_EXPECTS(mapped_size > 0, "Unexpected zero-sized file");

    mapped_data = mmap(NULL, mapped_size, PROT_READ, MAP_PRIVATE, fd, 0);
    CUDF_EXPECTS(mapped_data != MAP_FAILED, "Cannot memory-mapping file");
  }

  ~DataSource() {
    if (mapped_data) {
      munmap(mapped_data, mapped_size);
    }
    if (fd) {
      close(fd);
    }
  }

  const uint8_t *data() const { return static_cast<uint8_t *>(mapped_data); }
  size_t size() const { return mapped_size; }

 private:
  void *mapped_data = nullptr;
  size_t mapped_size = 0;
  int fd = 0;
};

/**
 * @brief Function that translates ORC datatype to GDF dtype
 **/
constexpr std::pair<gdf_dtype, gdf_dtype_extra_info> to_dtype(
    const orc::SchemaType &schema) {
  switch (schema.kind) {
    case orc::BOOLEAN:
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
      return std::make_pair(GDF_DATE64, gdf_dtype_extra_info{TIME_UNIT_ms});
    case orc::DATE:
      // There isn't a GDF_DATE32 -> np.dtype mapping so use np.datetime64
      return std::make_pair(GDF_DATE64, gdf_dtype_extra_info{TIME_UNIT_ms});
    case orc::DECIMAL:
      // There isn't an arbitrary-precision type in cuDF, so map as float
      static_assert(DECIMALS_AS_FLOAT64 == 1, "Missing decimal->float");
      return std::make_pair(GDF_FLOAT64, gdf_dtype_extra_info{TIME_UNIT_NONE});
    default:
      break;
  }

  return std::make_pair(GDF_invalid, gdf_dtype_extra_info{TIME_UNIT_NONE});
}

constexpr orc::gpu::StreamIndexType to_stream_index(
    const orc::StreamKind kind) {
  switch (kind) {
    case orc::DATA:
      return orc::gpu::CI_DATA;
    case orc::LENGTH:
    case orc::SECONDARY:
      return orc::gpu::CI_DATA2;
    case orc::DICTIONARY_DATA:
      return orc::gpu::CI_DICTIONARY;
    case orc::PRESENT:
      return orc::gpu::CI_PRESENT;
    default:
      // Skip this stream as it's not strictly required
      break;
  }

  return orc::gpu::CI_NUM_STREAMS;
}

/**
 * @brief A helper class for ORC file metadata. Provides some additional
 * convenience methods for initializing and accessing metadata.
 **/
class OrcMetadata {
  using OrcStripeInfo =
      std::pair<const orc::StripeInformation *, orc::StripeFooter>;

 public:
  explicit OrcMetadata(const uint8_t *data_, size_t len_)
      : data(data_), len(len_) {
    const auto ps_length = data[len - 1];
    const auto ps_data = &data[len - ps_length - 1];

    // Read uncompressed postscript section
    orc::ProtobufReader pb;
    pb.init(ps_data, ps_length);
    CUDF_EXPECTS(pb.read(&ps, ps_length), "Cannot read postscript");
    CUDF_EXPECTS(ps.footerLength + ps_length < len, "Invalid footer length");

    LOG_PRINTF("\n[+] PostScript:\n");
    LOG_PRINTF(" postscriptLength = %d\n", ps_length);
    LOG_PRINTF(" footerLength = %zd\n", (size_t)ps.footerLength);
    LOG_PRINTF(" compression = %d\n", ps.compression);
    LOG_PRINTF(" compressionBlockSize = %d\n", ps.compressionBlockSize);
    LOG_PRINTF(" version(%zd) = {%d,%d}\n", ps.version.size(),
               (ps.version.size() > 0) ? (int32_t)ps.version[0] : -1,
               (ps.version.size() > 1) ? (int32_t)ps.version[1] : -1);
    LOG_PRINTF(" metadataLength = %zd\n", (size_t)ps.metadataLength);
    LOG_PRINTF(" magic = \"%s\"\n", ps.magic.c_str());

    // If compression is used, all the rest of the metadata is compressed
    // If no compressed is used, the decompressor is simply a pass-through
    decompressor = std::make_unique<orc::OrcDecompressor>(
        ps.compression, ps.compressionBlockSize);

    // Read compressed filefooter section
    size_t ff_length = 0;
    auto ff_data = decompressor->Decompress(ps_data - ps.footerLength,
                                            ps.footerLength, &ff_length);
    pb.init(ff_data, ff_length);
    CUDF_EXPECTS(pb.read(&ff, ff_length), "Cannot read filefooter");

    LOG_PRINTF("\n[+] FileFooter:\n");
    LOG_PRINTF(" headerLength = %zd\n", (size_t)ff.headerLength);
    LOG_PRINTF(" contentLength = %zd\n", (size_t)ff.contentLength);
    for (size_t i = 0; i < ff.stripes.size(); i++) {
      LOG_PRINTF(
          " stripe #%zd @ %zd: %d rows, index+data+footer: %zd+%zd+%d bytes\n",
          i, (size_t)ff.stripes[i].offset, ff.stripes[i].numberOfRows,
          (size_t)ff.stripes[i].indexLength, (size_t)ff.stripes[i].dataLength,
          ff.stripes[i].footerLength);
    }
    for (size_t i = 0; i < ff.types.size(); i++) {
      LOG_PRINTF(" column %zd: kind=%d, parent=%d\n", i, ff.types[i].kind,
                 ff.types[i].parent_idx);
      if (ff.types[i].subtypes.size() > 0) {
        LOG_PRINTF("   subtypes = ");
        for (int j = 0; j < (int)ff.types[i].subtypes.size(); j++) {
          LOG_PRINTF("%c%d", (j) ? ',' : '{', ff.types[i].subtypes[j]);
        }
        LOG_PRINTF("}\n");
      }
      if (ff.types[i].fieldNames.size() > 0) {
        LOG_PRINTF("   fieldNames = ");
        for (int j = 0; j < (int)ff.types[i].fieldNames.size(); j++) {
          LOG_PRINTF("%c\"%s\"", (j) ? ',' : '{',
                 ff.types[i].fieldNames[j].c_str());
        }
        LOG_PRINTF("}\n");
      }
    }
    for (size_t i = 0; i < ff.metadata.size(); i++) {
      LOG_PRINTF(" metadata: \"%s\" = \"%s\"\n", ff.metadata[i].name.c_str(),
             ff.metadata[i].value.c_str());
    }
    LOG_PRINTF(" numberOfRows = %zd\n", (size_t)ff.numberOfRows);
    LOG_PRINTF(" rowIndexStride = %d\n", ff.rowIndexStride);
  }

  /**
   * @brief Filters and reads the info of only a selection of stripes
   *
   * @param[in] row_start Starting row of the selection
   * @param[in,out] row_count Total number of rows selected
   *
   * @return List of stripe info and total number of selected rows
   **/
  auto select_stripes(int row_start, int &row_count) {
    std::vector<OrcStripeInfo> selection;

    // Exclude non-needed stripes
    row_start = std::max(row_start, 0);
    if (row_count == -1) {
      row_count = get_total_rows();
    }
    CUDF_EXPECTS(row_count >= 0, "Invalid row count");
    CUDF_EXPECTS(row_start <= get_total_rows(), "Invalid row start");
    while (ff.stripes.size() > 0 && ff.stripes[0].numberOfRows <= uint32_t(row_start)) {
      ff.numberOfRows -= ff.stripes[0].numberOfRows;
      row_start -= ff.stripes[0].numberOfRows;
      ff.stripes.erase(ff.stripes.begin());
    }
    if (row_count == 0) {
      row_count = ff.numberOfRows - std::min<int>(row_start, ff.numberOfRows);
    } else {
      row_count = std::min<int>(row_count, ff.numberOfRows - std::min<int>(row_start, ff.numberOfRows));
    }
    if (ff.numberOfRows > uint64_t(row_count)) {
      uint64_t row = 0;
      for (size_t i = 0; i < ff.stripes.size(); i++) {
        if (row >= uint64_t(row_count)) {
          ff.stripes.resize(i);
          ff.numberOfRows = row;
          break;
        }
        row += ff.stripes[i].numberOfRows;
      }
    }
    assert(row_count == ff.numberOfRows);

    // Read each stripe's stripefooter metadata
    selection.resize(ff.stripes.size());
    for (size_t i = 0; i < ff.stripes.size(); ++i) {
      const auto &stripe = ff.stripes[i];
      const auto sf_comp_offset =
          stripe.offset + stripe.indexLength + stripe.dataLength;
      const auto sf_comp_length = stripe.footerLength;
      CUDF_EXPECTS(sf_comp_offset + sf_comp_length < len,
                   "Invalid stripe information");

      size_t sf_length = 0;
      auto sf_data = decompressor->Decompress(data + sf_comp_offset,
                                              sf_comp_length, &sf_length);
      orc::ProtobufReader pb;
      pb.init(sf_data, sf_length);
      CUDF_EXPECTS(pb.read(&selection[i].second, sf_length),
                   "Cannot read stripefooter");
      selection[i].first = &stripe;
    }

    return selection;
  }

  /**
   * @brief Filters and reduces down to a selection of columns
   *
   * @param[in] use_cols Array of column names to select
   * @param[in] use_cols_len Length of the column name array
   * @param[out] has_timestamp_column Whether there is a orc::TIMESTAMP column
   *
   * @return A list of ORC column indexes
   **/
  auto select_columns(const char **use_cols, int use_cols_len,
                      bool &has_timestamp_column) {
    std::vector<int> selection;

    if (use_cols) {
      std::vector<std::string> use_names(use_cols, use_cols + use_cols_len);
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
  inline int get_num_rowgroups() const { return ff.stripes.size(); }
  inline int get_num_columns() const { return ff.types.size(); }

 public:
  orc::PostScript ps;
  orc::FileFooter ff;
  std::unique_ptr<orc::OrcDecompressor> decompressor;

 private:
  const uint8_t *const data;
  const size_t len;
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

/**
 * @brief Helper function that populates column descriptors stream/chunk
 *
 * @param[in] chunks List of column chunk descriptors
 * @param[in] stripeinfo List of stripe metadata
 * @param[in] stripefooter List of stripe footer metadata
 * @param[in] orc2gdf Mapping of ORC columns to cuDF columns
 * @param[in] gdf2orc Mapping of cuDF columns to ORC columns
 * @param[in] types List of schema types in the dataset
 * @param[out] num_dictionary_entries Number of required dictionary entries
 * @param[in,out] chunks List of column chunk descriptions
 * @param[in,out] stream_info List of column stream info
 *
 * @return size_t Size in bytes of readable stream data found
 **/
size_t gather_stream_info(const size_t stripe_index,
                          const orc::StripeInformation& stripeinfo,
                          const orc::StripeFooter &stripefooter,
                          const std::vector<int> &orc2gdf,
                          const std::vector<int> &gdf2orc,
                          const std::vector<orc::SchemaType> types,
                          size_t* num_dictionary_entries,
                          hostdevice_vector<orc::gpu::ColumnDesc> &chunks,
                          std::vector<OrcStreamInfo> &stream_info) {

  const auto num_columns = gdf2orc.size();
  uint64_t src_offset = 0;
  uint64_t dst_offset = 0;
  for (const auto &stream : stripefooter.streams) {
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
        if (schema_type.kind == orc::STRUCT &&
            stream.kind == orc::PRESENT) {
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
      if (src_offset >= stripeinfo.indexLength) {
        const auto idx = to_stream_index(stream.kind);
        if (idx < orc::gpu::CI_NUM_STREAMS) {
          auto &chunk = chunks[stripe_index * num_columns + col];
          chunk.strm_id[idx] = stream_info.size();
          chunk.strm_len[idx] = stream.length;

          if (idx == orc::gpu::CI_DICTIONARY) {
            chunk.dictionary_start = *num_dictionary_entries;
            chunk.dict_len =
                stripefooter.columns[stream.column].dictionarySize;
            *num_dictionary_entries +=
                stripefooter.columns[stream.column].dictionarySize;
          }
        }
      }
      stream_info.emplace_back(stripeinfo.offset + src_offset, dst_offset,
                               stream.length, col, stripe_index);
      dst_offset += stream.length;
    }
    src_offset += stream.length;
  }

  return dst_offset;
}

/**
 * @brief Decompresses the stripe data, at stream granularity
 *
 * @param[in] chunks List of column chunk descriptors
 * @param[in] stripe_data List of source stripe column data
 * @param[in] decompressor Originally host decompressor
 * @param[in] stream_info List of stream to column mappings
 * @param[in] num_stripes Number of stripes making up column chunks
 *
 * @return device_buffer<uint8_t> Device buffer to decompressed page data
 **/
device_buffer<uint8_t> decompress_stripe_data(
    const hostdevice_vector<orc::gpu::ColumnDesc> &chunks,
    const std::vector<device_buffer<uint8_t>> &stripe_data,
    const orc::OrcDecompressor *decompressor,
    std::vector<OrcStreamInfo> &stream_info, size_t num_stripes) {

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
  CUDA_TRY(PostDecompressionReassemble(compinfo.device_ptr(), compinfo.size()));

  // Update the stream information with the updated uncompressed info
  // TBD: We could update the value from the information we already
  // have in stream_info[], but using the gpu results also updates
  // max_uncompressed_size to the actual uncompressed size, or zero if
  // decompression failed.
  CUDA_TRY(cudaMemcpyAsync(compinfo.host_ptr(), compinfo.device_ptr(),
                           compinfo.memory_size(), cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaStreamSynchronize(0));

  // const auto num_stripes = md.ff.stripes.size();
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

  return decomp_data;
}

/**
 * @brief Converts the stripe column data and outputs to gdf_columns
 *
 * @param[in] chunks List of column chunk descriptors
 * @param[in] num_dicts Number of dictionary entries required
 * @param[in] skip_rows Number of rows to offset from start
 * @param[in] timezone_table Local time to UTC conversion table
 * @param[in,out] columns List of gdf_columns
 **/
void decode_stream_data(const hostdevice_vector<orc::gpu::ColumnDesc> &chunks,
                        size_t num_dicts, size_t skip_rows,
                        const std::vector<int64_t> &timezone_table,
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
                            : gdf_dtype_size(columns[j]->dtype);
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
  CUDA_TRY(DecodeOrcColumnData(chunks.device_ptr(), global_dict.data().get(),
                               num_columns, num_stripes, num_rows, skip_rows,
                               tz_table.data().get(), tz_table.size()));
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

//
// Implementation for reading Apache ORC data and outputting to cuDF columns.
//
gdf_error read_orc(orc_read_arg *args) {

  DataSource input(args->source);

  OrcMetadata md(input.data(), input.size());
  CUDF_EXPECTS(md.get_num_columns() > 0, "No columns found");

  // Select only stripes required (aka row groups)
  int skip_rows = args->skip_rows;
  int num_rows = args->num_rows;
  const auto selected_stripes = md.select_stripes(skip_rows, num_rows);

  // Select only columns required
  bool has_time_stamp_column = false;
  const auto selected_cols = md.select_columns(
      args->use_cols, args->use_cols_len, has_time_stamp_column);
  const int num_columns = selected_cols.size();

  // Association between each ORC column and its gdf_column
  std::vector<int32_t> orc_col_map(md.get_num_columns(), -1);

  // Initialize gdf_columns, but hold off on allocating storage space
  std::vector<gdf_column_wrapper> columns;
  LOG_PRINTF("[+] Selected columns: %d\n", num_columns);
  for (const auto &col : selected_cols) {
    auto dtype_info = to_dtype(md.ff.types[col]);

    // Map each ORC column to its gdf_column
    orc_col_map[col] = columns.size();

    columns.emplace_back(static_cast<gdf_size_type>(num_rows),
                         dtype_info.first, dtype_info.second,
                         md.ff.GetColumnName(col));

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

    size_t stripe_start_row = 0;
    size_t num_dict_entries = 0;
    for (size_t i = 0; i < selected_stripes.size(); ++i) {
      const auto stripe_info = selected_stripes[i].first;
      const auto stripe_footer = selected_stripes[i].second;

      auto stream_count = stream_info.size();
      const auto total_data_size = gather_stream_info(
          i, *stripe_info, stripe_footer, orc_col_map, selected_cols,
          md.ff.types, &num_dict_entries, chunks, stream_info);
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
        CUDA_TRY(cudaMemcpyAsync(d_dst, input.data() + offset, len,
                                 cudaMemcpyHostToDevice));
      }

      // Update chunks to reference streams pointers
      for (int j = 0; j < num_columns; j++) {
        auto &chunk = chunks[i * num_columns + j];
        chunk.start_row = stripe_start_row;
        chunk.num_rows = stripe_info->numberOfRows;
        chunk.encoding_kind = stripe_footer.columns[selected_cols[j]].kind;
        chunk.type_kind = md.ff.types[selected_cols[j]].kind;
        chunk.decimal_scale = md.ff.types[selected_cols[j]].scale;
        for (int k = 0; k < orc::gpu::CI_NUM_STREAMS; k++) {
          if (chunk.strm_len[k] > 0) {
            chunk.streams[k] = d_data + stream_info[chunk.strm_id[k]].dst_pos;
          }
        }
      }
      stripe_start_row += stripe_info->numberOfRows;
    }

    // Setup table for converting timestamp columns from local to UTC time
    std::vector<int64_t> tz_table;
    if (has_time_stamp_column && !selected_stripes.empty()) {
      CUDF_EXPECTS(BuildTimezoneTransitionTable(
                       tz_table, selected_stripes[0].second.writerTimezone),
                   "Cannot setup timezone LUT");
    }

    if (md.ps.compression != orc::NONE) {
      auto decomp_data =
          decompress_stripe_data(chunks, stripe_data, md.decompressor.get(),
                                 stream_info, selected_stripes.size());
      stripe_data.clear();
      stripe_data.push_back(std::move(decomp_data));
    }

    for (auto &column : columns) {
      CUDF_EXPECTS(column.allocate() == GDF_SUCCESS, "Cannot allocate columns");
    }
    decode_stream_data(chunks, num_dict_entries, skip_rows, tz_table, columns);
  } else {
    // Columns' data's memory is still expected for an empty dataframe
    for (auto &column : columns) {
      CUDF_EXPECTS(column.allocate() == GDF_SUCCESS, "Cannot allocate columns");
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
  args->data = (gdf_column **)malloc(sizeof(gdf_column *) * num_columns);
  for (int i = 0; i < num_columns; ++i) {
    args->data[i] = columns[i].release();
  }
  args->num_cols_out = num_columns;
  args->num_rows_out = num_rows;

  return GDF_SUCCESS;
}
