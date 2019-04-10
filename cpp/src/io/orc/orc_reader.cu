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

#include "cudf.h"
#include "io/comp/gpuinflate.h"
#include "io/utilities/wrapper_utils.hpp"
#include "utilities/cudf_utils.h"
#include "utilities/error_utils.hpp"

#include <cuda_runtime.h>
#include <nvstrings/NVStrings.h>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>

#include <array>
#include <cstring>
#include <iostream>
#include <numeric>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#if 1
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
      return std::make_pair(GDF_TIMESTAMP, gdf_dtype_extra_info{TIME_UNIT_ns});
    case orc::DATE:
      return std::make_pair(GDF_DATE32, gdf_dtype_extra_info{TIME_UNIT_NONE});
    case orc::DECIMAL:
      // Currently unhandled as there isn't an explicit mapping
    default:
      break;
  }

  return std::make_pair(GDF_invalid, gdf_dtype_extra_info{TIME_UNIT_NONE});
}

/**
 * @brief A helper class for ORC file metadata. Provides some additional
 * convenience methods for initializing and accessing metadata.
 **/
class OrcMetadata {
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
    LOG_PRINTF(" version(%zd) = {%d,%d}\n", ps.version.size(), (ps.version.size() > 0) ? (int32_t)ps.version[0] : -1, (ps.version.size() > 1) ? (int32_t)ps.version[1] : -1);
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
        printf("}\n");
      }
      if (ff.types[i].fieldNames.size() > 0) {
        printf("   fieldNames = ");
        for (int j = 0; j < (int)ff.types[i].fieldNames.size(); j++) {
          printf("%c\"%s\"", (j) ? ',' : '{',
                 ff.types[i].fieldNames[j].c_str());
        }
        printf("}\n");
      }
    }
    for (size_t i = 0; i < ff.metadata.size(); i++) {
      printf(" metadata: \"%s\" = \"%s\"\n", ff.metadata[i].name.c_str(),
             ff.metadata[i].value.c_str());
    }
    printf(" numberOfRows = %zd\n", (size_t)ff.numberOfRows);
    printf(" rowIndexStride = %d\n", ff.rowIndexStride);
  }

  void select_stripes(uint64_t min_row, uint64_t num_rows) {
    // Exclude non-needed stripes
    while (ff.stripes.size() > 0 && ff.stripes[0].numberOfRows <= min_row) {
      ff.numberOfRows -= ff.stripes[0].numberOfRows;
      min_row -= ff.stripes[0].numberOfRows;
      ff.stripes.erase(ff.stripes.begin());
    }
    num_rows = std::min(num_rows,
                        ff.numberOfRows - std::min(min_row, ff.numberOfRows));
    if (ff.numberOfRows > num_rows) {
      uint64_t row = 0;
      for (size_t i = 0; i < ff.stripes.size(); i++) {
        if (row >= num_rows) {
          ff.stripes.resize(i);
          ff.numberOfRows = row;
          break;
        }
        row += ff.stripes[i].numberOfRows;
      }
    }

    // Read stripefooter metadata
    sf.resize(ff.stripes.size());
    for (size_t i = 0; i < ff.stripes.size(); ++i)
    {
      const auto stripe = ff.stripes[i];
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
      CUDF_EXPECTS(pb.read(&sf[i], sf_length), "Cannot read stripefooter");

#if VERBOSE_OUTPUT
        printf("StripeFooter(%d/%zd):\n", 1+i, ff.stripes.size());
        printf(" %d streams:\n", (int)sf.streams.size());
        for (int j = 0; j < (int)sf.streams.size(); j++)
        {
            printf(" [%d] column=%d, kind=%d, len=%zd\n", j, sf.streams[j].column, sf.streams[j].kind, (size_t)sf.streams[j].length);
        }
        printf(" %d columns:\n", (int)sf.columns.size());
        for (int j = 0; j < (int)sf.columns.size(); j++)
        {
            printf(" [%d] kind=%d, dictionarySize=%d\n", j, sf.columns[j].kind, sf.columns[j].dictionarySize);
        }
#endif
    }
  }

  inline int get_total_rows() const { return ff.numberOfRows; }
  inline int get_num_rowgroups() const { return ff.stripes.size(); }
  inline int get_num_columns() const { return ff.types.size(); }

public:
  orc::PostScript ps;
  orc::FileFooter ff;
  std::vector<orc::StripeFooter> sf;
  std::unique_ptr<orc::OrcDecompressor> decompressor;

 private:
  const uint8_t *const data;
  const size_t len;
};

/**
 * @brief Struct that maps ORC streams to columns
 **/
struct OrcStrmInfo
{
    uint64_t offset;        // offset in file
    size_t dst_pos;         // offset in memory relative to the beginning of the compressed stripe data
    uint32_t length;        // length in file
    uint32_t gdf_idx;       // gdf column index
    uint32_t stripe_idx;    // stripe index
};

/**
 * @brief Reads Apache ORC data and returns an array of gdf_columns.
 *
 * @param[in,out] args Structure containing input and output args
 *
 * @return gdf_error GDF_SUCCESS if successful, otherwise an error code.
 **/
gdf_error read_orc(orc_read_arg *args) {

  int num_columns = 0;
  int num_rows = 0;

  DataSource input(args->source);

  OrcMetadata md(input.data(), input.size());
  CUDF_EXPECTS(md.get_num_columns() > 0, "No columns found");

  static_assert(sizeof(orc::gpu::CompressedStreamInfo) <= 256 &&
                    !(sizeof(orc::gpu::CompressedStreamInfo) & 7),
                "Unexpected sizeof(CompressedStreamInfo)");
  static_assert(sizeof(orc::gpu::ColumnDesc) <= 256 &&
                    !(sizeof(orc::gpu::ColumnDesc) & 7),
                "Unexpected sizeof(ColumnDesc)");

  // Select only rowgroups required
  md.select_stripes(0, 0x7fffffff);

  // Select only columns required (if it exists), otherwise select all
  std::vector<int32_t> gdf2orc;                           // Map gdf columns to orc columns
  std::vector<int32_t> orc2gdf(md.get_num_columns(), -1); // Map orc columns to gdf columns
  if (args->use_cols) {
    std::vector<std::string> use_names(args->use_cols,
                                       args->use_cols + args->use_cols_len);
    int index = 0;
    for (const auto &use_name : use_names) {
      for (int i = 0; i < md.get_num_columns(); ++i, ++index) {
        if (index >= md.get_num_columns()) {
          index = 0;
        }
        if (md.ff.GetColumnName(index) == use_name) {
          orc2gdf[index] = gdf2orc.size();
          gdf2orc.emplace_back(index);
          index++;
        }
      }
    }
  } else {
    // For now, only select all leaf nodes
    for (int i = 0; i < md.get_num_columns(); ++i) {
      if (md.ff.types[i].subtypes.size() == 0) {
        orc2gdf[i] = gdf2orc.size();
        gdf2orc.emplace_back(i);
      }
    }
  }

  // Initialize gdf_columns, but hold off on allocating storage space
  std::vector<gdf_column_wrapper> columns;
  LOG_PRINTF("[+] Selected columns: %d\n", num_columns);
  for (const auto &col : gdf2orc) {
    auto dtype_info = to_dtype(md.ff.types[col]);

    columns.emplace_back(static_cast<gdf_size_type>(md.ff.numberOfRows),
                         dtype_info.first, dtype_info.second,
                         md.ff.GetColumnName(col));

    LOG_PRINTF(" %2zd: name=%s size=%zd type=%d data=%lx valid=%lx\n",
               columns.size() - 1, columns.back()->col_name,
               (size_t)columns.back()->size, columns.back()->dtype,
               (uint64_t)columns.back()->data, (uint64_t)columns.back()->valid);
  }

  num_rows = md.get_total_rows();
  num_columns = (int)gdf2orc.size();

  // Select columns
  size_t total_compressed_size;
  int num_streams;
  std::vector<OrcStrmInfo> stream_info;
  std::vector<uint8_t *> compressed_stripe_data, uncompressed_stripe_data;
  orc::gpu::DictionaryEntry *global_dictionary = nullptr;
  size_t stripe_start_row;
  uint32_t num_dictionary_entries;

  if (num_rows > 0 && num_columns > 0) {
    // Allocate row index: essentially 2D array indexed by stripe id & gdf column index
    const auto num_column_chunks = md.ff.stripes.size() * num_columns;
    hostdevice_vector<orc::gpu::ColumnDesc> chunks(num_column_chunks);

    // Read stripe footers
    total_compressed_size = 0;
    stripe_start_row = 0;
    num_dictionary_entries = 0;
    for (int i = 0; i < (int)md.sf.size(); i++)
    {
        size_t strm_count;
        uint64_t src_offset, dst_offset, index_length;
        uint8_t *data_dev = nullptr;

        // Read stream data
        src_offset = 0;
        dst_offset = 0;
        index_length = md.ff.stripes[i].indexLength;
        strm_count = stream_info.size();
        for (int j = 0; j < (int)md.sf[i].streams.size(); j++)
        {
            uint32_t strm_length = (uint32_t)md.sf[i].streams[j].length;
            uint32_t column_id = md.sf[i].streams[j].column;
            int32_t gdf_idx = -1;
            if (column_id < orc2gdf.size())
            {
                gdf_idx = orc2gdf[column_id];
                if (gdf_idx < 0 && md.ff.types[column_id].subtypes.size() != 0)
                {
                    // This column may be a parent column, in which case the PRESENT stream may be needed
                    bool needed = (md.ff.types[column_id].kind == orc::STRUCT && md.sf[i].streams[j].kind == orc::PRESENT);
                    if (needed)
                    {
                        for (int k = 0; k < (int)md.ff.types[column_id].subtypes.size(); k++)
                        {
                            uint32_t idx = md.ff.types[column_id].subtypes[k];
                            int32_t child_idx = (idx < orc2gdf.size()) ? orc2gdf[idx] : -1;
                            if (child_idx >= 0)
                            {
                                gdf_idx = child_idx;
                                chunks[i * num_columns + gdf_idx].strm_id[orc::gpu::CI_PRESENT] = (uint32_t)stream_info.size();
                                chunks[i * num_columns + gdf_idx].strm_len[orc::gpu::CI_PRESENT] = strm_length;
                            }
                        }
                    }
                }
            }
            if (src_offset >= index_length && gdf_idx >= 0)
            {
                int ci_kind = orc::gpu::CI_NUM_STREAMS;
                switch (md.sf[i].streams[j].kind)
                {
                case orc::DATA:
                    ci_kind = orc::gpu::CI_DATA;
                    break;
                case orc::LENGTH:
                case orc::SECONDARY:
                    ci_kind = orc::gpu::CI_DATA2;
                    break;
                case orc::DICTIONARY_DATA:
                    ci_kind = orc::gpu::CI_DICTIONARY;
                    chunks[i * num_columns + gdf_idx].dictionary_start = num_dictionary_entries;
                    chunks[i * num_columns + gdf_idx].dict_len = md.sf[i].columns[column_id].dictionarySize;
                    num_dictionary_entries += md.sf[i].columns[column_id].dictionarySize;
                    break;
                case orc::PRESENT:
                    ci_kind = orc::gpu::CI_PRESENT;
                    break;
                default:
                    // TBD: Could skip loading this stream
                    break;
                }
                if (ci_kind < orc::gpu::CI_NUM_STREAMS)
                {
                    chunks[i * num_columns + gdf_idx].strm_id[ci_kind] = (uint32_t)stream_info.size();
                    chunks[i * num_columns + gdf_idx].strm_len[ci_kind] = strm_length;
                }
            }
            if (gdf_idx >= 0)
            {
                OrcStrmInfo info;
                info.offset = md.ff.stripes[i].offset + src_offset;
                info.length = strm_length;
                info.dst_pos = dst_offset;
                info.gdf_idx = gdf_idx;
                info.stripe_idx = i;
                stream_info.push_back(info); // FIXME: Use emplace_back
                dst_offset += strm_length;
            }
            src_offset += strm_length;
        }
        if (dst_offset > 0)
        {
            RMM_ALLOC((void **)&data_dev, dst_offset, 0);
            if (!data_dev)
                goto error_exit;
            while (strm_count < stream_info.size())
            {
                // Coalesce consecutive streams into one read
                uint64_t len = stream_info[strm_count].length;
                uint64_t offset = stream_info[strm_count].offset;
                void *dst = data_dev + stream_info[strm_count].dst_pos;
                strm_count++;
                while (strm_count < stream_info.size() && stream_info[strm_count].offset == offset + len)
                {
                    len += stream_info[strm_count].length;
                    strm_count++;
                }
                cudaMemcpyAsync(dst, input.data() + offset, len, cudaMemcpyHostToDevice, 0); // TODO: datasource::gpuread
                total_compressed_size += len;
            }
            // Update stream pointers
            for (int j = 0; j < num_columns; j++)
            {
                for (int k = 0; k < orc::gpu::CI_NUM_STREAMS; k++)
                {
                    if (chunks[i * num_columns + j].strm_len[k] > 0)
                    {
                        uint32_t strm_id = chunks[i * num_columns + j].strm_id[k];
                        chunks[i * num_columns + j].streams[k] = data_dev + stream_info[strm_id].dst_pos;
                    }
                }
                chunks[i * num_columns + j].start_row = (uint32_t)stripe_start_row;
                chunks[i * num_columns + j].num_rows = md.ff.stripes[i].numberOfRows;
                chunks[i * num_columns + j].encoding_kind = md.sf[i].columns[gdf2orc[j]].kind;
                chunks[i * num_columns + j].type_kind = md.ff.types[gdf2orc[j]].kind;
            }
        }
        compressed_stripe_data.push_back(data_dev);
        stripe_start_row += md.ff.stripes[i].numberOfRows;
    }

    printf("[CPU] Read %zd bytes\n", total_compressed_size);
    // Allocate global dictionary
    if (num_dictionary_entries > 0)
    {
        RMM_ALLOC((void **)&global_dictionary, num_dictionary_entries * sizeof(orc::gpu::DictionaryEntry), 0);
    }

    // Setup decompression
    num_streams = (int)stream_info.size();
    printf(" %d data streams, %d dictionary entries\n", num_streams, num_dictionary_entries);
    if (md.ps.compression != orc::NONE)
    {
        uint32_t total_compressed_blocks;
        size_t total_uncompressed_size;
        double decompression_time = 0;

        hostdevice_vector<orc::gpu::CompressedStreamInfo> streams(0, num_streams);
        for (int i = 0; i < num_streams; i++) {
          streams.insert(orc::gpu::CompressedStreamInfo(
              compressed_stripe_data[stream_info[i].stripe_idx] +
                  stream_info[i].dst_pos,
              stream_info[i].length));
        }
        CUDA_TRY(cudaMemcpyAsync(streams.device_ptr(), streams.host_ptr(),
                                 streams.memory_size(),
                                 cudaMemcpyHostToDevice));
        CUDA_TRY(ParseCompressedStripeData(streams.device_ptr(), streams.size(),
                                           md.ps.compressionBlockSize,
                                           md.decompressor->GetLog2MaxCompressionRatio()));
        CUDA_TRY(cudaMemcpyAsync(streams.host_ptr(), streams.device_ptr(),
                                 streams.memory_size(),
                                 cudaMemcpyDeviceToHost));
        CUDA_TRY(cudaStreamSynchronize(0));

        total_compressed_blocks = 0;
        total_uncompressed_size = 0;
        for (int i = 0; i < num_streams; i++)
        {
            total_compressed_blocks += streams[i].num_compressed_blocks;
            total_uncompressed_size += streams[i].max_uncompressed_size;
        }
        if (total_uncompressed_size > 0)
        {
            uint8_t *uncompressed_data = nullptr;

            size_t uncomp_ofs;
            printf("%d compressed blocks, max_uncompressed_size=%zd\n", total_compressed_blocks, total_uncompressed_size);
            RMM_ALLOC((void **)&uncompressed_data, total_uncompressed_size, 0);
            if (!uncompressed_data)
                goto error_exit;
            uncompressed_stripe_data.push_back(uncompressed_data);

            rmm::device_vector<gpu_inflate_input_s> inflate_in(total_compressed_blocks);
            rmm::device_vector<gpu_inflate_status_s> inflate_out(total_compressed_blocks);

            uncomp_ofs = 0;
            for (int i = 0, pos = 0; i < num_streams; i++)
            {
                streams[i].uncompressed_data = uncompressed_data + uncomp_ofs;
                streams[i].decctl = inflate_in.data().get() + pos;
                streams[i].decstatus = inflate_out.data().get() + pos;
                streams[i].max_compressed_blocks = streams[i].num_compressed_blocks;
                stream_info[i].dst_pos = uncomp_ofs; // Now indicates the offset relative to base uncompressed data
                uncomp_ofs += streams[i].max_uncompressed_size;
                pos += streams[i].num_compressed_blocks;
            }

            // Parse again, this time populating the decompression input/output buffers
            CUDA_TRY(cudaMemcpyAsync(streams.device_ptr(), streams.host_ptr(),
                                     streams.memory_size(),
                                     cudaMemcpyHostToDevice));
            CUDA_TRY(ParseCompressedStripeData(
                streams.device_ptr(), streams.size(),
                md.ps.compressionBlockSize,
                md.decompressor->GetLog2MaxCompressionRatio()));
            switch (md.ps.compression) {
              case orc::ZLIB:
                CUDA_TRY(gpuinflate(inflate_in.data().get(),
                                    inflate_out.data().get(),
                                    total_compressed_blocks, 0));
                break;
              case orc::SNAPPY:
                CUDA_TRY(gpu_unsnap(inflate_in.data().get(),
                                    inflate_out.data().get(),
                                    total_compressed_blocks));
                break;
              default:
                CUDF_EXPECTS(false, "Unexpected decompression dispatch");
                break;
            }
            CUDA_TRY(PostDecompressionReassemble(streams.device_ptr(),
                                                 streams.size()));

            // Update the stream information in device memory with the updated
            // pointers to the uncompressed data buffer
            // TBD: We could update the value from the information we already
            // have in stream_info[], but using the gpu results also updates
            // max_uncompressed_size to the actual uncompressed size, or zero if
            // decompression failed.
            CUDA_TRY(cudaMemcpyAsync(streams.host_ptr(), streams.device_ptr(),
                                     streams.memory_size(),
                                     cudaMemcpyDeviceToHost));
            CUDA_TRY(cudaStreamSynchronize(0));
            for (int i = 0; i < (int)md.ff.stripes.size(); i++)
            {
                for (int j = 0; j < num_columns; j++)
                {
                    orc::gpu::ColumnDesc *ck = &chunks[i * num_columns + j];
                    for (uint32_t k = 0; k < orc::gpu::CI_NUM_STREAMS; k++)
                    {
                        uint32_t len = ck->strm_len[k];
                        uint32_t strm_id = ck->strm_id[k];
                        if (len > 0 && strm_id < (uint32_t)num_streams)
                        {
                            ck->streams[k] = streams[strm_id].uncompressed_data;
                            ck->strm_len[k] = (uint32_t)streams[strm_id].max_uncompressed_size;
                        }
                    }
                }
            }
            printf("[GPU] Decompressed %zd bytes in %.1fms (%.2fMB/s)\n", total_uncompressed_size, decompression_time * 1000.0, 1.e-6 * total_uncompressed_size / decompression_time);

        }
        // Free compressed slice data after decompression (not needed any further)
        for (size_t i = 0; i < compressed_stripe_data.size(); i++)
        {
            RMM_FREE(compressed_stripe_data[i], 0);
            compressed_stripe_data[i] = nullptr;
        }
    }
    else
    {
        for (size_t i = 0; i < compressed_stripe_data.size(); i++)
        {
            uncompressed_stripe_data.push_back(compressed_stripe_data[i]);
            compressed_stripe_data[i] = nullptr;
        }
    }

    // Allocate column data
    for (auto &column : columns) {
      CUDF_EXPECTS(column.allocate() == GDF_SUCCESS, "Cannot allocate columns");
    }

    // Finalize column chunk initialization
    for (int i = 0; i < (int)md.ff.stripes.size(); i++) {
      for (int j = 0; j < num_columns; j++) {
        auto &chunk = chunks[i * num_columns + j];
        chunk.valid_map_base = reinterpret_cast<uint32_t *>(columns[j]->valid);
        chunk.column_data_base = columns[j]->data;
        chunk.dtype_len = (columns[j]->dtype == GDF_STRING)
                              ? sizeof(std::pair<const char *, size_t>)
                              : gdf_dtype_size(columns[j]->dtype);
      }
    }

    CUDA_TRY(cudaMemcpyAsync(chunks.device_ptr(), chunks.host_ptr(),
                             chunks.memory_size(), cudaMemcpyHostToDevice));
    CUDA_TRY(DecodeNullsAndStringDictionaries(chunks.device_ptr(),
                                              global_dictionary, num_columns,
                                              md.get_num_rowgroups(), num_rows,
                                              0));
    CUDA_TRY(DecodeOrcColumnData(chunks.device_ptr(), global_dictionary,
                                 num_columns, md.get_num_rowgroups(), num_rows,
                                 0));
    CUDA_TRY(cudaMemcpyAsync(chunks.host_ptr(), chunks.device_ptr(),
                             chunks.memory_size(), cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaStreamSynchronize(0));

    printf("[GPU] Decoded bytes\n");
    for (int i = 0; i < num_columns; i++) {
      for (int j = 0; j < (int)md.ff.stripes.size(); j++) {
        columns[i]->null_count += chunks[j * num_columns + i].null_count;
      }
      LOG_PRINTF(
          "columns[%d].null_count = %d/%d (start_row=%d, nrows=%d, "
          "strm_len=%d)\n",
          i, columns[i]->null_count, columns[i]->size, chunks[i].start_row,
          chunks[i].num_rows, chunks[i].strm_len[orc::gpu::CI_PRESENT]);
    }
  } else {
    // Columns' data's memory is still expected for an empty dataframe
    for (auto &column : columns) {
      CUDF_EXPECTS(column.allocate() == GDF_SUCCESS, "Cannot allocate columns");
    }
  }

  // Transfer ownership to raw pointer output arguments
  args->data = (gdf_column **)malloc(sizeof(gdf_column *) * num_columns);
  for (int i = 0; i < num_columns; ++i) {
    args->data[i] = columns[i].release();

    // For string dtype, allocate and return an NvStrings container instance,
    // deallocating the original string list memory in the process.
    // This container takes a list of string pointers and lengths, and copies
    // into its own memory so the source memory must not be released yet.
    if (args->data[i]->dtype == GDF_STRING) {
      using str_pair = std::pair<const char *, size_t>;

      auto str_list = static_cast<str_pair *>(args->data[i]->data);
      auto str_data = NVStrings::create_from_index(str_list, num_rows);
      RMM_FREE(std::exchange(args->data[i]->data, str_data), 0);
    }
  }
  args->num_cols_out = num_columns;
  args->num_rows_out = num_rows;
  args->index_col = nullptr;

error_exit:
  for (size_t i = 0; i < compressed_stripe_data.size(); i++)
  {
      RMM_FREE(compressed_stripe_data[i], 0);
      compressed_stripe_data[i] = nullptr;
  }
  for (size_t i = 0; i < uncompressed_stripe_data.size(); i++)
  {
      RMM_FREE(uncompressed_stripe_data[i], 0);
      uncompressed_stripe_data[i] = nullptr;
  }
  RMM_FREE(global_dictionary, 0);

  return GDF_SUCCESS;
}
