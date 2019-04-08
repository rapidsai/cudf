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

size_t GetGDFTypeLength(gdf_dtype dtype)
{
    size_t dtype_len = 0;
    switch (dtype)
    {
    case GDF_INT8:
        dtype_len = 1;
        break;
    case GDF_INT16:
        dtype_len = 2;
        break;
    case GDF_INT32:
    case GDF_FLOAT32:
    case GDF_DATE32:
    case GDF_CATEGORY: // NOTE: Category type converts the underlying string type into a 32-bit hash
        dtype_len = 4;
        break;
    case GDF_INT64:
    case GDF_FLOAT64:
    case GDF_DATE64:
    case GDF_TIMESTAMP:
        dtype_len = 8;
        break;
    // NOTE: String returns the size of the std::pair needed to create the nvStrings array
    case GDF_STRING:
        dtype_len = sizeof(std::pair<char *,size_t>); // For now, just the index
        break;
    default:
        return 0;
    }
    return dtype_len;
}

/**
 * @brief A helper class that wraps a gdf_column and any associated memory.
 *
 * This abstraction initializes and manages a gdf_column (fields and memory)
 * while still allowing direct access. Memory is automatically deallocated
 * unless ownership is transferred via releasing and assigning the raw pointer.
 **/
class gdf_column_wrapper {
 public:
  gdf_column_wrapper(gdf_size_type size, gdf_dtype dtype,
                     gdf_dtype_extra_info dtype_info, const std::string name) {
    col = (gdf_column *)malloc(sizeof(gdf_column));
    col->col_name = (char *)malloc(name.length() + 1);
    strcpy(col->col_name, name.c_str());
    gdf_column_view_augmented(col, nullptr, nullptr, size, dtype, 0, dtype_info);
  }

  ~gdf_column_wrapper() {
    if (col) {
      RMM_FREE(col->data, 0);
      RMM_FREE(col->valid, 0);
      free(col->col_name);
    }
    free(col);
  };

  gdf_column_wrapper(const gdf_column_wrapper &other) = delete;
  gdf_column_wrapper(gdf_column_wrapper &&other) : col(other.col) {
    other.col = nullptr;
  }

  gdf_error allocate() {
    // For strings, just store the ptr + length. Eventually, column's data ptr
    // is replaced with an NvString instance created from these pairs.
    const auto num_rows = std::max(col->size, 1);
    const auto column_byte_width = (col->dtype == GDF_STRING)
                                       ? sizeof(std::pair<const char*, int>)
                                       : gdf_dtype_size(col->dtype);

    RMM_TRY(RMM_ALLOC(&col->data, num_rows * column_byte_width, 0));
    RMM_TRY(RMM_ALLOC(&col->valid, gdf_valid_allocation_size(num_rows), 0));
    CUDA_TRY(cudaMemset(col->valid, 0, gdf_valid_allocation_size(num_rows)));

    return GDF_SUCCESS;
  }

  gdf_column *operator->() const { return col; }
  gdf_column *get() const { return col; }
  gdf_column *release() {
    auto temp = col;
    col = nullptr;
    return temp;
  }

 private:
  gdf_column *col = nullptr;
};

/**
 * @brief Function that translates Parquet datatype to GDF dtype
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

// Map orc streams to columns
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

  std::vector<gdf_column_wrapper> columns;
  //int num_columns = 0;
  //int num_rows = 0;
  int index_col = -1;

  DataSource input(args->source);

  // Select columns
  int postscript_length = input.data()[input.size() - 1];
  const uint8_t *postscript = &input.data()[input.size() - postscript_length - 1];
  orc::PostScript ps;
  orc::FileFooter ff;
  orc::ProtobufReader pb;
  const uint8_t *uncompressed_footer;
  size_t footer_length, total_compressed_size;

  int num_streams, num_columns;
  orc::gpu::CompressedStreamInfo *strm_desc = nullptr, *strm_desc_dev = nullptr;
  std::vector<OrcStrmInfo> stream_info;
  std::vector<uint8_t *> compressed_stripe_data, uncompressed_stripe_data;
  std::vector<int32_t> gdf2orc;       // Map gdf columns to orc columns
  std::vector<int32_t> orc2gdf;       // Map orc columns to gdf columns
  orc::gpu::ColumnDesc *chunks = nullptr, *chunks_dev = nullptr;
  orc::gpu::DictionaryEntry *global_dictionary = nullptr;
  size_t stripe_start_row;
  uint32_t num_dictionary_entries;
  uint64_t first_row = 0;
  uint64_t num_rows = 0x7fffffff;

  static_assert(sizeof(orc::gpu::CompressedStreamInfo) <= 256 && !(sizeof(orc::gpu::CompressedStreamInfo) & 7), "Unexpected sizeof(CompressedStreamInfo)");
  static_assert(sizeof(orc::gpu::ColumnDesc) <= 256 && !(sizeof(orc::gpu::ColumnDesc) & 7), "Unexpected sizeof(ColumnDesc)");
  printf("postscript length = %d\n", postscript_length);
  pb.init(postscript, postscript_length);
  CUDF_EXPECTS(pb.read(&ps, postscript_length),
              "Failed to read postscript metadata");
  printf("PostScript:\n");
  printf(" footerLength = %zd\n", (size_t)ps.footerLength);
  printf(" compression = %d\n", ps.compression);
  printf(" compressionBlockSize = %d\n", ps.compressionBlockSize);
  printf(" version(%zd) = {%d,%d}\n", ps.version.size(), (ps.version.size() > 0) ? (int32_t)ps.version[0] : -1, (ps.version.size() > 1) ? (int32_t)ps.version[1] : -1);
  printf(" metadataLength = %zd\n", (size_t)ps.metadataLength);
  printf(" magic = \"%s\"\n", ps.magic.c_str());
  CUDF_EXPECTS(ps.footerLength + postscript_length < input.size(), "Invalid footer length");

  orc::OrcDecompressor decompressor(ps.compression, ps.compressionBlockSize);
  uncompressed_footer = decompressor.Decompress(postscript - ps.footerLength, ps.footerLength, &footer_length);
  CUDF_EXPECTS(uncompressed_footer, "Failed to uncompress file footer");
  pb.init(uncompressed_footer, footer_length);
  CUDF_EXPECTS(pb.read(&ff, footer_length), "Failed to read file footer");
  printf("FileFooter:\n");
  printf(" headerLength = %zd\n", (size_t)ff.headerLength);
  printf(" contentLength = %zd\n", (size_t)ff.contentLength);
  for (int i = 0; i < (int)ff.stripes.size(); i++)
  {
      printf(" stripe #%d @ %zd: %d rows, index+data+footer: %zd+%zd+%d bytes\n", i, (size_t)ff.stripes[i].offset, ff.stripes[i].numberOfRows, (size_t)ff.stripes[i].indexLength, (size_t)ff.stripes[i].dataLength, ff.stripes[i].footerLength);
  }
  for (int i = 0; i < (int)ff.types.size(); i++)
  {
      printf(" column %d: kind=%d, parent=%d\n", i, ff.types[i].kind, ff.types[i].parent_idx);
      if (ff.types[i].subtypes.size() > 0)
      {
          printf("   subtypes = ");
          for (int j = 0; j < (int)ff.types[i].subtypes.size(); j++)
          {
              printf("%c%d", (j) ? ',' : '{', ff.types[i].subtypes[j]);
          }
          printf("}\n");
      }
      if (ff.types[i].fieldNames.size() > 0)
      {
          printf("   fieldNames = ");
          for (int j = 0; j < (int)ff.types[i].fieldNames.size(); j++)
          {
              printf("%c\"%s\"", (j) ? ',' : '{', ff.types[i].fieldNames[j].c_str());
          }
          printf("}\n");
      }
  }
  for (int i = 0; i < (int)ff.metadata.size(); i++)
  {
      printf(" metadata: \"%s\" = \"%s\"\n", ff.metadata[i].name.c_str(), ff.metadata[i].value.c_str());
  }
  printf(" numberOfRows = %zd\n", (size_t)ff.numberOfRows);
  printf(" rowIndexStride = %d\n", ff.rowIndexStride);
  // Modify the footer to exclude non-needed stripes
  while (ff.stripes.size() > 0 && ff.stripes[0].numberOfRows <= first_row)
  {
      ff.numberOfRows -= ff.stripes[0].numberOfRows;
      first_row -= ff.stripes[0].numberOfRows;
      ff.stripes.erase(ff.stripes.begin());
  }
  num_rows = std::min(num_rows, ff.numberOfRows - std::min(first_row, ff.numberOfRows));
  if (ff.numberOfRows > num_rows)
  {
      uint64_t row = 0;
      for (size_t i = 0; i < ff.stripes.size(); i++)
      {
          if (row >= num_rows)
          {
              ff.stripes.resize(i);
              ff.numberOfRows = row;
              break;
          }
          row += ff.stripes[i].numberOfRows;
      }
  }
  // Select columns
  orc2gdf.resize(ff.types.size(), -1);
  if (args->use_cols_len > 0)
  {
      // Find columns by name
      gdf2orc.resize(args->use_cols_len);
      for (int i = 0, column_id = 0; i < args->use_cols_len; i++)
      {
          int num_orc_columns = (int)ff.types.size();
          gdf2orc[i] = -1;
          for (int j = 0; j < num_orc_columns; j++, column_id++)
          {
              if (column_id >= num_orc_columns)
              {
                  column_id = 0;
              }
              if (ff.GetColumnName(column_id) == args->use_cols[i])
              {
                  gdf2orc[i] = column_id;
                  orc2gdf[column_id] = i;
                  column_id++;
                  break;
              }
          }
          if (gdf2orc[i] < 0)
          {
              printf("Column not found: \"%s\"\n", args->use_cols[i]);
              return GDF_FILE_ERROR;
          }
      }
  }
  else
  {
      // Select all columns
      for (int i = 0; i < (int)ff.types.size(); i++)
      {
          bool col_en = (ff.types[i].subtypes.size() == 0); // For now, select all leaf nodes in the schema
          if (col_en)
          {
              int32_t gdf_idx = (int32_t)gdf2orc.size();
              gdf2orc.resize(gdf_idx + 1);
              gdf2orc[gdf_idx] = i;
              orc2gdf[i] = gdf_idx;
          }
      }
  }
  // Allocate gdf columns
  num_columns = (int)gdf2orc.size();
  for (int i = 0; i < num_columns; i++)
  {
    auto dtype_info = to_dtype(ff.types[gdf2orc[i]]);

    columns.emplace_back(static_cast<gdf_size_type>(ff.numberOfRows),
                         dtype_info.first, dtype_info.second,
                         ff.GetColumnName(gdf2orc[i]));

    LOG_PRINTF(" %2zd: name=%s size=%zd type=%d data=%lx valid=%lx\n",
               columns.size() - 1, columns.back()->col_name,
               (size_t)columns.back()->size, columns.back()->dtype,
               (uint64_t)columns.back()->data, (uint64_t)columns.back()->valid);
  }

  if (num_rows > 0 && num_columns > 0) {
    // Allocate row index: essentially 2D array indexed by stripe id & gdf column index
    cudaMallocHost((void **)&chunks, ff.stripes.size() * num_columns * sizeof(chunks[0]));
    RMM_ALLOC((void **)&chunks_dev, ff.stripes.size() * num_columns * sizeof(chunks_dev[0]), 0);
    if (!(chunks && chunks_dev))
        goto error_exit;
    memset(chunks, 0, ff.stripes.size() * num_columns * sizeof(chunks[0]));
    // Read stripe footers
    total_compressed_size = 0;
    stripe_start_row = 0;
    num_dictionary_entries = 0;
    for (int i = 0; i < (int)ff.stripes.size(); i++)
    {
        size_t sfooter_offset = ff.stripes[i].offset + ff.stripes[i].indexLength + ff.stripes[i].dataLength;
        size_t sfooter_length = ff.stripes[i].footerLength;
        orc::StripeFooter sf;
        const uint8_t *uncomp;
        size_t uncomp_len = 0, strm_count;
        uint64_t src_offset, dst_offset, index_length;
        uint8_t *data_dev = nullptr;

        if (sfooter_offset + sfooter_length >= input.size())
        {
            printf("Invalid stripe information\n");
            return GDF_CUDA_ERROR;
        }
        uncomp = decompressor.Decompress(input.data() + sfooter_offset, sfooter_length, &uncomp_len);
        pb.init(uncomp, uncomp_len);
        pb.read(&sf, uncomp_len);
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
        // Read stream data
        src_offset = 0;
        dst_offset = 0;
        index_length = ff.stripes[i].indexLength;
        strm_count = stream_info.size();
        for (int j = 0; j < (int)sf.streams.size(); j++)
        {
            uint32_t strm_length = (uint32_t)sf.streams[j].length;
            uint32_t column_id = sf.streams[j].column;
            int32_t gdf_idx = -1;
            if (column_id < orc2gdf.size())
            {
                gdf_idx = orc2gdf[column_id];
                if (gdf_idx < 0 && ff.types[column_id].subtypes.size() != 0)
                {
                    // This column may be a parent column, in which case the PRESENT stream may be needed
                    bool needed = (ff.types[column_id].kind == orc::STRUCT && sf.streams[j].kind == orc::PRESENT);
                    if (needed)
                    {
                        for (int k = 0; k < (int)ff.types[column_id].subtypes.size(); k++)
                        {
                            uint32_t idx = ff.types[column_id].subtypes[k];
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
                switch (sf.streams[j].kind)
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
                    chunks[i * num_columns + gdf_idx].dict_len = sf.columns[column_id].dictionarySize;
                    num_dictionary_entries += sf.columns[column_id].dictionarySize;
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
                info.offset = ff.stripes[i].offset + src_offset;
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
                chunks[i * num_columns + j].num_rows = ff.stripes[i].numberOfRows;
                chunks[i * num_columns + j].encoding_kind = sf.columns[gdf2orc[j]].kind;
                chunks[i * num_columns + j].type_kind = ff.types[gdf2orc[j]].kind;
            }
        }
        compressed_stripe_data.push_back(data_dev);
        stripe_start_row += ff.stripes[i].numberOfRows;
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
    if (ps.compression != orc::NONE)
    {
        uint32_t total_compressed_blocks;
        size_t total_uncompressed_size;
        double decompression_time = 0;
        cudaMallocHost((void **)&strm_desc, num_streams * sizeof(orc::gpu::CompressedStreamInfo));
        RMM_ALLOC((void **)&strm_desc_dev, num_streams * sizeof(orc::gpu::CompressedStreamInfo), 0);
        if (!(strm_desc && strm_desc_dev))
            goto error_exit;
        for (int i = 0; i < num_streams; i++)
        {
            strm_desc[i].compressed_data = compressed_stripe_data[stream_info[i].stripe_idx] + stream_info[i].dst_pos;
            strm_desc[i].uncompressed_data = nullptr;
            strm_desc[i].compressed_data_size = stream_info[i].length;
            strm_desc[i].decctl = nullptr;
            strm_desc[i].max_compressed_blocks = 0;
            strm_desc[i].num_compressed_blocks = 0;
            strm_desc[i].max_uncompressed_size = 0;
        }
        cudaMemcpyAsync(strm_desc_dev, strm_desc, num_streams * sizeof(orc::gpu::CompressedStreamInfo), cudaMemcpyHostToDevice);
        ParseCompressedStripeData(strm_desc_dev, num_streams, ps.compressionBlockSize, decompressor.GetLog2MaxCompressionRatio());
        cudaMemcpyAsync(strm_desc, strm_desc_dev, num_streams * sizeof(orc::gpu::CompressedStreamInfo), cudaMemcpyDeviceToHost);
        cudaStreamSynchronize(0);
        total_compressed_blocks = 0;
        total_uncompressed_size = 0;
        for (int i = 0; i < num_streams; i++)
        {
            total_compressed_blocks += strm_desc[i].num_compressed_blocks;
            total_uncompressed_size += strm_desc[i].max_uncompressed_size;
        }
        if (total_uncompressed_size > 0)
        {
            uint8_t *uncompressed_data = nullptr;
            gpu_inflate_input_s *inflate_in = nullptr;
            gpu_inflate_status_s *inflate_out = nullptr;
            size_t uncomp_ofs;
            printf("%d compressed blocks, max_uncompressed_size=%zd\n", total_compressed_blocks, total_uncompressed_size);
            RMM_ALLOC((void **)&uncompressed_data, total_uncompressed_size, 0);
            if (!uncompressed_data)
                goto error_exit;
            uncompressed_stripe_data.push_back(uncompressed_data);
            if (total_compressed_blocks > 0)
            {
                RMM_ALLOC((void **)&inflate_in, total_compressed_blocks * (sizeof(gpu_inflate_input_s) + sizeof(gpu_inflate_status_s)), 0);
                if (!inflate_in)
                    goto error_exit;
                inflate_out = reinterpret_cast<gpu_inflate_status_s *>(inflate_in + total_compressed_blocks);
                cudaMemsetAsync(inflate_out, 0, total_compressed_blocks * sizeof(gpu_inflate_status_s));
            }
            uncomp_ofs = 0;
            for (int i = 0, pos = 0; i < num_streams; i++)
            {
                strm_desc[i].uncompressed_data = uncompressed_data + uncomp_ofs;
                strm_desc[i].decctl = inflate_in + pos;
                strm_desc[i].decstatus = inflate_out + pos;
                strm_desc[i].max_compressed_blocks = strm_desc[i].num_compressed_blocks;
                stream_info[i].dst_pos = uncomp_ofs; // Now indicates the offset relative to base uncompressed data
                uncomp_ofs += strm_desc[i].max_uncompressed_size;
                pos += strm_desc[i].num_compressed_blocks;
            }
            // Parse again, this time populating the decompression input/output buffers
            cudaMemcpyAsync(strm_desc_dev, strm_desc, num_streams * sizeof(orc::gpu::CompressedStreamInfo), cudaMemcpyHostToDevice);
            ParseCompressedStripeData(strm_desc_dev, num_streams, ps.compressionBlockSize, decompressor.GetLog2MaxCompressionRatio());
            switch (ps.compression)
            {
            case orc::ZLIB:
                gpuinflate(inflate_in, inflate_out, total_compressed_blocks, 0);
                break;
            case orc::SNAPPY:
                gpu_unsnap(inflate_in, inflate_out, total_compressed_blocks);
                break;
            default:
                printf("Unsupported GPU compression\n");
                goto error_exit;
            }
            PostDecompressionReassemble(strm_desc_dev, num_streams);
            // Update pointers to uncompressed data
            // TBD: We could update the value from the information we already have in stream_info[], but using the gpu results also updates max_uncompressed_size
            // to the actual uncompressed size, or zero if decompression failed.
            cudaMemcpyAsync(strm_desc, strm_desc_dev, num_streams * sizeof(orc::gpu::CompressedStreamInfo), cudaMemcpyDeviceToHost);
            cudaStreamSynchronize(0);
            for (int i = 0; i < (int)ff.stripes.size(); i++)
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
                            ck->streams[k] = strm_desc[strm_id].uncompressed_data;
                            ck->strm_len[k] = (uint32_t)strm_desc[strm_id].max_uncompressed_size;
                        }
                    }
                }
            }
            printf("[GPU] Decompressed %zd bytes in %.1fms (%.2fMB/s)\n", total_uncompressed_size, decompression_time * 1000.0, 1.e-6 * total_uncompressed_size / decompression_time);
            RMM_FREE(inflate_in, 0);
        }
        // Free compressed slice data after decompression (not needed any further)
        for (size_t i = 0; i < compressed_stripe_data.size(); i++)
        {
            RMM_FREE(compressed_stripe_data[i], 0);
            compressed_stripe_data[i] = nullptr;
        }
        cudaFreeHost(strm_desc);
        RMM_FREE(strm_desc_dev, 0);
        strm_desc = strm_desc_dev = nullptr;
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
    for (int i = 0; i < (int)ff.stripes.size(); i++)
    {
        for (int j = 0; j < num_columns; j++)
        {
            orc::gpu::ColumnDesc *ck = &chunks[i * num_columns + j];
            ck->valid_map_base = reinterpret_cast<uint32_t *>(columns[j]->valid);
            ck->column_data_base = columns[j]->data;
            ck->dtype_len = (uint8_t)GetGDFTypeLength(columns[j]->dtype);
        }
    }

    // Copy column chunk data to device
    cudaMemcpyAsync(chunks_dev, chunks, num_columns * ff.stripes.size() * sizeof(orc::gpu::ColumnDesc), cudaMemcpyHostToDevice);
    DecodeNullsAndStringDictionaries(chunks_dev, global_dictionary, num_columns, (uint32_t)ff.stripes.size(), num_rows, first_row);
    DecodeOrcColumnData(chunks_dev, global_dictionary, num_columns, (uint32_t)ff.stripes.size(), num_rows, first_row);
    cudaMemcpyAsync(chunks, chunks_dev, num_columns * ff.stripes.size() * sizeof(orc::gpu::ColumnDesc), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(0);
    printf("[GPU] Decoded bytes\n");
#if 0
    if (num_dictionary_entries > 0)
    {
        orc::gpu::DictionaryEntry *host_dictionary = nullptr;
        RMM_ALLOC_HOST((void **)&host_dictionary, num_dictionary_entries * sizeof(orc::gpu::DictionaryEntry));
        if (!host_dictionary)
            goto error_exit;
        cudaMemcpyAsync(host_dictionary, global_dictionary, num_dictionary_entries * sizeof(orc::gpu::DictionaryEntry), cudaMemcpyDeviceToHost);
        cudaStreamSynchronize(0);
        printf("global dictionary:\n");
        for (uint32_t i = 0; i < 1000; i++)
        {
            if (i < num_dictionary_entries /*&& (i < 50 || !host_dictionary[i].len)*/)
            {
                printf("[%d] %d bytes @ %d\n", i, host_dictionary[i].len, host_dictionary[i].pos);
            }
        }
        RMM_FREE_HOST(host_dictionary);
    }
#endif
    for (int i = 0; i < num_columns; i++)
    {
        gdf_size_type null_count = 0;
        for (int j = 0; j < (int)ff.stripes.size(); j++)
        {
            null_count += (gdf_size_type)chunks[j * num_columns + i].null_count;
        }
        columns[i]->null_count = null_count;
        printf("columns[%d].null_count = %d/%d (start_row=%d, nrows=%d, strm_len=%d)\n", i, null_count, columns[i]->size, chunks[i].start_row, chunks[i].num_rows, chunks[i].strm_len[orc::gpu::CI_PRESENT]);
    }
  } else {
    // Allocate column data
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
  if (index_col != -1) {
    args->index_col = (int *)malloc(sizeof(int));
    *args->index_col = index_col;
  } else {
    args->index_col = nullptr;
  }

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
  cudaFreeHost(chunks);
  cudaFreeHost(strm_desc);
  RMM_FREE(global_dictionary, 0);
  RMM_FREE(strm_desc_dev, 0);
  RMM_FREE(chunks_dev, 0);

  return GDF_SUCCESS;
}
