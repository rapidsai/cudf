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

#include "orc_writer_impl.hpp"

#include <cstring>

#include "orc.h"
#include "orc_gpu.h"

#include <io/utilities/wrapper_utils.hpp>

namespace {

template <typename T>
using pinned_buffer = std::unique_ptr<T, decltype(&cudaFreeHost)>;

}  // namespace

namespace cudf {
namespace io {
namespace orc {

/**
 * @brief Function that translates GDF dtype to ORC datatype
 **/
constexpr TypeKind to_orckind(gdf_dtype dtype) {
  switch (dtype) {
    case GDF_INT8:
      return BYTE;
    case GDF_INT16:
      return SHORT;
    case GDF_INT32:
      return INT;
    case GDF_INT64:
      return LONG;
    case GDF_FLOAT32:
      return FLOAT;
    case GDF_FLOAT64:
      return DOUBLE;
    case GDF_BOOL8:
      return BOOLEAN;
    case GDF_DATE32:
      return DATE;
    case GDF_DATE64:
      return TIMESTAMP;
    case GDF_TIMESTAMP:
      return TIMESTAMP;
    case GDF_CATEGORY:
      return INT;
    case GDF_STRING:
      return STRING;
    default:
      return INVALID_TYPE_KIND;
  }
}

/**
 * @brief Function that copies all chunks belonging to a stream
 **/
size_t gather_stripe_stream(uint8_t *dst, const gpu::EncChunk *chunks,
                            gpu::StreamIndexType strm_type,
                            size_t num_rowgroups, size_t num_columns) {
  size_t dst_pos = 0;
  for (size_t g = 0; g < num_rowgroups; g++) {
    const gpu::EncChunk *ck = &chunks[g * num_columns];
    uint32_t chunk_len = ck->strm_len[strm_type];
    CUDA_TRY(cudaMemcpyAsync(dst + dst_pos, ck->streams[strm_type], chunk_len,
                             cudaMemcpyDeviceToHost));
    dst_pos += chunk_len;
  }
  CUDA_TRY(cudaStreamSynchronize(0));

  return dst_pos;
}

writer::Impl::Impl(std::string filepath, writer_options const& options) {
  outfile_.open(filepath, std::ios::out | std::ios::binary | std::ios::trunc);
  CUDF_EXPECTS(outfile_.is_open(), "Cannot open output file");
}

void writer::Impl::write(const cudf::table& table) {
    auto columns = table.begin();
    const int num_columns = table.num_columns();

    std::vector<uint8_t> buf;
    ProtobufWriter pbw(&buf);
    PostScript ps;
    FileFooter ff;
    StripeFooter sf;
    size_t ps_length;
    std::vector<int32_t> stream_ids;
    std::vector<size_t> strm_offsets;
    size_t num_rowgroups, num_chunks, rleout_bfr_size;
    bool has_timestamp_column = false;
    int32_t max_stream_size;
    pinned_buffer<uint8_t> stream_io_buf{nullptr, cudaFreeHost};

    // PostScript
    ps.compression = NONE;
    ps.compressionBlockSize = 256 * 1024; // TODO: Pick smaller values if too few compression blocks
    ps.version = {0,12};
    ps.metadataLength = 0; // TODO: Write stripe statistics
    ps.magic = "ORC";
    // File Footer
    ff.headerLength = ps.magic.length();
    ff.numberOfRows = 0;
    ff.rowIndexStride = 10000;
    ff.types.resize(1 + num_columns);
    ff.types[0].kind = STRUCT;
    ff.types[0].subtypes.resize(num_columns);
    ff.types[0].fieldNames.resize(num_columns);
    stream_ids.resize(num_columns * gpu::CI_NUM_STREAMS, -1);
    sf.columns.resize(num_columns + 1);
    sf.streams.resize(num_columns + 1);
    sf.streams[0].column = 0;
    sf.streams[0].kind = ROW_INDEX;
    sf.streams[0].length = 0;
    num_chunks = 0;
    for (int i = 0; i < num_columns; i++)
    {
        ff.numberOfRows = std::max(ff.numberOfRows, (uint64_t)columns[i]->size);
        sf.streams[1+i].column = 1 + i;
        sf.streams[1+i].kind = ROW_INDEX;
        sf.streams[1+i].length = 0;
    }
    sf.columns[0].kind = DIRECT;
    sf.columns[0].dictionarySize = 0;
    for (int i = 0; i < num_columns; i++)
    {
        TypeKind kind = to_orckind(columns[i]->dtype);
        StreamKind data_kind = DATA, data2_kind = LENGTH;
        ColumnEncodingKind encoding_kind = DIRECT;
        int32_t present_stream_size = 0, data_stream_size = 0, data2_stream_size = 0;

        ff.types[1 + i].kind = kind;
        if (columns[i]->null_count != 0 || (uint64_t)columns[i]->size != ff.numberOfRows)
        {
            present_stream_size = ((ff.rowIndexStride + 7) >> 3);
            present_stream_size += (present_stream_size + 0x7f) >> 7;
        }
        switch(kind)
        {
        case BOOLEAN:
            data_stream_size = ((ff.rowIndexStride + 0x3ff) >> 10) * (128 + 1);
            break;
        case BYTE:
            data_stream_size = ((ff.rowIndexStride + 0x7f) >> 7) * (128 + 1);
            break;
        case SHORT:
            data_stream_size = ((ff.rowIndexStride + 0x1ff) >> 9) * (512 * 2 + 2);
            encoding_kind = DIRECT_V2;
            break;
        case FLOAT:
            if (!columns[i]->null_count) {
              data_stream_size = -1; // Pass through (no RLE encoding for floating point)
            } else {
              data_stream_size = ((ff.rowIndexStride + 0x1ff) >> 9) * (512 * 4 + 2);
            }
            break;
        case INT:
        case DATE:
            data_stream_size = ((ff.rowIndexStride + 0x1ff) >> 9) * (512 * 4 + 2);
            encoding_kind = DIRECT_V2;
            break;
        case DOUBLE:
            if (!columns[i]->null_count) {
              data_stream_size = -1; // Pass through (no RLE encoding for floating point)
            } else {
              data_stream_size = ((ff.rowIndexStride + 0x1ff) >> 9) * (512 * 8 + 2);
            }
            break;
        case LONG:
            data_stream_size = ((ff.rowIndexStride + 0x1ff) >> 9) * (512 * 8 + 2);
            encoding_kind = DIRECT_V2;
            break;
        case STRING:
            data_stream_size = -1; // TODO
            data2_stream_size = data_stream_size = ((ff.rowIndexStride + 0x1ff) >> 9) * (512 * 4 + 2);
            encoding_kind = DIRECT_V2;
            break;
        case TIMESTAMP:
            data2_stream_size = data_stream_size = ((ff.rowIndexStride + 0x1ff) >> 9) * (512*4 + 2);
            data2_kind = SECONDARY;
            encoding_kind = DIRECT_V2;
            has_timestamp_column = true;
            break;
        default:
            break;
        }
        if (present_stream_size != 0)
        {
            uint32_t present_stream_id = (uint32_t)sf.streams.size();
            sf.streams.resize(present_stream_id + 1);
            sf.streams[present_stream_id].column = 1 + i;
            sf.streams[present_stream_id].kind = PRESENT;
            sf.streams[present_stream_id].length = present_stream_size;
            stream_ids[i * gpu::CI_NUM_STREAMS + gpu::CI_PRESENT] = (int32_t)present_stream_id;
        }
        if (data_stream_size != 0)
        {
            uint32_t data_stream_id = (uint32_t)sf.streams.size();
            sf.streams.resize(data_stream_id + 1);
            sf.streams[data_stream_id].column = 1 + i;
            sf.streams[data_stream_id].kind = data_kind;
            sf.streams[data_stream_id].length = std::max(data_stream_size, 0);
            stream_ids[i * gpu::CI_NUM_STREAMS + gpu::CI_DATA] = (int32_t)data_stream_id;
        }
        if (data2_stream_size != 0)
        {
            uint32_t data_stream_id = (uint32_t)sf.streams.size();
            sf.streams.resize(data_stream_id + 1);
            sf.streams[data_stream_id].column = 1 + i;
            sf.streams[data_stream_id].kind = data2_kind;
            sf.streams[data_stream_id].length = std::max(data2_stream_size, 0);
            stream_ids[i * gpu::CI_NUM_STREAMS + gpu::CI_DATA2] = (int32_t)data_stream_id;
        }
        ff.types[0].subtypes[i] = 1 + i;
        if (columns[i]->col_name)
            ff.types[0].fieldNames[i].assign(columns[i]->col_name);
        else
            ff.types[0].fieldNames[i] = "_col" + std::to_string(i);
        sf.columns[1 + i].kind = encoding_kind;
        sf.columns[1 + i].dictionarySize = 0;
    }
    sf.writerTimezone = (has_timestamp_column) ? "UTC" : "";
    num_rowgroups = (ff.numberOfRows + ff.rowIndexStride - 1) / ff.rowIndexStride;
    num_chunks = num_rowgroups * num_columns;
    strm_offsets.resize(sf.streams.size());
    rleout_bfr_size = 0;
    for (size_t i = 0; i < sf.streams.size(); i++)
    {
        strm_offsets[i] = rleout_bfr_size;
        rleout_bfr_size += (sf.streams[i].length + 3) & ~3;
    }
    hostdevice_vector<gpu::EncChunk> chunks(num_chunks);
    device_buffer<uint8_t> rleout_bfr_dev(rleout_bfr_size * num_chunks);
    for (size_t j = 0; j < num_rowgroups; j++)
    {
        for (size_t i = 0; i < (size_t)num_columns; i++)
        {
            gpu::EncChunk *ck = &chunks[j * num_columns + i];

            ck->valid_map_base = (const uint32_t *)columns[i]->valid;
            ck->column_data_base = columns[i]->data;
            ck->start_row = (uint32_t)(j * ff.rowIndexStride);
            ck->num_rows = (uint32_t)std::min((uint32_t)ff.rowIndexStride, (uint32_t)(ff.numberOfRows - ck->start_row));
            ck->valid_rows = (uint32_t)columns[i]->size;
            ck->encoding_kind = (uint8_t)sf.columns[i].kind;
            ck->type_kind = (uint8_t)ff.types[1+i].kind;
            ck->dtype_len = 0;
            switch(columns[i]->dtype_info.time_unit)
            {
            case TIME_UNIT_s:   ck->scale = 9; break;
            case TIME_UNIT_ms:  ck->scale = 6; break;
            case TIME_UNIT_us:  ck->scale = 3; break;
            case TIME_UNIT_ns:  ck->scale = 0; break;
            default:            ck->scale = 0; break;
            }
            for (int k = 0; k < gpu::CI_NUM_STREAMS; k++)
            {
                int32_t strm_id = stream_ids[i * gpu::CI_NUM_STREAMS + k];
                ck->strm_id[k] = strm_id;
                if (strm_id >= 0)
                {
                    ck->strm_len[k] = (uint32_t)sf.streams[strm_id].length;
                    if (k == gpu::CI_DATA && ck->strm_len[k] == 0)
                    {
                        // Pass-through
                        ck->streams[k] = nullptr;
                        ck->dtype_len = (ck->type_kind == DOUBLE) ? 8 : 4;
                        ck->strm_len[k] = ck->num_rows * ck->dtype_len;
                    }
                    else
                    {
                        ck->streams[k] = rleout_bfr_dev.data() + rleout_bfr_size * j + strm_offsets[strm_id];
                    }
                }
                else
                {
                    ck->strm_len[k] = 0;
                    ck->streams[k] = nullptr;
                }
            }
        }
    }
    // Encode column data
    CUDA_TRY(cudaMemcpyAsync(chunks.device_ptr(), chunks.host_ptr(),
                             chunks.memory_size(), cudaMemcpyHostToDevice));
    CUDA_TRY(EncodeOrcColumnData(chunks.device_ptr(), (uint32_t)num_columns,
                                 (uint32_t)num_rowgroups));
    CUDA_TRY(cudaMemcpyAsync(chunks.host_ptr(), chunks.device_ptr(),
                             chunks.memory_size(), cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaStreamSynchronize(0));
    // Decide stripe boundaries
    for (size_t group = 0, stripe_size = 0, stripe_start = 0; group <= num_rowgroups; group++)
    {
        size_t ck_size = 0;
        if (group < num_rowgroups)
        {
            for (int i = 0; i < num_columns; i++)
            {
                gpu::EncChunk *ck = &chunks[group * num_columns + i];
                for (int k = 0; k < gpu::CI_NUM_STREAMS; k++)
                {
                    ck_size += ck->strm_len[k];
                }
            }
        }
        if ((stripe_size != 0 && stripe_size + ck_size > 64 * 1024 * 1024) || group == num_rowgroups) // TBD: Stripe size hardcoded to 64MB
        {
            // Flush and start a new stripe
            uint64_t stripe_end = std::min((uint64_t)group * ff.rowIndexStride, ff.numberOfRows);
            size_t stripe_id = ff.stripes.size();
            ff.stripes.resize(stripe_id + 1);
            ff.stripes[stripe_id].dataLength = stripe_size;
            ff.stripes[stripe_id].numberOfRows = (uint32_t)(stripe_end - stripe_start);
            stripe_size = 0;
            stripe_start = stripe_end;
        }
        stripe_size += ck_size;
    }

    // Write file header
    outfile_.write(ps.magic.c_str(), ps.magic.length());

    // Write stripe data
    max_stream_size = 0;
    for (size_t stripe_id = 0, group = 0; stripe_id < ff.stripes.size(); stripe_id++)
    {
        size_t groups_in_stripe = (ff.stripes[stripe_id].numberOfRows + ff.rowIndexStride - 1) / ff.rowIndexStride;
        int max_size = 0;
        ff.stripes[stripe_id].offset = outfile_.tellp();
        // Write index streams
        ff.stripes[stripe_id].indexLength = 0;
        for (size_t strm = 0; strm <= (size_t)num_columns; strm++)
        {
            TypeKind kind = ff.types[strm].kind;
            int32_t present_blk = -1, present_pos = -1, present_size = 0;
            int32_t data_blk = -1, data_pos = -1, data_size = 0;
            int32_t data2_blk = -1, data2_pos = -1, data2_size = 0;

            buf.resize(0);
            // TBD: Not sure we need an empty index stream for record column 0
            if (strm != 0)
            {
                gpu::EncChunk *ck = &chunks[strm - 1];
                if (ck->strm_id[gpu::CI_PRESENT] > 0)
                {
                    present_pos = 0;
                }
                if (ck->strm_id[gpu::CI_DATA] > 0)
                {
                    data_pos = 0;
                }
                if (ck->strm_id[gpu::CI_DATA2] > 0)
                {
                    data2_pos = 0;
                }
            }
            for (size_t g = group; g < group + groups_in_stripe; g++)
            {
                pbw.put_row_index_entry(present_blk, present_pos, data_blk, data_pos, data2_blk, data2_pos, kind);
                if (strm != 0)
                {
                    gpu::EncChunk *ck = &chunks[g * num_columns + strm - 1];
                    if (present_pos >= 0)
                    {
                        present_pos += ck->strm_len[gpu::CI_PRESENT];
                        present_size += ck->strm_len[gpu::CI_PRESENT];
                    }
                    if (data_pos >= 0)
                    {
                        data_pos += ck->strm_len[gpu::CI_DATA];
                        data_size += ck->strm_len[gpu::CI_DATA];
                    }
                    if (data2_pos >= 0)
                    {
                        data2_pos += ck->strm_len[gpu::CI_DATA2];
                        data2_size += ck->strm_len[gpu::CI_DATA2];
                    }
                }
            }
            max_size = std::max(max_size, present_size);
            max_size = std::max(max_size, data_size);
            max_size = std::max(max_size, data2_size);
            sf.streams[strm].length = buf.size();
            outfile_.write(reinterpret_cast<char*>(buf.data()), buf.size());
            ff.stripes[stripe_id].indexLength += buf.size();
        }
        if (max_size > max_stream_size)
        {
            max_stream_size = max_size;
            stream_io_buf =
                pinned_buffer<uint8_t>{[](size_t size) {
                                         uint8_t *ptr = nullptr;
                                         CUDA_TRY(cudaMallocHost(&ptr, size));
                                         return ptr;
                                       }(max_stream_size),
                                       cudaFreeHost};
        }
        // Write data streams
        ff.stripes[stripe_id].dataLength = 0;
        for (int i = 0; i < num_columns; i++)
        {
            gpu::EncChunk *ck = &chunks[group * num_columns + i];
            if (ck->strm_id[gpu::CI_PRESENT] > 0)
            {
                size_t len = gather_stripe_stream(stream_io_buf.get(), ck, gpu::CI_PRESENT, groups_in_stripe, num_columns);
                outfile_.write(reinterpret_cast<char*>(stream_io_buf.get()), len);
                ff.stripes[stripe_id].dataLength += len;
                sf.streams[ck->strm_id[gpu::CI_PRESENT]].length = len;
            }
            if (ck->strm_id[gpu::CI_DATA] > 0)
            {
                size_t len = gather_stripe_stream(stream_io_buf.get(), ck, gpu::CI_DATA, groups_in_stripe, num_columns);
                outfile_.write(reinterpret_cast<char*>(stream_io_buf.get()), len);
                ff.stripes[stripe_id].dataLength += len;
                sf.streams[ck->strm_id[gpu::CI_DATA]].length = len;
            }
            if (ck->strm_id[gpu::CI_DATA2] > 0)
            {
                size_t len = gather_stripe_stream(stream_io_buf.get(), ck, gpu::CI_DATA2, groups_in_stripe, num_columns);
                outfile_.write(reinterpret_cast<char*>(stream_io_buf.get()), len);
                ff.stripes[stripe_id].dataLength += len;
                sf.streams[ck->strm_id[gpu::CI_DATA2]].length = len;
            }
        }
        // Write stripe footer
        buf.resize(0);
        ff.stripes[stripe_id].footerLength = (uint32_t)pbw.write(&sf);
        outfile_.write(reinterpret_cast<char*>(buf.data()), ff.stripes[stripe_id].footerLength);
        group += groups_in_stripe;
    }

    // TBD: We may want to add pandas or spark column metadata strings here
    ff.contentLength = outfile_.tellp();
    buf.resize(0);
    ps.footerLength = pbw.write(&ff);
    ps_length = pbw.write(&ps);
    buf.push_back((uint8_t)ps_length);

    // Write metadata
    outfile_.write(reinterpret_cast<char*>(buf.data()), buf.size());
    outfile_.flush();
}

writer::writer(std::string filepath, writer_options const& options)
    : impl_(std::make_unique<Impl>(filepath, options)) {}

void writer::write_all(const cudf::table& table) { impl_->write(table); }

writer::~writer() = default;

}  // namespace orc
}  // namespace io
}  // namespace cudf
