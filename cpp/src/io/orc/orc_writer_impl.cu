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

#include <io/utilities/wrapper_utils.hpp>
#include <nvstrings/NVCategory.h>
#include <nvstrings/NVStrings.h>

namespace {

template <typename T>
using pinned_buffer = std::unique_ptr<T, decltype(&cudaFreeHost)>;

}  // namespace

namespace cudf {
namespace io {
namespace orc {

/**
 * @brief Function that translates GDF compression to ORC compression
 **/
constexpr CompressionKind to_orckind(compression_type compression) {
  switch (compression) {
    case compression_type::snappy:
      return SNAPPY;
    case compression_type::none:
    default:
      return NONE;
  }
}

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
    case GDF_TIMESTAMP:
      return TIMESTAMP;
    case GDF_CATEGORY:
      return INT;
    case GDF_STRING:
    case GDF_STRING_CATEGORY:
      return STRING;
    default:
      return INVALID_TYPE_KIND;
  }
}

/**
 * @brief Function that translates time unit to nanoscale multiple
 **/
template <typename T>
constexpr T to_clockscale(gdf_time_unit time_unit) {
  switch (time_unit) {
    case TIME_UNIT_s:
      return 9;
    case TIME_UNIT_ms:
      return 6;
    case TIME_UNIT_us:
      return 3;
    case TIME_UNIT_ns:
    default:
      return 0;
  }
}

writer::Impl::Impl(std::string filepath, writer_options const &options) {
  compression_ = options.compression;

  outfile_.open(filepath, std::ios::out | std::ios::binary | std::ios::trunc);
  CUDF_EXPECTS(outfile_.is_open(), "Cannot open output file");
}

void writer::Impl::write(const cudf::table& table) {
    auto columns = table.begin();
    const int num_columns = table.num_columns();

    size_t ps_length;
    std::vector<size_t> strm_offsets;
    std::vector<int> str_col_ids;

    size_t compressed_bfr_size;
    uint32_t num_compressed_blocks = 0;

    PostScript ps;
    ps.footerLength = 0;
    ps.compression = to_orckind(compression_);
    ps.compressionBlockSize = 256 * 1024; // TODO: Pick smaller values if too few compression blocks
    ps.version = {0,12};
    ps.metadataLength = 0; // TODO: Write stripe statistics
    ps.magic = MAGIC;

    FileFooter ff;
    ff.headerLength = ps.magic.length();
    ff.numberOfRows = 0;
    ff.rowIndexStride = ROW_INDEX_STRIDE;
    ff.types.resize(1 + num_columns);
    ff.types[0].kind = STRUCT;
    ff.types[0].subtypes.resize(num_columns);
    ff.types[0].fieldNames.resize(num_columns);

    std::vector<int> str_col_map(num_columns);

    // Init streams from columns
    auto streams = [&](size_t& num_rows) {
      std::vector<Stream> streams(num_columns + 1);
      streams[0].column = 0;
      streams[0].kind = ROW_INDEX;
      streams[0].length = 0;
      for (int i = 0; i < num_columns; ++i) {
        num_rows = std::max<uint64_t>(num_rows, table.get_column(i)->size);
        streams[1 + i].column = 1 + i;
        streams[1 + i].kind = ROW_INDEX;
        streams[1 + i].length = 0;

        str_col_map[i] = str_col_ids.size();
        if (columns[i]->dtype == GDF_STRING ||
            columns[i]->dtype == GDF_STRING_CATEGORY) {
          str_col_ids.push_back(i);
        }
      }
      return streams;
    }(ff.numberOfRows);

    const auto num_rowgroups = (ff.numberOfRows + ROW_INDEX_STRIDE - 1) / ROW_INDEX_STRIDE;
    const auto num_chunks = num_rowgroups * num_columns;

    const auto num_string_columns = str_col_ids.size();
    const auto num_dict_chunks = num_rowgroups * num_string_columns;

    // Build initial dictionaries
    hostdevice_vector<gpu::DictionaryChunk> dict(num_dict_chunks);
    device_buffer<uint32_t> dict_index(num_string_columns * ff.numberOfRows);
    device_buffer<uint32_t> dict_data(num_string_columns * ff.numberOfRows);
    std::vector<device_buffer<std::pair<const char*,size_t>>> str_indices(num_string_columns);
    if (num_dict_chunks != 0) {
      build_dictionary_chunks(table, ff.numberOfRows, str_col_ids, dict_data,
                              dict_index, str_indices, dict);
    }

    // Decide stripe boundaries early on, based on uncompressed size
    std::vector<uint32_t> stripe_list;
    for (size_t g = 0, stripe_start = 0, stripe_size = 0; g < num_rowgroups; g++)
    {
      size_t rowgroup_size = 0;
      for (int i = 0; i < num_columns; i++) {
        if (columns[i]->dtype == GDF_STRING ||
            columns[i]->dtype == GDF_STRING_CATEGORY) {
          const auto &dt = dict[g * num_string_columns + str_col_map[i]];
          rowgroup_size += 1 * ROW_INDEX_STRIDE;
          rowgroup_size += dt.string_char_count;
        } else {
          rowgroup_size += cudf::size_of(columns[i]->dtype) * ROW_INDEX_STRIDE;
        }
      }

      // Apply rows per stripe limit to limit string dictionaries
      const uint32_t max_stripe_rows =
          (num_string_columns != 0) ? 1000000 : 5000000;
      if (g > stripe_start &&
          (stripe_size + rowgroup_size > MAX_STRIPE_SIZE ||
           (g + 1 - stripe_start) * ROW_INDEX_STRIDE > max_stripe_rows)) {
        stripe_list.push_back(g - stripe_start);
        stripe_start = g;
        stripe_size = 0;
      }
      stripe_size += rowgroup_size;
      if (g + 1 == num_rowgroups) {
        stripe_list.push_back(num_rowgroups - stripe_start);
      }
    }

    // Build stripe-level dictionaries
    const size_t num_stripes = stripe_list.size();
    const size_t num_stripe_dict = num_stripes * num_string_columns;
    hostdevice_vector<gpu::StripeDictionary> stripe_dict(num_stripe_dict);
    if (num_dict_chunks != 0) {
      build_stripe_dictionaries(ff.numberOfRows, str_col_ids, stripe_list, dict,
                                dict_index, stripe_dict);
    }

    // Initialize streams
    std::vector<int32_t> stream_ids(num_columns * gpu::CI_NUM_STREAMS, -1);
    auto sf = init_stripe_footer(table, streams, stripe_list, dict, stripe_dict,
                                 str_col_map, stream_ids, ff.types);

    // Allocate combined RLE and string data buffer
    size_t strdata_bfr_size = 0;
    auto rleout_bfr_dev = [&]() {
      strm_offsets.resize(sf.streams.size());

      size_t rleout_bfr_size = 0;
      for (size_t i = 0; i < sf.streams.size(); i++) {
        if (((sf.streams[i].kind == DICTIONARY_DATA ||
              sf.streams[i].kind == LENGTH) &&
             sf.columns[sf.streams[i].column].kind == DICTIONARY_V2) ||
            (sf.streams[i].kind == DATA &&
             ff.types[sf.streams[i].column].kind == STRING &&
             sf.columns[sf.streams[i].column].kind == DIRECT_V2)) {
          strm_offsets[i] = strdata_bfr_size;
          strdata_bfr_size += sf.streams[i].length;
        } else {
          strm_offsets[i] = rleout_bfr_size;
          rleout_bfr_size += (sf.streams[i].length * num_rowgroups + 7) & ~7;
        }
      }
      strdata_bfr_size = (strdata_bfr_size + 7) & ~7;

      return device_buffer<uint8_t>(rleout_bfr_size + strdata_bfr_size);
    }();

    hostdevice_vector<gpu::EncChunk> chunks(num_chunks);
    for (size_t j = 0, stripe_start = 0, stripe_id = 0; j < num_rowgroups; j++)
    {
        for (size_t i = 0; i < (size_t)num_columns; i++)
        {
            gpu::EncChunk *ck = &chunks[j * num_columns + i];
            ck->valid_map_base =
                reinterpret_cast<const uint32_t *>(columns[i]->valid);
            ck->start_row = (uint32_t)(j * ROW_INDEX_STRIDE);
            ck->num_rows = std::min<uint32_t>(ROW_INDEX_STRIDE, ff.numberOfRows - ck->start_row);
            ck->valid_rows = columns[i]->size;
            ck->encoding_kind = (uint8_t)sf.columns[1+i].kind;
            ck->type_kind = (uint8_t)ff.types[1+i].kind;
            ck->dtype_len = 0;
            ck->scale = to_clockscale<uint8_t>(columns[i]->dtype_info.time_unit);
            if (ck->type_kind == STRING) {
              if (ck->encoding_kind == DICTIONARY_V2) {
                auto index = stripe_id * num_string_columns + str_col_map[i];
                ck->column_data_base = stripe_dict[index].dict_index;
              } else {
                ck->column_data_base = str_indices[str_col_map[i]].data();
              }
              ck->dtype_len = 1;
            } else {
              ck->column_data_base = columns[i]->data;
              ck->dtype_len = cudf::size_of(columns[i]->dtype);
            }

            for (int k = 0; k < gpu::CI_NUM_STREAMS; k++)
            {
                int32_t strm_id = stream_ids[i * gpu::CI_NUM_STREAMS + k];
                ck->strm_id[k] = strm_id;
                if (strm_id >= 0)
                {
                    if (k == gpu::CI_DICTIONARY || (k == gpu::CI_DATA2 && ck->encoding_kind == DICTIONARY_V2))
                    {
                        if (j == stripe_start)
                        {
                            const gpu::StripeDictionary *stripe = &stripe_dict[stripe_id * num_string_columns + str_col_map[i]];
                            ck->strm_len[k] = (k == gpu::CI_DICTIONARY) ? stripe->dict_char_count : (((stripe->num_strings + 0x1ff) >> 9) * (512 * 4 + 2));
                            if (stripe_id == 0)
                            {
                                ck->streams[k] = rleout_bfr_dev.data() + strm_offsets[strm_id];
                            }
                            else
                            {
                                const gpu::EncChunk *ck_up = &chunks[stripe[-(int32_t)num_string_columns].start_chunk * num_columns + i];
                                ck->streams[k] = ck_up->streams[k] + ck_up->strm_len[k];
                            }
                        }
                        else
                        {
                            ck->strm_len[k] = 0;
                            ck->streams[k] = ck[-num_columns].streams[k];
                        }
                    }
                    else if (k == gpu::CI_DATA && ck->type_kind == STRING && ck->encoding_kind == DIRECT_V2)
                    {
                        ck->strm_len[k] = dict[j * num_string_columns + str_col_map[i]].string_char_count;
                        ck->streams[k] = (j == 0) ? rleout_bfr_dev.data() + strm_offsets[strm_id] : (ck[-num_columns].streams[k] + ck[-num_columns].strm_len[k]);
                    }
                    else if (k == gpu::CI_DATA && sf.streams[strm_id].length == 0 && (ck->type_kind == DOUBLE || ck->type_kind == FLOAT))
                    {
                        // Pass-through
                        ck->streams[k] = nullptr;
                        ck->strm_len[k] = ck->num_rows * ck->dtype_len;
                    }
                    else
                    {
                        ck->streams[k] = rleout_bfr_dev.data() + strdata_bfr_size + strm_offsets[strm_id] + sf.streams[strm_id].length * j;
                        ck->strm_len[k] = (uint32_t)sf.streams[strm_id].length;
                    }
                }
                else
                {
                    ck->strm_len[k] = 0;
                    ck->streams[k] = nullptr;
                }
            }
        }
        if (j + 1 == stripe_start + stripe_list[stripe_id])
        {
            stripe_start = j + 1;
            stripe_id++;
        }
    }
    CUDA_TRY(cudaMemcpyAsync(chunks.device_ptr(), chunks.host_ptr(),
                             chunks.memory_size(), cudaMemcpyHostToDevice));
    // Encode string dictionaries
    if (num_dict_chunks != 0) {
      CUDA_TRY(EncodeStripeDictionaries(
          stripe_dict.device_ptr(), chunks.device_ptr(),
          (uint32_t)num_string_columns, (uint32_t)num_columns,
          (uint32_t)stripe_list.size()));
    }
    CUDA_TRY(EncodeOrcColumnData(chunks.device_ptr(), (uint32_t)num_columns,
                                 (uint32_t)num_rowgroups));
    CUDA_TRY(cudaStreamSynchronize(0));

    // Initialize stripe data in file footer
    const auto num_data_streams = sf.streams.size() - (num_columns + 1); // Exclude index streams
    const auto num_stripe_streams = stripe_list.size() * num_data_streams;
    hostdevice_vector<gpu::StripeStream> strm_desc(num_stripe_streams);
    ff.stripes =
        compact_streams(num_columns, ff.numberOfRows,
                        num_data_streams, stripe_list, chunks, strm_desc);

    // Allocate intermediate output stream buffer
    num_compressed_blocks = 0;
    compressed_bfr_size = 0;
    auto stream_output = [&]() {
      size_t max_stream_size = 0;

      for (size_t stripe_id = 0; stripe_id < stripe_list.size(); stripe_id++) {
        for (size_t i = 0; i < num_data_streams; i++) {
          gpu::StripeStream *ss = &strm_desc[stripe_id * num_data_streams + i];
          size_t stream_size = ss->stream_size;
          if (ps.compression != NONE) {
            ss->first_block = num_compressed_blocks;
            ss->bfr_offset = compressed_bfr_size;

            auto num_blocks =
                std::max<uint32_t>((stream_size + ps.compressionBlockSize - 1) /
                                       ps.compressionBlockSize,
                                   1);
            stream_size += num_blocks * 3;
            num_compressed_blocks += num_blocks;
            compressed_bfr_size += stream_size;
          }
          max_stream_size = std::max(max_stream_size, stream_size);
        }
      }

      return pinned_buffer<uint8_t>{[](size_t size) {
                                      uint8_t *ptr = nullptr;
                                      CUDA_TRY(cudaMallocHost(&ptr, size));
                                      return ptr;
                                    }(max_stream_size),
                                    cudaFreeHost};
    }();

    // Compress the data streams
    device_buffer<uint8_t> compressed_data(compressed_bfr_size);
    hostdevice_vector<gpu_inflate_status_s> comp_out(num_compressed_blocks);
    hostdevice_vector<gpu_inflate_input_s> comp_in(num_compressed_blocks);
    if (ps.compression != NONE) {
      CUDA_TRY(cudaMemcpyAsync(strm_desc.device_ptr(), strm_desc.host_ptr(),
                               strm_desc.memory_size(),
                               cudaMemcpyHostToDevice));
      CUDA_TRY(CompressOrcDataStreams(
          compressed_data.data(), strm_desc.device_ptr(), chunks.device_ptr(),
          comp_in.device_ptr(), comp_out.device_ptr(),
          (uint32_t)num_stripe_streams, num_compressed_blocks, ps.compression,
          ps.compressionBlockSize));
      CUDA_TRY(cudaMemcpyAsync(strm_desc.host_ptr(), strm_desc.device_ptr(),
                               strm_desc.memory_size(),
                               cudaMemcpyDeviceToHost));
      CUDA_TRY(cudaMemcpyAsync(comp_out.host_ptr(), comp_out.device_ptr(),
                               comp_out.memory_size(), cudaMemcpyDeviceToHost));
      CUDA_TRY(cudaStreamSynchronize(0));
    }

    // Write file header
    outfile_.write(ps.magic.c_str(), ps.magic.length());

    ProtobufWriter pbw_(&buffer_);
    for (size_t stripe_id = 0, group = 0; stripe_id < ff.stripes.size(); stripe_id++)
    {
        size_t groups_in_stripe = (ff.stripes[stripe_id].numberOfRows + ROW_INDEX_STRIDE - 1) / ROW_INDEX_STRIDE;
        ff.stripes[stripe_id].offset = outfile_.tellp();
        // Write index streams
        ff.stripes[stripe_id].indexLength = 0;
        for (size_t strm = 0; strm <= (size_t)num_columns; strm++)
        {
            TypeKind kind = ff.types[strm].kind;
            int32_t present_blk = -1, present_pos = -1, present_comp_pos = -1, present_comp_sz = -1;
            int32_t data_blk = -1, data_pos = -1, data_comp_pos = -1, data_comp_sz = -1;
            int32_t data2_blk = -1, data2_pos = -1, data2_comp_pos = -1, data2_comp_sz = -1;

            buffer_.resize((ps.compression != NONE) ? 3 : 0);
            // TBD: Not sure we need an empty index stream for record column 0
            if (strm != 0)
            {
                gpu::EncChunk *ck = &chunks[strm - 1];
                if (ck->strm_id[gpu::CI_PRESENT] > 0)
                {
                    present_pos = 0;
                    if (ps.compression != NONE)
                    {
                        const gpu::StripeStream *ss = &strm_desc[stripe_id * num_data_streams + ck->strm_id[gpu::CI_PRESENT] - (num_columns + 1)];
                        present_blk = ss->first_block;
                        present_comp_pos = 0;
                        present_comp_sz = ss->stream_size;
                    }
                }
                if (ck->strm_id[gpu::CI_DATA] > 0)
                {
                    data_pos = 0;
                    if (ps.compression != NONE)
                    {
                        const gpu::StripeStream *ss = &strm_desc[stripe_id * num_data_streams + ck->strm_id[gpu::CI_DATA] - (num_columns + 1)];
                        data_blk = ss->first_block;
                        data_comp_pos = 0;
                        data_comp_sz = ss->stream_size;
                    }
                }
                if (ck->strm_id[gpu::CI_DATA2] > 0)
                {
                    data2_pos = 0;
                    if (ps.compression != NONE)
                    {
                        const gpu::StripeStream *ss = &strm_desc[stripe_id * num_data_streams + ck->strm_id[gpu::CI_DATA2] - (num_columns + 1)];
                        data2_blk = ss->first_block;
                        data2_comp_pos = 0;
                        data2_comp_sz = ss->stream_size;
                    }
                }
            }
            if (kind == STRING && sf.columns[strm].kind == DICTIONARY_V2)
            {
                kind = INT; // Change string dictionary to int from index point of view
            }
            for (size_t g = group; g < group + groups_in_stripe; g++)
            {
                pbw_.put_row_index_entry(present_comp_pos, present_pos, data_comp_pos, data_pos, data2_comp_pos, data2_pos, kind);
                if (strm != 0)
                {
                    gpu::EncChunk *ck = &chunks[g * num_columns + strm - 1];
                    if (present_pos >= 0)
                    {
                        present_pos += ck->strm_len[gpu::CI_PRESENT];
                        while (present_blk >= 0 && (size_t)present_pos >= ps.compressionBlockSize && present_comp_pos + 3 + comp_out[present_blk].bytes_written < (size_t)present_comp_sz)
                        {
                            present_pos -= ps.compressionBlockSize;
                            present_comp_pos += 3 + comp_out[present_blk].bytes_written;
                            present_blk++;
                        }
                    }
                    if (data_pos >= 0)
                    {
                        data_pos += ck->strm_len[gpu::CI_DATA];
                        while (data_blk >= 0 && (size_t)data_pos >= ps.compressionBlockSize && data_comp_pos + 3 + comp_out[data_blk].bytes_written < (size_t)data_comp_sz)
                        {
                            data_pos -= ps.compressionBlockSize;
                            data_comp_pos += 3 + comp_out[data_blk].bytes_written;
                            data_blk++;
                        }
                    }
                    if (data2_pos >= 0)
                    {
                        data2_pos += ck->strm_len[gpu::CI_DATA2];
                        while (data2_blk >= 0 && (size_t)data2_pos >= ps.compressionBlockSize && data2_comp_pos + 3 + comp_out[data2_blk].bytes_written < (size_t)data2_comp_sz)
                        {
                            data2_pos -= ps.compressionBlockSize;
                            data2_comp_pos += 3 + comp_out[data2_blk].bytes_written;
                            data2_blk++;
                        }
                    }
                }
            }
            sf.streams[strm].length = buffer_.size();
            if (ps.compression != NONE)
            {
                uint32_t uncomp_ix_len = (uint32_t)(sf.streams[strm].length - 3) * 2 + 1;
                buffer_[0] = static_cast<uint8_t>(uncomp_ix_len >> 0);
                buffer_[1] = static_cast<uint8_t>(uncomp_ix_len >> 8);
                buffer_[2] = static_cast<uint8_t>(uncomp_ix_len >> 16);
            }
            outfile_.write(reinterpret_cast<char*>(buffer_.data()), buffer_.size());
            ff.stripes[stripe_id].indexLength += buffer_.size();
        }
        // Write data streams
        ff.stripes[stripe_id].dataLength = 0;
        for (size_t i = 0; i < num_data_streams; i++)
        {
            const gpu::StripeStream *ss = &strm_desc[stripe_id * num_data_streams + i];
            const gpu::EncChunk *ck = &chunks[group * num_columns + ss->column_id];
            size_t len = ss->stream_size;
            sf.streams[ck->strm_id[ss->strm_type]].length = len;
            if (len > 0)
            {
                uint8_t *strm_dev = (ps.compression == NONE) ? ck->streams[ss->strm_type] : (compressed_data.data() + ss->bfr_offset);
                CUDA_TRY(cudaMemcpyAsync(stream_output.get(), strm_dev, len, cudaMemcpyDeviceToHost, 0));
                CUDA_TRY(cudaStreamSynchronize(0));
                outfile_.write(reinterpret_cast<char*>(stream_output.get()), len);
                ff.stripes[stripe_id].dataLength += len;
            }
            if (ck->encoding_kind == DICTIONARY_V2)
            {
                uint32_t column_id = ss->column_id;
                sf.columns[1 + column_id].dictionarySize = stripe_dict[stripe_id * num_string_columns + str_col_map[column_id]].num_strings;
            }
        }
        // Write stripe footer
        buffer_.resize((ps.compression != NONE) ? 3 : 0);
        pbw_.write(&sf);
        ff.stripes[stripe_id].footerLength = (uint32_t)buffer_.size();
        if (ps.compression != NONE)
        {
            uint32_t uncomp_sf_len = (ff.stripes[stripe_id].footerLength - 3) * 2 + 1;
            buffer_[0] = static_cast<uint8_t>(uncomp_sf_len >> 0);
            buffer_[1] = static_cast<uint8_t>(uncomp_sf_len >> 8);
            buffer_[2] = static_cast<uint8_t>(uncomp_sf_len >> 16);
        }
        outfile_.write(reinterpret_cast<char*>(buffer_.data()), ff.stripes[stripe_id].footerLength);
        group += groups_in_stripe;
    }

    // TBD: We may want to add pandas or spark column metadata strings here
    ff.contentLength = outfile_.tellp();
    buffer_.resize((ps.compression != NONE) ? 3 : 0);
    pbw_.write(&ff);
    ps.footerLength = buffer_.size();
    if (ps.compression != NONE)
    {
        // TODO: If the file footer ends up larger than the compression block size, we'll need to insert additional 3-byte block headers
        uint32_t uncomp_ff_len = (uint32_t)(ps.footerLength - 3) * 2 + 1;
        buffer_[0] = static_cast<uint8_t>(uncomp_ff_len >> 0);
        buffer_[1] = static_cast<uint8_t>(uncomp_ff_len >> 8);
        buffer_[2] = static_cast<uint8_t>(uncomp_ff_len >> 16);
    }
    ps_length = pbw_.write(&ps);
    buffer_.push_back((uint8_t)ps_length);

    // Write metadata
    outfile_.write(reinterpret_cast<char*>(buffer_.data()), buffer_.size());
    outfile_.flush();
}

void writer::Impl::build_dictionary_chunks(
    const cudf::table &table, size_t num_rows, const std::vector<int> &str_col_ids,
    const device_buffer<uint32_t> &dict_data,
    const device_buffer<uint32_t> &dict_index,
    std::vector<device_buffer<std::pair<const char*, size_t>>>& indices,
    hostdevice_vector<gpu::DictionaryChunk> &dict) {
  auto columns = table.begin();
  const auto num_string_columns = str_col_ids.size();
  const auto num_rowgroups = dict.size() / num_string_columns;

  for (int i = 0; i < (int)num_string_columns; i++) {
    const gdf_column *col = columns[str_col_ids[i]];

    indices[i].resize(col->size);
    if (col->dtype == GDF_STRING) {
      auto *str = static_cast<NVStrings *>(col->data);
      CUDF_EXPECTS(str->create_index(indices[i].data()) == 0,
                   "Cannot retrieve nvcategory string pairs");
    } else if (col->dtype == GDF_STRING_CATEGORY) {
      auto *cat = static_cast<NVCategory *>(col->dtype_info.category);
      CUDF_EXPECTS(cat->create_index(indices[i].data()) == 0,
                   "Cannot retrieve nvcategory string pairs");
    } else {
      CUDF_FAIL("Expected a string-type column");
    }

    for (int g = 0; g < (int)num_rowgroups; g++) {
      gpu::DictionaryChunk *ck = &dict[g * num_string_columns + i];
      ck->valid_map_base = (col->null_count != 0)
                               ? reinterpret_cast<const uint32_t *>(col->valid)
                               : nullptr;
      ck->column_data_base = indices[i].data();
      ck->dict_data = dict_data.data() + i * num_rows + g * ROW_INDEX_STRIDE;
      ck->dict_index = dict_index.data() + i * num_rows;  // Indexed by abs row
      ck->start_row = g * ROW_INDEX_STRIDE;
      ck->num_rows = std::min<uint32_t>(
          ROW_INDEX_STRIDE, std::max(col->size - (int)ck->start_row, 0));
      ck->num_strings = 0;
      ck->string_char_count = 0;
      ck->num_dict_strings = 0;
      ck->dict_char_count = 0;
    }
  }

  CUDA_TRY(cudaMemcpyAsync(dict.device_ptr(), dict.host_ptr(),
                           dict.memory_size(), cudaMemcpyHostToDevice));
  CUDA_TRY(InitDictionaryIndices(dict.device_ptr(), num_string_columns,
                                 num_rowgroups));
  CUDA_TRY(cudaMemcpyAsync(dict.host_ptr(), dict.device_ptr(),
                           dict.memory_size(), cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaStreamSynchronize(0));
}

void writer::Impl::build_stripe_dictionaries(
    size_t num_rows, const std::vector<int> &str_col_ids,
    const std::vector<uint32_t> &stripe_list,
    const hostdevice_vector<gpu::DictionaryChunk> &dict,
    const device_buffer<uint32_t> &dict_index,
    hostdevice_vector<gpu::StripeDictionary> &stripe_dict) {
  const auto num_string_columns = str_col_ids.size();
  const auto num_rowgroups = dict.size() / num_string_columns;

  for (size_t i = 0; i < num_string_columns; i++) {
    size_t direct_cost = 0, dict_cost = 0;

    for (size_t j = 0, g = 0; j < stripe_list.size(); j++) {
      uint32_t num_chunks = stripe_list[j];

      auto &sd = stripe_dict[j * num_string_columns + i];
      sd.column_data_base = dict[i].column_data_base;
      sd.dict_data = dict[g * num_string_columns + i].dict_data;
      sd.dict_index = dict_index.data() + i * num_rows; // Indexed by abs row
      sd.column_id = str_col_ids[i];
      sd.start_chunk = (uint32_t)g;
      sd.num_chunks = num_chunks;
      sd.num_strings = 0;
      sd.dict_char_count = 0;
      for (size_t k = g; k < g + num_chunks; k++) {
        const auto &dt = dict[k * num_string_columns + i];
        sd.num_strings += dt.num_dict_strings;

        direct_cost += dt.string_char_count;
        dict_cost += dt.dict_char_count + dt.num_dict_strings;
      }

      g += num_chunks;
    }

    // Early disable of dictionary if it doesn't look good at the chunk level
    if (enable_dictionary_ && dict_cost >= direct_cost) {
      for (size_t j = 0; j < stripe_list.size(); j++) {
        stripe_dict[j * num_string_columns + i].dict_data = nullptr;
      }
    }
  }

  CUDA_TRY(cudaMemcpyAsync(stripe_dict.device_ptr(), stripe_dict.host_ptr(),
                           stripe_dict.memory_size(), cudaMemcpyHostToDevice));
  CUDA_TRY(BuildStripeDictionaries(
      stripe_dict.device_ptr(), stripe_dict.host_ptr(), dict.device_ptr(),
      stripe_list.size(), num_rowgroups, num_string_columns));
  CUDA_TRY(cudaMemcpyAsync(stripe_dict.host_ptr(), stripe_dict.device_ptr(),
                           stripe_dict.memory_size(), cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaStreamSynchronize(0));
}

StripeFooter writer::Impl::init_stripe_footer(
    const cudf::table &table, std::vector<Stream> &streams,
    const std::vector<uint32_t> &stripe_list,
    const hostdevice_vector<gpu::DictionaryChunk> &dict,
    const hostdevice_vector<gpu::StripeDictionary> &stripe_dict,
    const std::vector<int> &str_col_map, std::vector<int32_t> &stream_ids,
    std::vector<SchemaType> &types) {
  auto columns = table.begin();
  const auto num_columns = table.num_columns();

  StripeFooter sf;
  sf.streams = std::move(streams);
  sf.columns.resize(num_columns + 1);
  sf.columns[0].kind = DIRECT;
  sf.columns[0].dictionarySize = 0;

  bool has_timestamp_column = false;
  for (int i = 0; i < num_columns; i++) {
    TypeKind kind = to_orckind(columns[i]->dtype);
    StreamKind data_kind = DATA;
    StreamKind data2_kind = LENGTH;
    ColumnEncodingKind encoding_kind = DIRECT;

    int64_t present_stream_size = 0;
    int64_t data_stream_size = 0;
    int64_t data2_stream_size = 0;
    int64_t dict_stream_size = 0;
    if (columns[i]->null_count != 0 || columns[i]->size != table.num_rows()) {
      present_stream_size = ((ROW_INDEX_STRIDE + 7) >> 3);
      present_stream_size += (present_stream_size + 0x7f) >> 7;
    }

    switch (kind) {
      case BOOLEAN:
        data_stream_size = ((ROW_INDEX_STRIDE + 0x3ff) >> 10) * (128 + 1);
        break;
      case BYTE:
        data_stream_size = ((ROW_INDEX_STRIDE + 0x7f) >> 7) * (128 + 1);
        break;
      case SHORT:
        data_stream_size = ((ROW_INDEX_STRIDE + 0x1ff) >> 9) * (512 * 2 + 2);
        encoding_kind = DIRECT_V2;
        break;
      case FLOAT:
        // Pass through if no nulls (no RLE encoding for floating point)
        data_stream_size =
            (columns[i]->null_count)
                ? ((ROW_INDEX_STRIDE + 0x1ff) >> 9) * (512 * 4 + 2)
                : INT64_C(-1);
        break;
      case INT:
      case DATE:
        data_stream_size = ((ROW_INDEX_STRIDE + 0x1ff) >> 9) * (512 * 4 + 2);
        encoding_kind = DIRECT_V2;
        break;
      case DOUBLE:
        // Pass through if no nulls (no RLE encoding for floating point)
        data_stream_size =
            (columns[i]->null_count)
                ? ((ROW_INDEX_STRIDE + 0x1ff) >> 9) * (512 * 8 + 2)
                : INT64_C(-1);
        break;
      case LONG:
        data_stream_size = ((ROW_INDEX_STRIDE + 0x1ff) >> 9) * (512 * 8 + 2);
        encoding_kind = DIRECT_V2;
        break;
      case STRING: {
        auto num_string_columns = stripe_dict.size() / stripe_list.size();
        bool enable_dict = enable_dictionary_;
        uint32_t scol = str_col_map[i];
        size_t direct_data_size = 0;
        size_t dict_data_size = 0;
        size_t dict_strings = 0;
        size_t dict_lengths_div512 = 0;
        for (size_t stripe = 0, g = 0; stripe < stripe_list.size(); stripe++) {
          const auto& sd = stripe_dict[stripe * num_string_columns + scol];
          enable_dict = (enable_dict && sd.dict_data != nullptr);
          if (enable_dict) {
            dict_strings += sd.num_strings;
            dict_lengths_div512 += (sd.num_strings + 0x1ff) >> 9;
            dict_data_size += sd.dict_char_count;
          }

          for (uint32_t k = 0; k < stripe_list[stripe]; k++, g++) {
            direct_data_size +=
                dict[g * num_string_columns + scol].string_char_count;
          }
        }
        if (enable_dict) {
          uint32_t dict_bits = 0;
          for (dict_bits = 1; dict_bits < 32; dict_bits <<= 1) {
            if (dict_strings <= (1ull << dict_bits)) break;
          }
          const auto valid_count = columns[i]->size - columns[i]->null_count;
          dict_data_size += (dict_bits * valid_count + 7) >> 3;
        }

        // Decide between direct or dictionary encoding
        if (enable_dict && dict_data_size < direct_data_size) {
          data_stream_size = ((ROW_INDEX_STRIDE + 0x1ff) >> 9) * (512 * 4 + 2);
          data2_stream_size = dict_lengths_div512 * (512 * 4 + 2);
          dict_stream_size = std::max<size_t>(dict_data_size, 1);
          encoding_kind = DICTIONARY_V2;
        } else {
          data_stream_size = std::max<size_t>(direct_data_size, 1);
          data2_stream_size =
              ((ROW_INDEX_STRIDE + 0x1ff) >> 9) * (512 * 4 + 2);
          encoding_kind = DIRECT_V2;
        }
        break;
      }
      case TIMESTAMP:
        data2_stream_size = data_stream_size =
            ((ROW_INDEX_STRIDE + 0x1ff) >> 9) * (512 * 4 + 2);
        data2_kind = SECONDARY;
        has_timestamp_column = true;
        encoding_kind = DIRECT_V2;
        break;
      default:
        break;
    }

    if (present_stream_size != 0) {
      uint32_t present_stream_id = (uint32_t)sf.streams.size();
      sf.streams.resize(present_stream_id + 1);
      sf.streams[present_stream_id].column = 1 + i;
      sf.streams[present_stream_id].kind = PRESENT;
      sf.streams[present_stream_id].length = present_stream_size;
      stream_ids[i * gpu::CI_NUM_STREAMS + gpu::CI_PRESENT] =
          (int32_t)present_stream_id;
    }
    if (data_stream_size != 0) {
      uint32_t data_stream_id = (uint32_t)sf.streams.size();
      sf.streams.resize(data_stream_id + 1);
      sf.streams[data_stream_id].column = 1 + i;
      sf.streams[data_stream_id].kind = data_kind;
      sf.streams[data_stream_id].length =
          std::max<int64_t>(data_stream_size, 0);
      stream_ids[i * gpu::CI_NUM_STREAMS + gpu::CI_DATA] =
          (int32_t)data_stream_id;
    }
    if (data2_stream_size != 0) {
      uint32_t data_stream_id = (uint32_t)sf.streams.size();
      sf.streams.resize(data_stream_id + 1);
      sf.streams[data_stream_id].column = 1 + i;
      sf.streams[data_stream_id].kind = data2_kind;
      sf.streams[data_stream_id].length =
          std::max<int64_t>(data2_stream_size, 0);
      stream_ids[i * gpu::CI_NUM_STREAMS + gpu::CI_DATA2] =
          (int32_t)data_stream_id;
    }
    if (dict_stream_size != 0) {
      uint32_t dict_stream_id = (uint32_t)sf.streams.size();
      sf.streams.resize(dict_stream_id + 1);
      sf.streams[dict_stream_id].column = 1 + i;
      sf.streams[dict_stream_id].kind = DICTIONARY_DATA;
      sf.streams[dict_stream_id].length = dict_stream_size;
      stream_ids[i * gpu::CI_NUM_STREAMS + gpu::CI_DICTIONARY] =
          (int32_t)dict_stream_id;
    }
    sf.columns[1 + i].kind = encoding_kind;
    sf.columns[1 + i].dictionarySize = 0;

    // Copy out the filefooter types info
    types[1 + i].kind = kind;
    types[0].subtypes[i] = 1 + i;
    if (columns[i]->col_name) {
      types[0].fieldNames[i].assign(columns[i]->col_name);
    } else {
      types[0].fieldNames[i] = "_col" + std::to_string(i);
    }
  }
  sf.writerTimezone = (has_timestamp_column) ? "UTC" : "";

  return sf;
}

std::vector<StripeInformation> writer::Impl::compact_streams(
    gdf_size_type num_columns, size_t num_rows, size_t num_data_streams,
    const std::vector<uint32_t> &stripe_list,
    hostdevice_vector<gpu::EncChunk> &chunks,
    hostdevice_vector<gpu::StripeStream> &strm_desc) {
  std::vector<StripeInformation> stripes(stripe_list.size());

  size_t group = 0;
  size_t stripe_start = 0;
  for (size_t stripe_id = 0; stripe_id < stripe_list.size(); stripe_id++) {
    size_t stripe_group_end = group + stripe_list[stripe_id];

    for (gdf_size_type i = 0; i < num_columns; i++) {
      gpu::EncChunk *ck = &chunks[group * num_columns + i];
      for (int k = 0; k <= gpu::CI_DICTIONARY; k++) {
        int32_t strm_id = ck->strm_id[k];
        if (strm_id >= num_columns + 1) {
          gpu::StripeStream *ss = &strm_desc[stripe_id * num_data_streams +
                                             strm_id - (num_columns + 1)];
          ss->stream_size = 0;
          ss->first_chunk_id = (uint32_t)(group * num_columns + i);
          ss->num_chunks = (uint32_t)(stripe_group_end - group);
          ss->column_id = i;
          ss->strm_type = (uint8_t)k;
        }
      }
    }

    group = stripe_group_end;
    size_t stripe_end = std::min(group * ROW_INDEX_STRIDE, num_rows);
    stripes[stripe_id].numberOfRows = (uint32_t)(stripe_end - stripe_start);
    stripe_start = stripe_end;
  }

  CUDA_TRY(cudaMemcpyAsync(strm_desc.device_ptr(), strm_desc.host_ptr(),
                           strm_desc.memory_size(), cudaMemcpyHostToDevice));
  CUDA_TRY(CompactOrcDataStreams(strm_desc.device_ptr(), chunks.device_ptr(),
                                 strm_desc.size(), num_columns));
  CUDA_TRY(cudaMemcpyAsync(strm_desc.host_ptr(), strm_desc.device_ptr(),
                           strm_desc.memory_size(), cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaMemcpyAsync(chunks.host_ptr(), chunks.device_ptr(),
                           chunks.memory_size(), cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaStreamSynchronize(0));

  return stripes;
}

writer::writer(std::string filepath, writer_options const& options)
    : impl_(std::make_unique<Impl>(filepath, options)) {}

void writer::write_all(const cudf::table& table) { impl_->write(table); }

writer::~writer() = default;

}  // namespace orc
}  // namespace io
}  // namespace cudf
