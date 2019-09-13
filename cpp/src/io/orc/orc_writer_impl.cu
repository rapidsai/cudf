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
 * @brief Function that copies all chunks belonging to a stream
 **/
size_t gather_stripe_stream(uint8_t *dst, const gpu::EncChunk *chunks,
                            gpu::StreamIndexType strm_type,
                            size_t num_rowgroups, size_t num_columns) {
  size_t dst_pos = 0;
  for (size_t g = 0; g < num_rowgroups; g++) {
    const gpu::EncChunk *ck = &chunks[g * num_columns];
    uint32_t chunk_len = ck->strm_len[strm_type];
    if (ck->streams[strm_type]) {
      CUDA_TRY(cudaMemcpyAsync(dst + dst_pos, ck->streams[strm_type], chunk_len,
                               cudaMemcpyDeviceToHost));
    } else {
      memset(dst + dst_pos, 0, chunk_len);
    }
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
    std::vector<int> str_col_ids;
    std::vector<int> str_col_map;
    std::vector<uint32_t> stripe_list;
    size_t num_rowgroups, num_chunks, num_string_columns, num_dict_chunks, rleout_bfr_size, strdata_bfr_size;
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
    str_col_map.resize(num_columns);
    num_string_columns = 0;
    for (int i = 0; i < num_columns; i++)
    {
        ff.numberOfRows = std::max(ff.numberOfRows, (uint64_t)columns[i]->size);
        sf.streams[1+i].column = 1 + i;
        sf.streams[1+i].kind = ROW_INDEX;
        sf.streams[1+i].length = 0;
        str_col_map[i] = (int)num_string_columns;
        if (columns[i]->dtype == GDF_STRING || columns[i]->dtype == GDF_STRING_CATEGORY)
        {
            str_col_ids.push_back(i);
            num_string_columns++;
        }
    }
    num_rowgroups = (ff.numberOfRows + ff.rowIndexStride - 1) / ff.rowIndexStride;
    num_chunks = num_rowgroups * num_columns;
    num_dict_chunks = num_rowgroups * num_string_columns;
    hostdevice_vector<gpu::DictionaryChunk> dict(num_dict_chunks);
    device_buffer<uint32_t> dict_index(num_string_columns * ff.numberOfRows);
    device_buffer<uint32_t> dict_data(num_string_columns * ff.numberOfRows);
    std::vector<device_buffer<std::pair<const char*,size_t>>> str_indices(num_string_columns);
    if (num_dict_chunks != 0)
    {
        // Create initial per-rowgroup string dictionaries
        for (int i = 0; i < (int)num_string_columns; i++)
        {
            const gdf_column *col = columns[str_col_ids[i]];

            str_indices[i].resize(col->size);
            if (col->dtype == GDF_STRING) {
              auto *str = static_cast<NVStrings *>(col->data);
              CUDF_EXPECTS(str->create_index(str_indices[i].data()) == 0,
                           "Cannot retrieve nvcategory string pairs");
            } else if (col->dtype == GDF_STRING_CATEGORY) {
              auto *cat = static_cast<NVCategory *>(col->dtype_info.category);
              CUDF_EXPECTS(cat->create_index(str_indices[i].data()) == 0,
                           "Cannot retrieve nvcategory string pairs");
            } else {
              CUDF_FAIL("Expected a string-type column");
            }

            for (int g = 0; g < (int)num_rowgroups; g++)
            {
                gpu::DictionaryChunk *ck = &dict[g * num_string_columns + i];
                ck->valid_map_base = (col->null_count != 0) ? reinterpret_cast<const uint32_t *>(col->valid) : nullptr;
                ck->column_data_base = str_indices[i].data();
                ck->dict_data = dict_data.data() + i * ff.numberOfRows + g * ff.rowIndexStride;
                ck->dict_index = dict_index.data() + i * ff.numberOfRows; // Indexed by absolute row
                ck->start_row = g * ff.rowIndexStride;
                ck->num_rows = std::min(ff.rowIndexStride, (uint32_t)std::max(col->size - (int)ck->start_row, 0));
                ck->num_strings = 0;
                ck->string_char_count = 0;
                ck->num_dict_strings = 0;
                ck->dict_char_count = 0;
            }
        }
        // Build string dictionaries and update character data and dictionary sizes
        CUDA_TRY(cudaMemcpyAsync(dict.device_ptr(), dict.host_ptr(),
                                 dict.memory_size(), cudaMemcpyHostToDevice));
        CUDA_TRY(InitDictionaryIndices(dict.device_ptr(),
                                       (uint32_t)num_string_columns,
                                       (uint32_t)num_rowgroups));
        CUDA_TRY(cudaMemcpyAsync(dict.host_ptr(), dict.device_ptr(),
                                 dict.memory_size(), cudaMemcpyDeviceToHost));
        CUDA_TRY(cudaStreamSynchronize(0));
    }
    // Decide stripe boundaries early on, based on uncompressed size
    for (size_t g = 0, stripe_start = 0, stripe_size = 0; g < num_rowgroups; g++)
    {
        const unsigned int kMaxStripeSize = 64 * 1024 * 1024;   // TBD: Stripe size hardcoded to 64MB
        const unsigned int max_stripe_rows = (num_string_columns) ? 1000000 : 5000000; // Limits dictionary size
        size_t rowgroup_size = 0;
        for (size_t i = 0; i < (size_t)num_columns; i++)
        {
            size_t dtype_len = 0;
            switch (columns[i]->dtype)
            {
            default:
            case GDF_INT8:
            case GDF_BOOL8:
                dtype_len = 1;
                break;
            case GDF_INT16:
                dtype_len = 2;
                break;
            case GDF_INT32:
            case GDF_FLOAT32:
            case GDF_DATE32:
            case GDF_CATEGORY:
                dtype_len = 4;
                break;
            case GDF_INT64:
            case GDF_FLOAT64:
            case GDF_DATE64:
            case GDF_TIMESTAMP:
                dtype_len = 8;
                break;
            case GDF_STRING:
            case GDF_STRING_CATEGORY:
                dtype_len = 1; // Count 1 byte for length
                rowgroup_size += dict[g * num_string_columns + str_col_map[i]].string_char_count;
                break;
            }
            rowgroup_size += dtype_len * ff.rowIndexStride;
        }
        if (g > stripe_start && (stripe_size + rowgroup_size > kMaxStripeSize || (g + 1 - stripe_start) * ff.rowIndexStride > max_stripe_rows) )
        {
            stripe_list.push_back((uint32_t)(g - stripe_start));
            stripe_start = g;
            stripe_size = 0;
        }
        stripe_size += rowgroup_size;
        if (g + 1 == num_rowgroups)
        {
            stripe_list.push_back((uint32_t)(num_rowgroups - stripe_start));
        }
    }

    // Build stripe-level dictionaries
    size_t num_stripes = stripe_list.size();
    size_t num_stripe_dict = num_stripes * num_string_columns;
    hostdevice_vector<gpu::StripeDictionary> stripe_dict(num_stripe_dict);
    if (num_dict_chunks != 0)
    {
        for (size_t i = 0; i < num_string_columns; i++)
        {
            size_t direct_cost = 0, dictionary_cost = 0;
            for (size_t j = 0, g = 0; j < num_stripes; j++)
            {
                uint32_t num_chunks = stripe_list[j];
                gpu::StripeDictionary *sd = &stripe_dict[j * num_string_columns + i];
                sd->column_data_base = dict[i].column_data_base;
                sd->dict_data = dict[g * num_string_columns + i].dict_data;
                sd->dict_index = dict_index.data() + i * ff.numberOfRows; // Indexed by absolute row
                sd->column_id = str_col_ids[i];
                sd->start_chunk = (uint32_t)g;
                sd->num_chunks = num_chunks;
                sd->num_strings = 0;
                for (size_t k = g; k < g + num_chunks; k++)
                {
                    direct_cost += dict[k * num_string_columns + i].string_char_count;
                    dictionary_cost += dict[k * num_string_columns + i].dict_char_count;
                    sd->num_strings += dict[k * num_string_columns + i].num_dict_strings;
                }
                dictionary_cost += sd->num_strings;
                sd->dict_char_count = 0;
                g += num_chunks;
            }
            // Early decision to disable dictionary if it doesn't look good at the chunk level
            if (dictionary_cost >= direct_cost)
            {
                for (size_t j = 0; j < num_stripes; j++)
                    stripe_dict[j * num_string_columns + i].dict_data = nullptr;
            }
        }
        CUDA_TRY(
            cudaMemcpyAsync(stripe_dict.device_ptr(), stripe_dict.host_ptr(),
                            stripe_dict.memory_size(), cudaMemcpyHostToDevice));
        CUDA_TRY(BuildStripeDictionaries(
            stripe_dict.device_ptr(), stripe_dict.host_ptr(), dict.device_ptr(),
            (uint32_t)num_stripes, (uint32_t)num_rowgroups,
            (uint32_t)num_string_columns));
        CUDA_TRY(
            cudaMemcpyAsync(stripe_dict.host_ptr(), stripe_dict.device_ptr(),
                            stripe_dict.memory_size(), cudaMemcpyDeviceToHost));
        CUDA_TRY(cudaStreamSynchronize(0));
    }
    // Initialize streams
    sf.columns[0].kind = DIRECT;
    sf.columns[0].dictionarySize = 0;
    strdata_bfr_size = 0;
    for (int i = 0; i < num_columns; i++)
    {
        TypeKind kind = to_orckind(columns[i]->dtype);
        StreamKind data_kind = DATA, data2_kind = LENGTH;
        ColumnEncodingKind encoding_kind = DIRECT;
        int64_t present_stream_size = 0, data_stream_size = 0, data2_stream_size = 0, dict_stream_size = 0;

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
            // Pass through if no nulls (no RLE encoding for floating point)
            data_stream_size = (columns[i]->null_count) ? ((ff.rowIndexStride + 0x1ff) >> 9) * (512 * 4 + 2) : INT64_C(-1);
            break;
        case INT:
        case DATE:
            data_stream_size = ((ff.rowIndexStride + 0x1ff) >> 9) * (512 * 4 + 2);
            encoding_kind = DIRECT_V2;
            break;
        case DOUBLE:
            // Pass through if no nulls (no RLE encoding for floating point)
            data_stream_size = (columns[i]->null_count) ? ((ff.rowIndexStride + 0x1ff) >> 9) * (512 * 8 + 2) : INT64_C(-1);
            break;
        case LONG:
            data_stream_size = ((ff.rowIndexStride + 0x1ff) >> 9) * (512 * 8 + 2);
            encoding_kind = DIRECT_V2;
            break;
        case STRING: {
            uint32_t scol = str_col_map[i], dict_bits;
            size_t direct_data_size = 0, dict_data_size = 0, dict_strings = 0, dict_lengths_div512 = 0, dict_overhead;
            bool enable_dictionary = true;
            for (size_t stripe_id = 0, g = 0; stripe_id < stripe_list.size(); stripe_id++)
            {
                const gpu::StripeDictionary *sd = &stripe_dict[stripe_id * num_string_columns + scol];
                enable_dictionary = (enable_dictionary && sd->dict_data != nullptr);
                dict_strings += sd->num_strings;
                dict_lengths_div512 += (sd->num_strings + 0x1ff) >> 9;
                dict_data_size += sd->dict_char_count;
                for (uint32_t k = 0; k < stripe_list[stripe_id]; k++, g++)
                    direct_data_size += dict[g * num_string_columns + scol].string_char_count;
            }
            for (dict_bits = 1; dict_bits < 32; dict_bits <<= 1)
            {
                if (dict_strings <= (1ull << dict_bits))
                    break;
            }
            dict_overhead = (dict_bits * (columns[i]->size - columns[i]->null_count) + 7) >> 3;
            //printf("col%d: dict_data_size(%zd strings) = %zd+%zd, direct_size = %zd\n", i, dict_strings, dict_data_size, dict_overhead, direct_data_size);
            if (enable_dictionary && dict_data_size + dict_overhead < direct_data_size)
            {
                // Dictionary encoding
                data_stream_size = ((ff.rowIndexStride + 0x1ff) >> 9) * (512 * 4 + 2);
                data2_stream_size = dict_lengths_div512 * (512 * 4 + 2);
                dict_stream_size = std::max(dict_data_size, (size_t)1);
                encoding_kind = DICTIONARY_V2;
            }
            else
            {
                // Direct encoding
                data_stream_size = std::max(direct_data_size, (size_t)1);
                data2_stream_size = ((ff.rowIndexStride + 0x1ff) >> 9) * (512 * 4 + 2);
                encoding_kind = DIRECT_V2;
            }
            break;
          }
        case TIMESTAMP:
            data2_stream_size = data_stream_size = ((ff.rowIndexStride + 0x1ff) >> 9) * (512*4 + 2);
            data2_kind = SECONDARY;
            has_timestamp_column = true;
            encoding_kind = DIRECT_V2;
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
            sf.streams[data_stream_id].length = std::max<int64_t>(data_stream_size, 0);
            stream_ids[i * gpu::CI_NUM_STREAMS + gpu::CI_DATA] = (int32_t)data_stream_id;
        }
        if (data2_stream_size != 0)
        {
            uint32_t data_stream_id = (uint32_t)sf.streams.size();
            sf.streams.resize(data_stream_id + 1);
            sf.streams[data_stream_id].column = 1 + i;
            sf.streams[data_stream_id].kind = data2_kind;
            sf.streams[data_stream_id].length = std::max<int64_t>(data2_stream_size, 0);
            stream_ids[i * gpu::CI_NUM_STREAMS + gpu::CI_DATA2] = (int32_t)data_stream_id;
        }
        if (dict_stream_size != 0)
        {
            uint32_t dict_stream_id = (uint32_t)sf.streams.size();
            sf.streams.resize(dict_stream_id + 1);
            sf.streams[dict_stream_id].column = 1 + i;
            sf.streams[dict_stream_id].kind = DICTIONARY_DATA;
            sf.streams[dict_stream_id].length = dict_stream_size;
            stream_ids[i * gpu::CI_NUM_STREAMS + gpu::CI_DICTIONARY] = (int32_t)dict_stream_id;
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
    strm_offsets.resize(sf.streams.size());
    rleout_bfr_size = 0;
    for (size_t i = 0; i < sf.streams.size(); i++)
    {
        if (((sf.streams[i].kind == DICTIONARY_DATA || sf.streams[i].kind == LENGTH) && sf.columns[sf.streams[i].column].kind == DICTIONARY_V2)
         || (sf.streams[i].kind == DATA && ff.types[sf.streams[i].column].kind == STRING && sf.columns[sf.streams[i].column].kind == DIRECT_V2))
        {
            strm_offsets[i] = strdata_bfr_size;
            strdata_bfr_size += sf.streams[i].length;
        }
        else
        {
            strm_offsets[i] = rleout_bfr_size;
            rleout_bfr_size += (sf.streams[i].length * num_rowgroups + 7) & ~7;
        }
    }
    strdata_bfr_size = (strdata_bfr_size + 7) & ~7;
    hostdevice_vector<gpu::EncChunk> chunks(num_chunks);
    device_buffer<uint8_t> rleout_bfr_dev(rleout_bfr_size + strdata_bfr_size);
    for (size_t j = 0, stripe_start = 0, stripe_id = 0; j < num_rowgroups; j++)
    {
        for (size_t i = 0; i < (size_t)num_columns; i++)
        {
            gpu::EncChunk *ck = &chunks[j * num_columns + i];
            ck->valid_map_base = (const uint32_t *)columns[i]->valid;
            ck->column_data_base = columns[i]->data;
            ck->start_row = (uint32_t)(j * ff.rowIndexStride);
            ck->num_rows = (uint32_t)std::min((uint32_t)ff.rowIndexStride, (uint32_t)(ff.numberOfRows - ck->start_row));
            ck->valid_rows = columns[i]->size;
            ck->encoding_kind = (uint8_t)sf.columns[1+i].kind;
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
            switch (ck->type_kind)
            {
            case SHORT:
                ck->dtype_len = 2;
                break;
            case INT:
            case FLOAT:
            case DATE:
                ck->dtype_len = 4;
                break;
            case LONG:
            case DOUBLE:
            case TIMESTAMP:
                ck->dtype_len = 8;
                break;
            case STRING:
                ck->column_data_base = str_indices[str_col_map[i]].data();
                // fall-through
            default:
                ck->dtype_len = 1;
                if (ck->encoding_kind == DICTIONARY_V2)
                    ck->column_data_base = stripe_dict[stripe_id * num_string_columns + str_col_map[i]].dict_index;
                break;
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
    // Encode column data
    CUDA_TRY(EncodeOrcColumnData(chunks.device_ptr(), (uint32_t)num_columns,
                                 (uint32_t)num_rowgroups));
    CUDA_TRY(cudaMemcpyAsync(chunks.host_ptr(), chunks.device_ptr(),
                             chunks.memory_size(), cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaStreamSynchronize(0));
    // Initialize stripe data in file footer
    ff.stripes.resize(stripe_list.size());
    for (size_t group = 0, stripe_id = 0, stripe_start = 0; stripe_id < stripe_list.size(); stripe_id++)
    {
        size_t stripe_data_length = 0, stripe_end;
        for (uint32_t g = 0; g < stripe_list[stripe_id]; g++)
        {
            for (int i = 0; i < num_columns; i++)
            {
                gpu::EncChunk *ck = &chunks[group * num_columns + i];
                for (int k = 0; k < gpu::CI_NUM_STREAMS; k++)
                {
                    stripe_data_length += ck->strm_len[k];
                }
            }
            group++;
        }
        stripe_end = std::min((uint64_t)group * ff.rowIndexStride, ff.numberOfRows);
        ff.stripes[stripe_id].dataLength = stripe_data_length;
        ff.stripes[stripe_id].numberOfRows = (uint32_t)(stripe_end - stripe_start);
        stripe_start = stripe_end;
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
                max_size = std::max(max_size, (int)chunks[group * num_columns + strm - 1].strm_len[gpu::CI_DICTIONARY]);
            }
            if (kind == STRING && sf.columns[strm].kind == DICTIONARY_V2)
            {
                kind = INT; // Change string dictionary to int from index point of view
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
            if (ck->strm_id[gpu::CI_DICTIONARY] > 0)
            {
                size_t len = gather_stripe_stream(stream_io_buf.get(), ck, gpu::CI_DICTIONARY, groups_in_stripe, num_columns);
                outfile_.write(reinterpret_cast<char*>(stream_io_buf.get()), len);
                ff.stripes[stripe_id].dataLength += len;
                sf.streams[ck->strm_id[gpu::CI_DICTIONARY]].length = len;
            }
            sf.columns[1 + i].dictionarySize = (sf.columns[1 + i].kind == DICTIONARY_V2) ? stripe_dict[stripe_id * num_string_columns + str_col_map[i]].num_strings : 0;
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
