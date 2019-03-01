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

#include "cudf.h"
#include "utilities/error_utils.h"
#include "io/comp/gpuinflate.h"

#include "rmm/rmm.h"
#include "rmm/thrust_rmm_allocator.h"
#include <cuda_runtime.h>

#include "parquet.h"
#include "parquet_gpu.h"

#include <iostream>
#include <cstring>

#define RMM_ALLOC_HOST(ptr, sz)     cudaMallocHost(ptr, sz)
#define RMM_FREE_HOST(ptr)          cudaFreeHost(ptr)

static constexpr int NUM_SUPPORTED_CODECS = 2;
static const parquet::Compression g_supportedCodecs[NUM_SUPPORTED_CODECS] = { parquet::GZIP, parquet::SNAPPY };
static const char * const g_supportedCodecsNames[NUM_SUPPORTED_CODECS] = { "GZIP", "SNAPPY" };

#define DUMP_PERF   1
#if DUMP_PERF
#include <chrono>
class perf_chk
{
private:
    std::chrono::high_resolution_clock::time_point m_tStart, m_tEnd;
    double *m_accd;
public:
    perf_chk(double *accd) { m_accd = accd; init(); }
    void init() { m_tStart = std::chrono::high_resolution_clock::now(); }
    ~perf_chk() { m_tEnd = std::chrono::high_resolution_clock::now(); *m_accd += 1.e-6 * (double)std::chrono::duration_cast<std::chrono::microseconds>(m_tEnd - m_tStart).count(); } // WTF
};
#define PERF_SAMPLE_START(acc)  { perf_chk _local_perf(acc); 
#define PERF_SAMPLE_END()       }
#define PERF_SAMPLE_RESET()     _local_perf.init()
#endif // DUMP_PERF

uint8_t *LoadFile(const char *input_fname, size_t *len)
{
    size_t file_size;
    FILE *fin = nullptr;
    uint8_t *raw = nullptr;

    *len = 0;
    fin = (input_fname) ? fopen(input_fname, "rb") : nullptr;
    if (!fin)
    {
        printf("Could not open \"%s\"\n", input_fname);
        return nullptr;
    }
    fseek(fin, 0, SEEK_END);
    file_size = ftell(fin);
    fseek(fin, 0, SEEK_SET);
    if (file_size <= 0)
    {
        printf("Invalid file size: %zd\n", file_size);
        fclose(fin);
        return nullptr;
    }
    *len = file_size;
    raw = new uint8_t[file_size];
    if (raw)
    {
        if (file_size != fread(raw, 1, file_size, fin))
        {
            printf("Failed to read %zd bytes\n", file_size);
            delete[] raw;
            raw = nullptr;
        }
    }
    fclose(fin);
    return raw;
}

// TODO: Move into metadata or schema class
std::string to_dot_string(std::vector<std::string> const &path_in_schema) {
  size_t n = path_in_schema.size();
  std::string s = (n > 0) ? path_in_schema[0] : "";
  for (size_t i = 1; i < n; i++) {
    s += '.';
    s += path_in_schema[i];
  }
  return s;
}

// TODO: Move into metadata or schema class
std::string get_index_col(parquet::FileMetaData md) {
  const auto it =
      std::find_if(md.key_value_metadata.begin(), md.key_value_metadata.end(),
                   [](const auto& item) { return item.key == "pandas"; });

  if (it != md.key_value_metadata.end()) {
    const auto pos = it->value.find("index_columns");

    if (pos != std::string::npos) {
      const auto begin = it->value.find('[', pos);
      const auto end = it->value.find(']', begin);
      if ((end - begin) > 4) {
        return it->value.substr(begin + 2, end - begin - 3);
      }
    }
  }

  return "";
}

/**---------------------------------------------------------------------------*
 * @brief Reads Apache Parquet-formatted data and returns an allocated array of
 * gdf_columns.
 * 
 * @param[in,out] args Structure containing input and output args 
 * 
 * @return gdf_error GDF_SUCCESS if successful, otherwise an error code.
 *---------------------------------------------------------------------------**/
gdf_error read_parquet(pq_read_arg *args) {
    uint8_t *raw = nullptr;
    size_t raw_size = 0;
    const parquet::file_header_s *fheader = nullptr;
    const parquet::file_ender_s *fender = nullptr;
    parquet::FileMetaData file_md;
    parquet::CPReader cp;
    std::vector<std::string> col_names; // Selected columns
    std::vector<const parquet::ColumnChunk*> chunk_map;   // Map from chunk id to parquet chunk
    std::vector<int> chunk_col; // Map from chunk to gdf column
    // GPU data
    size_t total_pages = 0, num_compressed_pages, total_decompressed_size;
    size_t compressed_page_cnt[NUM_SUPPORTED_CODECS];
    parquet::gpu::ColumnChunkDesc *chunk_desc = nullptr, *chunk_desc_dev = nullptr;
    parquet::gpu::PageInfo *page_index = nullptr, *page_index_dev = nullptr;
    size_t total_str_indices = 0;
    parquet::gpu::nvstrdesc_s *str_dict_index = nullptr;
    uint8_t *decompressed_pages = nullptr;
    gdf_column **columns = nullptr;
    int max_num_columns = 0, num_columns = 0, max_num_chunks = 0, num_chunks = 0;
    int* index_col = nullptr;
    std::string index_col_name = "";

    raw = LoadFile(args->source, &raw_size);
    if (!raw || raw_size < sizeof(parquet::file_header_s) + sizeof(parquet::file_ender_s))
    {
        printf("Failed to open parquet file \"%s\"\n", args->source);
        return GDF_FILE_ERROR;
    }
    fheader = (const parquet::file_header_s *)raw;
    fender = (const parquet::file_ender_s *)(raw + raw_size - sizeof(parquet::file_ender_s));
    if (fheader->magic != PARQUET_MAGIC || fender->magic != PARQUET_MAGIC)
    {
        printf("Invalid parquet magic (hdr=0x%x, end=0x%x, expected 0x%x)\n", fheader->magic, fender->magic, PARQUET_MAGIC);
        goto error_exit;
    }
    if ((fender->footer_len > raw_size - sizeof(parquet::file_header_s) - sizeof(parquet::file_ender_s))
     || (fender->footer_len <= 0))
    {
        printf("Invalid parquet footer length (%d bytes)\n", fender->footer_len);
        goto error_exit;
    }
    printf("Parquet file footer: %d bytes @ 0x%zx\n", fender->footer_len, raw_size - fender->footer_len - sizeof(parquet::file_ender_s));
    cp.init(raw + raw_size - fender->footer_len - sizeof(parquet::file_ender_s), fender->footer_len);

    if (!cp.read(&file_md))
    {
        printf("Error parsing file metadata\n");
    }
    if (!cp.InitSchema(&file_md))
    {
        printf("Failed to initialize schema\n");
    }
    printf(" parquet header byte count = %zd/%d\n", cp.bytecount(), fender->footer_len);
    printf(" version = %d\n", file_md.version);
    printf(" created_by = \"%s\"\n", file_md.created_by.c_str());
    printf(" schema (%zd entries):\n", file_md.schema.size());
    for (size_t i = 0; i < file_md.schema.size(); i++)
    {
        printf("  [%zd] type=%d, name=\"%s\", num_children=%d, rep_type=%d, max_def_lvl=%d, max_rep_lvl=%d\n", i, file_md.schema[i].type, file_md.schema[i].name.c_str(), file_md.schema[i].num_children, file_md.schema[i].repetition_type, file_md.schema[i].max_definition_level, file_md.schema[i].max_repetition_level);
    }
    printf(" num_rows = %zd\n", (size_t)file_md.num_rows);
    max_num_columns = file_md.row_groups.size() ? (int)file_md.row_groups[0].columns.size() : 0;
    printf(" num_columns = %d\n", max_num_columns);
    if (max_num_columns <= 0)
    {
        goto error_exit;
    }

    // Use user-specified column names
    // Otherwise generate names from schema path of columns in the 1st row group
    if (args->use_cols) {
      col_names.reserve(args->use_cols_len);
      for (int i = 0; i < args->use_cols_len; ++i) {
          col_names.emplace_back(args->use_cols[i]);
      }
    } else {
      col_names.resize(max_num_columns);
      for (int i = 0; i < max_num_columns; ++i) {
          col_names.emplace_back(to_dot_string(file_md.row_groups[0].columns[i].meta_data.path_in_schema));
      }
    }
    index_col_name = get_index_col(file_md);
    if (index_col_name != "") {
      index_col = (int *)malloc(sizeof(int));
      col_names.emplace_back(index_col_name);
    }

    num_columns = (int)col_names.size();
    printf("Selected %d columns:\n", num_columns);
    columns = (gdf_column **)malloc(num_columns * sizeof(gdf_column *));
    memset(columns, 0, num_columns * sizeof(gdf_column *));
    for (int i = 0; i < num_columns; i++)
    {
        const parquet::ColumnChunk *col_md;
        int col_idx, schema_idx;
        printf("[%d] \"%s\"\n", i, col_names[i].c_str());
        for (col_idx = 0; col_idx < max_num_columns; col_idx++)
        {
            std::string s = to_dot_string(file_md.row_groups[0].columns[col_idx].meta_data.path_in_schema);
            if (s == col_names[i])
            {
                if (s == index_col_name) {
                    *index_col = i;
                }
                break;
            }
        }
        if (col_idx >= max_num_columns)
        {
            printf(" column not found!\n");
            goto error_exit;
        }
        col_md = &file_md.row_groups[0].columns[col_idx];
        schema_idx = col_md->schema_idx;
        if (schema_idx < 0)
        {
            printf(" column not found in schema!\n");
            goto error_exit;
        }
        columns[i] = (gdf_column *)malloc(sizeof(gdf_column));
        memset(columns[i], 0, sizeof(gdf_column));
        columns[i]->size = static_cast<gdf_size_type>(file_md.num_rows);
        // TODO: Use the optional LogicalType value in the schema to refine the GDF datatype (dates, timestamps, etc)
        switch(file_md.schema[schema_idx].type)
        {
        case parquet::INT32:
            columns[i]->dtype = GDF_INT32;
            break;
        case parquet::INT64:
            columns[i]->dtype = GDF_INT64;
            break;
        case parquet::FLOAT:
            columns[i]->dtype = GDF_FLOAT32;
            break;
        case parquet::DOUBLE:
            columns[i]->dtype = GDF_FLOAT64;
            break;
        case parquet::BYTE_ARRAY:
            columns[i]->dtype = GDF_STRING;
            break;
        default:
            printf("Unsupported data type (%d)\n", file_md.schema[schema_idx].type);
            goto error_exit;
        }
        columns[i]->col_name = (char *)malloc(col_names[i].length() + 1);
        strcpy(columns[i]->col_name, col_names[i].c_str());
    }
    max_num_chunks = static_cast<int32_t>(file_md.row_groups.size() * num_columns);
    RMM_ALLOC_HOST((void **)&chunk_desc, sizeof(parquet::gpu::ColumnChunkDesc) * max_num_chunks);
    RMM_ALLOC((void **)&chunk_desc_dev, sizeof(parquet::gpu::ColumnChunkDesc) * max_num_chunks, 0);
    if (!(chunk_desc && chunk_desc_dev))
        goto error_exit;

    // Count and initialize gpu chunk description structures
    printf(" row_groups (%zd entries):\n", file_md.row_groups.size());
    chunk_map.resize(max_num_chunks);
    chunk_col.resize(max_num_chunks);
    for (size_t i = 0, row = 0; i < file_md.row_groups.size(); i++)
    {
        const parquet::RowGroup *g = &file_md.row_groups[i];
        printf("  [%zd] total_size=%zd, %zd rows, %zd columns:\n", i, (size_t)g->total_byte_size, (size_t)g->num_rows, g->columns.size());
        for (size_t j = 0; j < g->columns.size(); j++)
        {
            const parquet::ColumnChunk *col = &g->columns[j];
            std::string name = to_dot_string(col->meta_data.path_in_schema);
            for (int k = 0; k < num_columns; k++)
            {
                if (name == columns[k]->col_name)
                {
                    if (num_chunks < max_num_chunks)
                    {
                        parquet::gpu::ColumnChunkDesc *chunk = &chunk_desc[num_chunks];
                        size_t first_page_offset = (size_t)col->meta_data.data_page_offset;
                        if (col->meta_data.dictionary_page_offset != 0)
                        {
                            first_page_offset = std::min(first_page_offset, (size_t)col->meta_data.dictionary_page_offset);
                        }
                        chunk->compressed_data = nullptr;
                        chunk->compressed_size = col->meta_data.total_compressed_size;
                        chunk->num_values = col->meta_data.num_values;
                        chunk->start_row = row;
                        chunk->num_rows = static_cast<uint32_t>(g->num_rows);
                        chunk->max_def_level = (int16_t)file_md.schema[col->schema_idx].max_definition_level;
                        chunk->max_rep_level = (int16_t)file_md.schema[col->schema_idx].max_repetition_level;
                        chunk->def_level_bits = (uint8_t)cp.NumRequiredBits(file_md.schema[col->schema_idx].max_definition_level);
                        chunk->rep_level_bits = (uint8_t)cp.NumRequiredBits(file_md.schema[col->schema_idx].max_repetition_level);
                        chunk->data_type = (uint8_t)file_md.schema[col->schema_idx].type | (uint16_t)(file_md.schema[col->schema_idx].type_length << 3);
                        chunk->num_data_pages = 0;
                        chunk->num_dict_pages = 0;
                        chunk->max_num_pages = 0;
                        chunk->page_info = nullptr;
                        chunk->str_dict_index = nullptr;
                        chunk->valid_map_base = nullptr;
                        chunk->column_data_base = nullptr;
                        if (col->meta_data.total_compressed_size > 0)
                        {
                            RMM_ALLOC((void **)&chunk->compressed_data, col->meta_data.total_compressed_size, 0);
                            if (!chunk->compressed_data)
                                goto error_exit;
                            cudaMemcpyAsync(chunk->compressed_data, raw + first_page_offset, col->meta_data.total_compressed_size, cudaMemcpyHostToDevice);
                        }
                        chunk_map[num_chunks] = col;
                        chunk_col[num_chunks] = k;
                        num_chunks++;
                    }
                    else
                    {
                        printf("Too many chunks!!!\n");
                    }
                    break;
                }
            }
            printf("   col%zd \"%s\"@%-9zd, type=%d, codec=%d, num_values=%zd, schema_idx=%d\n", j, col->file_path.c_str(), (size_t)col->file_offset, col->meta_data.type, col->meta_data.codec, (size_t)col->meta_data.num_values, col->schema_idx);
            printf("      path in schema: \"%s\"\n", name.c_str());
            if (col->meta_data.encodings.size())
            {
                printf("      encodings={");
                for (size_t k=0; k<col->meta_data.encodings.size(); k++)
                    printf("%d,", col->meta_data.encodings[k]);
                printf("},\n");
            }
            printf("      total uncompressed size = %zd, compressed to %zd\n", (size_t)col->meta_data.total_uncompressed_size, (size_t)col->meta_data.total_compressed_size);
            if (col->offset_index_length || col->column_index_length)
            {
                printf("       index offset:%d@0x%zx, column:%d@0x%zx\n", col->offset_index_length, (size_t)col->offset_index_offset, col->column_index_length, (size_t)col->column_index_offset);
            }
            printf("      data page offset @%zd, index @%zd, dictionary @%zd\n", (size_t)col->meta_data.data_page_offset, (size_t)col->meta_data.index_page_offset, (size_t)col->meta_data.dictionary_page_offset);
        }
        row += g->num_rows;
    }
    // Copy column chunks to GPU and count data pages
    cudaMemcpyAsync(chunk_desc_dev, chunk_desc, sizeof(parquet::gpu::ColumnChunkDesc) * num_chunks, cudaMemcpyHostToDevice);
    DecodePageHeaders(chunk_desc_dev, num_chunks);
    cudaMemcpyAsync(chunk_desc, chunk_desc_dev, sizeof(parquet::gpu::ColumnChunkDesc) * num_chunks, cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(0);
    printf("[GPU] %d chunks:\n", num_chunks);
    total_pages = 0;
    for (size_t c = 0; c < num_chunks; c++)
    {
        printf("[%zd] %d rows, %d data pages, %d dictionary pages, data_type=0x%x\n", c, chunk_desc[c].num_rows, chunk_desc[c].num_data_pages, chunk_desc[c].num_dict_pages, chunk_desc[c].data_type);
        total_pages += chunk_desc[c].num_data_pages + chunk_desc[c].num_dict_pages;
    }

    // Decode page headers again, this time storing page info
    RMM_ALLOC((void **)&page_index_dev, sizeof(parquet::gpu::PageInfo) * total_pages, 0);
    RMM_ALLOC_HOST((void **)&page_index, sizeof(parquet::gpu::PageInfo) * total_pages);
    if (!(page_index && page_index_dev))
        goto error_exit;
    for (int32_t chunk = 0, page_cnt = 0; chunk < num_chunks; chunk++)
    {
        chunk_desc[chunk].max_num_pages = chunk_desc[chunk].num_data_pages + chunk_desc[chunk].num_dict_pages;
        chunk_desc[chunk].page_info = &page_index_dev[page_cnt];
        page_cnt += chunk_desc[chunk].max_num_pages;
    }
    cudaMemcpyAsync(chunk_desc_dev, chunk_desc, sizeof(parquet::gpu::ColumnChunkDesc) * num_chunks, cudaMemcpyHostToDevice);
    DecodePageHeaders(chunk_desc_dev, num_chunks);
    cudaMemcpyAsync(page_index, page_index_dev, sizeof(parquet::gpu::PageInfo) * total_pages, cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(0);
    printf("[GPU] %d pages:\n", (int)total_pages);
    for (size_t i = 0; i < total_pages; i++)
    {
        printf("[%zd] ck=%d, row=%d, flags=%d, num_values=%d, encoding=%d, size=%d\n", i, page_index[i].chunk_idx, page_index[i].chunk_row, page_index[i].flags, page_index[i].num_values, page_index[i].encoding, page_index[i].uncompressed_page_size);
    }

    // Decompress pages that are compressed
    memset(&compressed_page_cnt, 0, sizeof(compressed_page_cnt));
    num_compressed_pages = 0;
    total_decompressed_size = 0;
    for (int i = 0; i < NUM_SUPPORTED_CODECS; i++)
    {
        parquet::Compression codec = g_supportedCodecs[i];
        size_t codec_page_cnt = 0, page_cnt = 0;
        for (int chunk = 0; chunk < num_chunks; chunk++)
        {
            int32_t max_num_pages = chunk_desc[chunk].max_num_pages;
            if (chunk_map[chunk]->meta_data.codec == codec)
            {
                codec_page_cnt += max_num_pages;
                for (int k = 0; k < max_num_pages; k++)
                {
                    total_decompressed_size += page_index[page_cnt + k].uncompressed_page_size;
                }
            }
            page_cnt += max_num_pages;
        }
        if (codec_page_cnt != 0)
        {
            printf("[GPU] %s compression (%zd pages, %zd bytes)\n", g_supportedCodecsNames[i], codec_page_cnt, total_decompressed_size);
        }
        compressed_page_cnt[i] += codec_page_cnt;
        num_compressed_pages += codec_page_cnt;
    }
    
    if (num_compressed_pages > 0)
    {
        gpu_inflate_input_s *inflate_in = nullptr, *inflate_in_dev = nullptr;
        gpu_inflate_status_s *inflate_out = nullptr, *inflate_out_dev = nullptr;
        size_t decompressed_ofs = 0;
        int32_t comp_cnt = 0;
        double uncomp_time = 0;

        RMM_ALLOC_HOST((void **)&inflate_in, sizeof(gpu_inflate_input_s) * num_compressed_pages);
        RMM_ALLOC((void **)&inflate_in_dev, sizeof(gpu_inflate_input_s) * num_compressed_pages, 0);
        RMM_ALLOC_HOST((void **)&inflate_out, sizeof(gpu_inflate_status_s) * num_compressed_pages);
        RMM_ALLOC((void **)&inflate_out_dev, sizeof(gpu_inflate_status_s) * num_compressed_pages, 0);
        RMM_ALLOC((void **)&decompressed_pages, total_decompressed_size, 0);
#if PADUMP_ZLIB_BENCH
        uint8_t **zlib_src = new uint8_t *[num_compressed_pages];
        double zlib_time = 0;
#endif
        PERF_SAMPLE_START(&uncomp_time);
        for (int codec_idx = 0; codec_idx < NUM_SUPPORTED_CODECS; codec_idx++)
        {
            parquet::Compression codec = g_supportedCodecs[codec_idx];
            if (compressed_page_cnt[codec_idx] > 0)
            {
                int32_t start_pos = comp_cnt;

                // Fill in decompression in/out structures & update page ptr to point to the decompressed data
                for (int chunk = 0, page_cnt = 0; chunk < num_chunks; chunk++)
                {
                    if (chunk_map[chunk]->meta_data.codec == codec)
                    {
                        for (int k = 0; k < chunk_desc[chunk].max_num_pages; k++, comp_cnt++)
                        {
#if PADUMP_ZLIB_BENCH
                            ptrdiff_t src_ofs = page_index[page_cnt + k].compressed_page_data - chunk_desc[chunk].compressed_data;
                            size_t first_page_offset = (size_t)chunk_map[chunk]->meta_data.data_page_offset;
                            if (chunk_map[chunk]->meta_data.dictionary_page_offset != 0)
                            {
                                first_page_offset = std::min(first_page_offset, (size_t)chunk_map[chunk]->meta_data.dictionary_page_offset);
                            }
                            zlib_src[comp_cnt] = raw + first_page_offset + src_ofs;
#endif
                            inflate_in[comp_cnt].srcDevice = page_index[page_cnt + k].compressed_page_data;
                            inflate_in[comp_cnt].srcSize = page_index[page_cnt + k].compressed_page_size;
                            inflate_in[comp_cnt].dstDevice = decompressed_pages + decompressed_ofs;
                            inflate_in[comp_cnt].dstSize = page_index[page_cnt + k].uncompressed_page_size;
                            inflate_out[comp_cnt].bytes_written = 0;
                            inflate_out[comp_cnt].status = -1000;
                            inflate_out[comp_cnt].reserved = 0;
                            page_index[page_cnt + k].compressed_page_data = decompressed_pages + decompressed_ofs;
                            decompressed_ofs += page_index[page_cnt + k].uncompressed_page_size;
                        }
                    }
                    page_cnt += chunk_desc[chunk].max_num_pages;
                }
                cudaMemcpyAsync(inflate_in_dev + start_pos, inflate_in + start_pos, sizeof(gpu_inflate_input_s) * (comp_cnt - start_pos), cudaMemcpyHostToDevice);
                cudaMemcpyAsync(inflate_out_dev + start_pos, inflate_out + start_pos, sizeof(gpu_inflate_status_s) * (comp_cnt - start_pos), cudaMemcpyHostToDevice);
                switch(codec)
                {
                case parquet::GZIP:
                    gpuinflate(inflate_in_dev + start_pos, inflate_out_dev + start_pos, comp_cnt - start_pos, 1);
                    break;
                case parquet::SNAPPY:
                    gpu_unsnap(inflate_in_dev + start_pos, inflate_out_dev + start_pos, comp_cnt - start_pos);
                    break;
                default:
                    printf("This is a bug\n");
                    break;
                }
                cudaMemcpyAsync(inflate_out + start_pos, inflate_out_dev + start_pos, sizeof(gpu_inflate_status_s) * (comp_cnt - start_pos), cudaMemcpyDeviceToHost);
            }
        }
        cudaStreamSynchronize(0);
        PERF_SAMPLE_END();
        printf("%zd bytes in %.1fms (%.2fMB/s)\n", total_decompressed_size, uncomp_time * 1000.0, 1.e-6 * total_decompressed_size / uncomp_time);
        for (int i = 0; i < comp_cnt; i++)
        {
            if (inflate_out[i].status != 0 || inflate_out[i].bytes_written > 100000)
                printf("status[%d] = %d (%zd bytes)\n", i, inflate_out[i].status, (size_t)inflate_out[i].bytes_written);
        }
    #if PADUMP_ZLIB_BENCH
        if (compressed_page_cnt[0]) // NOTE: Assumes 1st entry is GZIP and a single codec is used
        {
            uint8_t *zlib_output = new uint8_t[total_decompressed_size];
            uint8_t *dst = zlib_output;
            uint8_t *decompressed_pages_host = nullptr;
            RMM_ALLOC_HOST((void **)&decompressed_pages_host, total_decompressed_size);
            PERF_SAMPLE_START(&zlib_time);
            for (int i = 0; i < gzip_page_cnt; i++)
            {
                size_t comp_len = inflate_in[i].srcSize;
                size_t uncomp_len = inflate_in[i].dstSize;
                int zerr = zlib_uncompress(dst, uncomp_len, zlib_src[i], comp_len);
                if (zerr)
                    printf("ZLIB: %d (data=%02x.%02x.%02x.%02x, src:%d, dst:%d)\n", zerr, zlib_src[i][0], zlib_src[i][1], zlib_src[i][2], zlib_src[i][3], (int)inflate_in[i].srcSize, (int)uncomp_len);
                dst += uncomp_len;
            }
            PERF_SAMPLE_END();
            cudaMemcpy(decompressed_pages_host, decompressed_pages, total_decompressed_size, cudaMemcpyDeviceToHost);
            for (size_t i=0; i<total_decompressed_size; i++)
            {
                if (zlib_output[i] != decompressed_pages_host[i])
                {
                    printf("mismatch at byte %zd: 0x%x/0x%x\n", i, decompressed_pages_host[i], zlib_output[i]);
                    break;
                }
            }
            delete[] zlib_output;
            RMM_FREE_HOST(decompressed_pages_host);
            printf("ZLIB: %zd bytes in %.1fms (%.2fMB/s)\n", total_decompressed_size, zlib_time * 1000.0, 1.e-6 * total_decompressed_size / zlib_time);
        }
        delete[] zlib_src;
    #endif                  
        RMM_FREE_HOST(inflate_in);
        RMM_FREE_HOST(inflate_out);
        RMM_FREE(inflate_out_dev, 0);
        RMM_FREE(inflate_in_dev, 0);
        // Update pages in device memory with the updated value of compressed_page_data, now pointing to the uncompressed data buffer
        cudaMemcpyAsync(page_index_dev, page_index, sizeof(parquet::gpu::PageInfo) * total_pages, cudaMemcpyHostToDevice);
        cudaStreamSynchronize(0);
    }

    // Allocate column data
    for (int i = 0; i < num_columns; i++)
    {
        size_t dtype_len = 0;
        switch (columns[i]->dtype)
        {
        case GDF_INT32:
        case GDF_FLOAT32:
            dtype_len = 4;
            break;
        case GDF_INT64:
        case GDF_FLOAT64:
            dtype_len = 8;
            break;
        case GDF_STRING:
            dtype_len = sizeof(parquet::gpu::nvstrdesc_s); // For now, just the index
            break;
        }
        // TBD: We don't really need to output a valid bit map if the element repetition type is 'required' (no null values)
        if (dtype_len != 0) // should always be true
        {
            RMM_ALLOC((void **)&columns[i]->valid, ((columns[i]->size + 0x1f) & ~0x1f) >> 3, 0);
            RMM_ALLOC(&columns[i]->data, dtype_len * columns[i]->size, 0);
            if (!(columns[i]->valid && columns[i]->data))
            {
                printf("Out of device memory\n");
                goto error_exit;
            }
        }
    }

    // Count the number of string dictionary entries
    total_str_indices = 0;
    for (size_t chunk = 0, page_cnt = 0; chunk < (size_t)num_chunks; chunk++)
    {
        const parquet::ColumnChunk *col = chunk_map[chunk];
        if (file_md.schema[col->schema_idx].type == parquet::BYTE_ARRAY && chunk_desc[chunk].num_dict_pages > 0)
        {
            total_str_indices += page_index[page_cnt].num_values; // NOTE: Assumes first page is always the dictionary page
        }
        page_cnt += chunk_desc[chunk].max_num_pages;
    }
    // Build index for string dictionaries since they can't be indexed directly due to variable-sized elements
    if (total_str_indices > 0)
    {
        RMM_ALLOC((void **)&str_dict_index, total_str_indices * sizeof(parquet::gpu::nvstrdesc_s), 0);
        if (!str_dict_index)
            goto error_exit;
    }
    // Update chunks with pointers to column data
    for (size_t chunk = 0, page_cnt = 0, str_ofs = 0; chunk < (size_t)num_chunks; chunk++)
    {
        const parquet::ColumnChunk *col = chunk_map[chunk];
        const gdf_column *col_gdf = columns[chunk_col[chunk]];
        if (file_md.schema[col->schema_idx].type == parquet::BYTE_ARRAY && chunk_desc[chunk].num_dict_pages > 0)
        {
            chunk_desc[chunk].str_dict_index = str_dict_index + str_ofs;
            str_ofs += page_index[page_cnt].num_values;
        }
        chunk_desc[chunk].valid_map_base = reinterpret_cast<uint32_t *>(col_gdf->valid);
        chunk_desc[chunk].column_data_base = col_gdf->data;
        page_cnt += chunk_desc[chunk].max_num_pages;
    }
    cudaMemcpyAsync(chunk_desc_dev, chunk_desc, sizeof(parquet::gpu::ColumnChunkDesc) * num_chunks, cudaMemcpyHostToDevice);
    if (total_str_indices > 0)
    {
        BuildStringDictionaryIndex(chunk_desc_dev, num_chunks);
        cudaStreamSynchronize(0);
    }

    // Decode page data
    if (total_pages > 0)
    {
        DecodePageData(page_index_dev, (int32_t)total_pages, chunk_desc_dev, num_chunks, file_md.num_rows);
        cudaMemcpyAsync(page_index, page_index_dev, sizeof(parquet::gpu::PageInfo) * total_pages, cudaMemcpyDeviceToHost);
        cudaStreamSynchronize(0);
        for (int i = 0; i < (int)total_pages; i++)
        {
            if (page_index[i].num_rows != 0)
            {
                printf("page[%d].valid_count = %d/%d\n", i, page_index[i].valid_count, page_index[i].num_rows);
                int chunk_idx = page_index[i].chunk_idx;
                if (chunk_idx >= 0 && chunk_idx < num_chunks)
                {
                    columns[chunk_col[chunk_idx]]->null_count += page_index[i].num_rows - page_index[i].valid_count;
                }
            }
        }
        /*if (columns[0]->valid)
        {
            uint32_t *valid_map = nullptr;
            size_t len32 = ((file_md.num_rows + 0x1f) & ~0x1f) >> 5;

            RMM_ALLOC_HOST((void **)&valid_map, len32 * sizeof(uint32_t));
            if (!valid_map)
            {
                printf("out of host mem (%zd bytes)!\n", len32 * sizeof(uint32_t));
                goto error_exit;
            }
            cudaMemcpyAsync(valid_map, columns[0]->valid, len32*sizeof(uint32_t), cudaMemcpyDeviceToHost);
            cudaStreamSynchronize(0);
            printf("col0_valid_map(%p):\n", valid_map);
            for (size_t i = 0; i < len32; i++)
            {
                printf("0x%08x,", valid_map[i]);
            }
            printf("\n");

            RMM_FREE_HOST(valid_map);
        }*/
        /*if (columns[2]->dtype == GDF_INT32)
        {
            int32_t *data = nullptr;
            size_t len = file_md.num_rows;

            RMM_ALLOC_HOST((void **)&data, len * sizeof(data[0]));
            cudaMemcpyAsync(data, columns[2]->data, len * sizeof(data[0]), cudaMemcpyDeviceToHost);
            cudaStreamSynchronize(0);
            printf("col0_data(int32):\n");
            for (size_t i = 0; i < len; i++)
            {
                printf(" %d\n", data[i]);
            }
            printf("\n");
            RMM_FREE_HOST(data);
        }*/
        /*if (columns[1]->dtype == GDF_STRING)
        {
            parquet::gpu::nvstrdesc_s *data = nullptr;
            size_t len = file_md.num_rows;
            std::string str;
            int col = 1;

            RMM_ALLOC_HOST((void **)&data, len * sizeof(data[0]));
            cudaMemcpyAsync(data, columns[col]->data, len * sizeof(data[0]), cudaMemcpyDeviceToHost);
            if (cudaSuccess != cudaStreamSynchronize(0))
            {
                printf("cuda error\n");
            }
            printf("str_data(%d):\n", col);
            for (size_t i = 0; i < len; i++) if (i <= 25 || i+25 >= len)
            {
                const char *ptr = data[i].ptr;
                const char *host_ptr = nullptr;
                int ck = -1;

                for (int chunk = 0; chunk < num_chunks; chunk++)
                {
                    if (chunk_col[chunk] == col)
                    {
                        const char *dev_base = reinterpret_cast<const char *>(chunk_desc[chunk].compressed_data);
                        if (ptr >= dev_base && ptr + data[i].count <= dev_base + chunk_desc[chunk].compressed_size)
                        {
                            size_t first_page_offset = (size_t)chunk_map[chunk]->meta_data.data_page_offset;
                            if (chunk_map[chunk]->meta_data.dictionary_page_offset != 0)
                            {
                                first_page_offset = std::min(first_page_offset, (size_t)chunk_map[chunk]->meta_data.dictionary_page_offset);
                            }
                            ck = chunk;
                            host_ptr = reinterpret_cast<const char *>(raw + first_page_offset + (ptr - dev_base));
                            break;
                        }
                    }
                }
                if (host_ptr)
                    str.assign(host_ptr, data[i].count);
                else
                    str = "<nullptr>";
                printf("[%d] ptr = %p (host=%p), len=%zd (chunk=%d), str=\"%s\"\n", (int)i, data[i].ptr, host_ptr, data[i].count, ck, str.c_str());

            }
            printf("\n");
            RMM_FREE_HOST(data);
        }*/
    }

    // Read pages on CPU (REMOVEME)
    for (size_t i = 0; i < file_md.row_groups.size(); i++)
    {
        const parquet::RowGroup *g = &file_md.row_groups[i];
        for (size_t j = 0; j < g->columns.size(); j++)
        {
            const parquet::ColumnChunk *col = &g->columns[j];
            if (col->meta_data.data_page_offset > 0 && (size_t)col->meta_data.data_page_offset < raw_size)
            {
                int64_t values_remaining = col->meta_data.num_values;
                size_t page_offset = (size_t)col->meta_data.data_page_offset;
                int32_t page_count = 0;
                if (col->meta_data.dictionary_page_offset != 0)
                {
                    page_offset = std::min(page_offset, (size_t)col->meta_data.dictionary_page_offset);
                }
                printf("Page headers for Row group #%zd, Column #%zd:\n", i, j);
                do
                {
                    parquet::PageHeader page_hdr;
                    int32_t num_values = 0;

                    cp.init(raw + page_offset, (size_t)(col->meta_data.data_page_offset + col->meta_data.total_compressed_size - page_offset));
                    if (cp.read(&page_hdr))
                    {
                        //printf(" page_type = %d, uncompressed_size = %d, compressed_size = %d\n", page_hdr.type, page_hdr.uncompressed_page_size, page_hdr.compressed_page_size);
                        switch (page_hdr.type)
                        {
                        case parquet::DATA_PAGE:
                            num_values = page_hdr.data_page_header.num_values;
                            /*if (page_hdr.data_page_header.encoding == parquet::PLAIN_DICTIONARY)
                            {
                                printf(" Data page(%d bytes): num_values = %d, encoding=%d (def:%d, rep:%d)\n", page_hdr.uncompressed_page_size, page_hdr.data_page_header.num_values, page_hdr.data_page_header.encoding, page_hdr.data_page_header.definition_level_encoding, page_hdr.data_page_header.repetition_level_encoding);
                                printf("data bytes = %02x.%02x.%02x.%02x.%02x.%02x...\n", raw[page_offset+cp.bytecount()], raw[page_offset + cp.bytecount() + 1], raw[page_offset + cp.bytecount() + 2], raw[page_offset + cp.bytecount() + 3], raw[page_offset + cp.bytecount() + 4], raw[page_offset + cp.bytecount() + 5]);
                                printf("             %02x.%02x.%02x.%02x.%02x.%02x...\n", raw[page_offset + cp.bytecount() + 6], raw[page_offset + cp.bytecount() + 7], raw[page_offset + cp.bytecount() + 8], raw[page_offset + cp.bytecount() + 9], raw[page_offset + cp.bytecount() + 10], raw[page_offset + cp.bytecount() + 11]);
                            }*/
                            if (file_md.schema[col->schema_idx].max_definition_level > 0)
                            {
                                switch (page_hdr.data_page_header.definition_level_encoding)
                                {
                                case parquet::RLE:
                                case parquet::BIT_PACKED:
                                    break;
                                default:
                                    printf("Invalid encoding type for definition levels\n");
                                    break;
                                }

                            }
                            break;
                        case parquet::DICTIONARY_PAGE:
                            printf(" Dictionary page: num_values = %d, encoding = %d\n", page_hdr.dictionary_page_header.num_values, page_hdr.dictionary_page_header.encoding);
                            break;
                        default:
                            printf(" <unsupported page type %d>\n", (int)page_hdr.type);
                        }
                        page_offset += cp.bytecount() + page_hdr.compressed_page_size;
                    }
                    else
                    {
                        printf(" <failed to read page header>\n");
                        break;
                    }
                    if (num_values < 0)
                    {
                        break;
                    }
                    values_remaining -= num_values;
                    page_count++;

                } while (values_remaining > 0);
                printf(" -> %d pages, %zd/%zd values\n", page_count, (size_t)(col->meta_data.num_values - values_remaining), (size_t)col->meta_data.num_values);
            }
        }
    }

    RMM_FREE(str_dict_index, 0);
    RMM_FREE(decompressed_pages, 0);
    RMM_FREE_HOST(page_index);
    RMM_FREE(page_index_dev, 0);
    if (chunk_desc)
    {
        for (int i = 0; i < num_chunks; i++)
        {
            RMM_FREE(chunk_desc[i].compressed_data, 0);
        }
    }
    RMM_FREE_HOST(chunk_desc);
    RMM_FREE(chunk_desc_dev, 0);
    delete[] raw;

    args->data = columns;
    args->num_cols_out = num_columns;
    args->index_col = index_col;

    return GDF_SUCCESS;

error_exit:
    RMM_FREE(str_dict_index, 0);
    RMM_FREE(decompressed_pages, 0);
    RMM_FREE_HOST(page_index);
    RMM_FREE(page_index_dev, 0);
    if (chunk_desc)
    {
        for (int i = 0; i < num_chunks; i++)
        {
            RMM_FREE(chunk_desc[i].compressed_data, 0);
        }
    }
    RMM_FREE_HOST(chunk_desc);
    RMM_FREE(chunk_desc_dev, 0);
    delete[] raw;

    if (columns)
    {
        for (int i = 0; i < num_columns; i++)
        {
          if (columns[i])
          {
            gdf_column_free(columns[i]);
            if (columns[i]->col_name)
            {
                free(columns[i]->col_name);
            }
            free(columns[i]);
          }
        }
        free(columns);
    }

    return GDF_CUDA_ERROR;
}
