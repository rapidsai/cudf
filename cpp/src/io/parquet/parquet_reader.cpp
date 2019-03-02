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

#include <cstring>
#include <iostream>
#include <numeric>

#define RMM_ALLOC_HOST(ptr, sz)     cudaMallocHost(ptr, sz)
#define RMM_FREE_HOST(ptr)          cudaFreeHost(ptr)

extern gdf_size_type gdf_get_num_chars_bitmask(gdf_size_type size);

static constexpr int NUM_SUPPORTED_CODECS = 2;
static const parquet::Compression g_supportedCodecs[NUM_SUPPORTED_CODECS] = { parquet::GZIP, parquet::SNAPPY };
static const char * const g_supportedCodecsNames[NUM_SUPPORTED_CODECS] = { "GZIP", "SNAPPY" };

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

gdf_dtype to_dtype(parquet::Type physical, parquet::ConvertedType logical) {
  // Logical type used for actual data interpretation; the legacy converted type
  // is superceded by 'logical' type whenever available.
  switch (logical) {
    case parquet::UINT_8:
    case parquet::INT_8:
      return GDF_INT8;
    case parquet::UINT_16:
    case parquet::INT_16:
      return GDF_INT16;
    case parquet::DATE:
      return GDF_DATE32;
    case parquet::TIMESTAMP_MILLIS:
      return GDF_DATE64;
    default:
      break;
  }

  // Physical storage type supported by Parquet; controls the on-disk storage
  // format in combination with the encoding type.
  switch (physical) {
    case parquet::BOOLEAN:
      return GDF_INT8;
    case parquet::INT32:
      return GDF_INT32;
    case parquet::INT64:
      return GDF_INT64;
    case parquet::FLOAT:
      return GDF_FLOAT32;
    case parquet::DOUBLE:
      return GDF_FLOAT64;
    case parquet::BYTE_ARRAY:
    case parquet::FIXED_LEN_BYTE_ARRAY:
      return GDF_STRING;
    case parquet::INT96:  // deprecated, only used by legacy implementations
    default:
      break;
  }

  return GDF_invalid;
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

gdf_error gdf_column_storage_alloc(gdf_column *col) {
  const auto num_rows = std::max(col->size, 1);
  const auto num_masks = gdf_get_num_chars_bitmask(num_rows);

  RMM_TRY(RMM_ALLOC(&col->valid, sizeof(gdf_valid_type) * num_masks, 0));
  CUDA_TRY(cudaMemset(col->valid, 0, sizeof(gdf_valid_type) * num_masks));

  if (col->dtype != gdf_dtype::GDF_STRING) {
    int column_byte_width = 0;
    get_column_byte_width(col, &column_byte_width);
    RMM_TRY(RMM_ALLOC(&col->data, num_rows * column_byte_width, 0));
  }

  return GDF_SUCCESS;
}

// TODO: Move to filemetadata class
void print_metadata(const parquet::FileMetaData &file_md) {
  printf(" version = %d\n", file_md.version);
  printf(" created_by = \"%s\"\n", file_md.created_by.c_str());
  printf(" schema (%zd entries):\n", file_md.schema.size());
  for (size_t i = 0; i < file_md.schema.size(); i++) {
    printf(
        "  [%zd] type=%d, name=\"%s\", num_children=%d, rep_type=%d, "
        "max_def_lvl=%d, max_rep_lvl=%d\n",
        i, file_md.schema[i].type, file_md.schema[i].name.c_str(),
        file_md.schema[i].num_children, file_md.schema[i].repetition_type,
        file_md.schema[i].max_definition_level,
        file_md.schema[i].max_repetition_level);
  }
  printf(" num_rows = %zd\n", (size_t)file_md.num_rows);
  printf(" row groups = %zd\n", file_md.row_groups.size());
  printf(" num_columns = %zd\n", file_md.row_groups[0].columns.size());
}

// TODO: Remove
void print_gdf_column(gdf_column *col, int index) {
  std::cout << "[" << index << "]:" << std::endl;
  std::cout << "name: " << col->col_name << std::endl;
  std::cout << "size: " << col->size << std::endl;
  std::cout << "type: " << col->dtype << std::endl;
  std::cout << std::endl;
}

// TODO: Move into class
/*struct ColumnChunkDesc : public parquet::gpu::ColumnChunkDesc {
  ColumnChunkDesc(const parquet::ColumnChunk* col, size_t num_rows, uint8_t type_length) {
    compressed_data = nullptr;
    compressed_size = col->meta_data.total_compressed_size;
    num_values = col->meta_data.num_values;
    start_row = row;
    num_rows = static_cast<uint32_t>(num_rows);
    max_def_level = (int16_t)file_md.schema[col->schema_idx].max_definition_level;
    max_rep_level = (int16_t)file_md.schema[col->schema_idx].max_repetition_level;
    def_level_bits = (uint8_t)cp.NumRequiredBits(file_md.schema[col->schema_idx].max_definition_level);
    rep_level_bits = (uint8_t)cp.NumRequiredBits(file_md.schema[col->schema_idx].max_repetition_level);

    auto data_type = file_md.schema[col->schema_idx].type;
    auto type_length = (data_type == parquet::FIXED_LEN_BYTE_ARRAY) ? (file_md.schema[col->schema_idx].type_length << 3) : 0;
    if (columns[k]->dtype == GDF_INT8)
        type_length = 1;
    else if (columns[k]->dtype == GDF_INT16)
        type_length = 2;
    chunk->data_type = static_cast<uint16_t>(data_type | (type_length << 3));

    num_data_pages = 0;
    num_dict_pages = 0;
    max_num_pages = 0;
    page_info = nullptr;
    str_dict_index = nullptr;
    valid_map_base = nullptr;
    column_data_base = nullptr;
  }
};*/

//thrust::system::cuda::experimental::pinned_allocator<float>;

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
    int max_num_chunks = 0, num_chunks = 0;

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
        return GDF_FILE_ERROR;
    }
    if ((fender->footer_len > raw_size - sizeof(parquet::file_header_s) - sizeof(parquet::file_ender_s))
     || (fender->footer_len <= 0))
    {
        printf("Invalid parquet footer length (%d bytes)\n", fender->footer_len);
       return GDF_FILE_ERROR;
    }
    printf("Parquet file footer: %d bytes @ 0x%zx\n", fender->footer_len, raw_size - fender->footer_len - sizeof(parquet::file_ender_s));
    cp.init(raw + raw_size - fender->footer_len - sizeof(parquet::file_ender_s), fender->footer_len);

    if (!cp.read(&file_md)) {
      printf("Error parsing file metadata\n");
    }
    if (!cp.InitSchema(&file_md)) {
      printf("Failed to initialize schema\n");
    }
    printf(" parquet header byte count = %zd/%d\n", cp.bytecount(), fender->footer_len);
    print_metadata(file_md);

    const auto max_num_columns = file_md.row_groups.size()
                                     ? (int)file_md.row_groups[0].columns.size()
                                     : 0;
    if (max_num_columns <= 0) {
      std::cout << "No columns found." << std::endl;
      return GDF_DATASET_EMPTY;
    }

    int* index_col = nullptr;
    std::string index_col_name = get_index_col(file_md);
    if (index_col_name != "") {
      index_col = (int *)malloc(sizeof(int));
      *index_col = 0;
    }

    // Begin with a list of all column indexes in the dataset
    std::vector<size_t> col_indexes(max_num_columns);
    std::iota(col_indexes.begin(), col_indexes.end(), 0);

    // If column names are specified, filter out ones not of interest
    if (args->use_cols) {
      auto names = std::vector<std::string>(args->use_cols, args->use_cols + args->use_cols_len);
      names.push_back(index_col_name);

      col_indexes.clear();
      for (int i = 0; i < max_num_columns; ++i) {
        auto& col = file_md.row_groups[0].columns[i];
        auto name = to_dot_string(col.meta_data.path_in_schema);

        if (std::find(names.begin(), names.end(), name) != names.end()) {
          col_indexes.push_back(i);
        }
      }
    }
    if (col_indexes.empty()) {
      std::cout << "No matching columns found." << std::endl;
      return GDF_SUCCESS;
    } else {
      std::cout << "Selected columns: " << col_indexes.size() << std::endl;
    }

    // Initialize gdf_columns array
    const auto num_columns = static_cast<int>(col_indexes.size());
    columns = (gdf_column **)malloc(num_columns * sizeof(gdf_column *));
    memset(columns, 0, num_columns * sizeof(gdf_column *));

    // Initialize each gdf_column element
    for (int i = 0; i < num_columns; ++i) {
        auto& col = file_md.row_groups[0].columns[col_indexes[i]];
        auto &schema = file_md.schema[col.schema_idx];
        auto name = to_dot_string(
            file_md.row_groups[0].columns[i].meta_data.path_in_schema);
        if (name == index_col_name) {
          *index_col = i;
        }

        columns[i] = (gdf_column *)calloc(sizeof(gdf_column), sizeof(gdf_column));
        columns[i]->size = static_cast<gdf_size_type>(file_md.num_rows);
        columns[i]->dtype = to_dtype(schema.type, schema.converted_type);
        columns[i]->col_name = (char *)malloc(name.length() + 1);
        strcpy(columns[i]->col_name, name.c_str());

        print_gdf_column(columns[i], i);
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

    // Initialize column chunk info
    for (size_t i = 0, row = 0; i < file_md.row_groups.size(); i++) {
      const parquet::RowGroup *g = &file_md.row_groups[i];
      printf("  [%zd] total_size=%zd, %zd rows, %zd columns:\n", i,
             g->total_byte_size, g->num_rows,
             g->columns.size());
      for (size_t j = 0; j < g->columns.size(); j++) {
        const parquet::ColumnChunk *col = &g->columns[j];
        const auto name = to_dot_string(col->meta_data.path_in_schema);

        for (int k = 0; k < num_columns; k++) {
          if (name == columns[k]->col_name) {
            if (num_chunks < max_num_chunks) {
              auto &schema = file_md.schema[col->schema_idx];
              auto *chunk = &chunk_desc[num_chunks];
              auto first_page_offset =
                  (col->meta_data.dictionary_page_offset != 0)
                      ? std::min(col->meta_data.data_page_offset,
                                 col->meta_data.dictionary_page_offset)
                      : col->meta_data.data_page_offset;

              chunk->compressed_data = nullptr;
              chunk->compressed_size = col->meta_data.total_compressed_size;
              chunk->num_values = col->meta_data.num_values;
              chunk->start_row = row;
              chunk->num_rows = (uint32_t)g->num_rows;
              chunk->max_def_level = (int16_t)schema.max_definition_level;
              chunk->max_rep_level = (int16_t)schema.max_repetition_level;
              chunk->def_level_bits =
                  (uint8_t)cp.NumRequiredBits(schema.max_definition_level);
              chunk->rep_level_bits =
                  (uint8_t)cp.NumRequiredBits(schema.max_repetition_level);

              // TODO: Convert to typedispatcher
              auto type_length = (schema.type == parquet::FIXED_LEN_BYTE_ARRAY)
                                     ? (schema.type_length << 3)
                                     : 0;
              if (columns[k]->dtype == GDF_INT8)
                type_length = 1;
              else if (columns[k]->dtype == GDF_INT16)
                type_length = 2;
              chunk->data_type = (uint16_t)(schema.type | (type_length << 3));
              chunk->num_data_pages = 0;
              chunk->num_dict_pages = 0;
              chunk->max_num_pages = 0;
              chunk->page_info = nullptr;
              chunk->str_dict_index = nullptr;
              chunk->valid_map_base = nullptr;
              chunk->column_data_base = nullptr;
              if (col->meta_data.total_compressed_size > 0) {
                RMM_ALLOC(&chunk->compressed_data, col->meta_data.total_compressed_size, 0);
                if (!chunk->compressed_data) goto error_exit;
                cudaMemcpyAsync(chunk->compressed_data, raw + first_page_offset,
                                col->meta_data.total_compressed_size,
                                cudaMemcpyHostToDevice);
              }
              chunk_map[num_chunks] = col;
              chunk_col[num_chunks] = k;
              num_chunks++;
            } else {
              printf("Too many chunks!!!\n");
            }
            break;
          }
        }
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
    if (total_pages > 0)
    {
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

          printf("%zd bytes in %.1fms (%.2fMB/s)\n", total_decompressed_size, uncomp_time * 1000.0, 1.e-6 * total_decompressed_size / uncomp_time);
          for (int i = 0; i < comp_cnt; i++)
          {
              if (inflate_out[i].status != 0 || inflate_out[i].bytes_written > 100000)
                  printf("status[%d] = %d (%zd bytes)\n", i, inflate_out[i].status, (size_t)inflate_out[i].bytes_written);
          }

          RMM_FREE_HOST(inflate_in);
          RMM_FREE_HOST(inflate_out);
          RMM_FREE(inflate_out_dev, 0);
          RMM_FREE(inflate_in_dev, 0);
          // Update pages in device memory with the updated value of compressed_page_data, now pointing to the uncompressed data buffer
          cudaMemcpyAsync(page_index_dev, page_index, sizeof(parquet::gpu::PageInfo) * total_pages, cudaMemcpyHostToDevice);
          cudaStreamSynchronize(0);
      }
    }

    // Allocate column data
    for (int i = 0; i < num_columns; i++)
    {
      if (columns[i]->dtype == GDF_STRING) {
        // For now, just store the startpos + length
        const auto num_rows = std::max(columns[i]->size, 1);
        const auto num_masks = gdf_get_num_chars_bitmask(num_rows);
        const auto dtype_len = sizeof(parquet::gpu::nvstrdesc_s);

        RMM_ALLOC(&columns[i]->valid, sizeof(gdf_valid_type) * num_masks, 0);
        RMM_ALLOC(&columns[i]->data, dtype_len * num_rows, 0);
        if (!(columns[i]->valid && columns[i]->data)) {
          printf("Out of device memory\n");
          goto error_exit;
        }
      } else {
        if (gdf_column_storage_alloc(columns[i]) != GDF_SUCCESS) {
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
