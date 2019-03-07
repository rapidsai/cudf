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
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/count.h>
#include <cuda_runtime.h>

#include "parquet.h"
#include "parquet_gpu.h"

#include <cstring>
#include <iostream>
#include <numeric>
#include <utility>

#define GDF_TRY(call)                                                 \
  {                                                                   \
    gdf_error gdf_status = call;                                      \
    if (gdf_status != GDF_SUCCESS) {                                  \
      std::cerr << "ERROR: "                                          \
                << " in line " << __LINE__ << " of file " << __FILE__ \
                << " failed with "                                    \
                << " (" << gdf_status << ")." << std::endl;           \
      return gdf_status;                                              \
    }                                                                 \
  }

extern gdf_size_type gdf_get_num_chars_bitmask(gdf_size_type size);

static constexpr int NUM_SUPPORTED_CODECS = 2;
static const parquet::Compression g_supportedCodecs[NUM_SUPPORTED_CODECS] = {
    parquet::GZIP, parquet::SNAPPY};
static const char *const g_supportedCodecsNames[NUM_SUPPORTED_CODECS] = {
    "GZIP", "SNAPPY"};

uint8_t *LoadFile(const char *input_fname, size_t *len) {
  size_t file_size;
  FILE *fin = nullptr;
  uint8_t *raw = nullptr;

  *len = 0;
  fin = (input_fname) ? fopen(input_fname, "rb") : nullptr;
  if (!fin) {
    printf("Could not open \"%s\"\n", input_fname);
    return nullptr;
  }
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fseek(fin, 0, SEEK_SET);
  if (file_size <= 0) {
    printf("Invalid file size: %zd\n", file_size);
    fclose(fin);
    return nullptr;
  }
  *len = file_size;
  raw = new uint8_t[file_size];
  if (raw) {
    if (file_size != fread(raw, 1, file_size, fin)) {
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

std::pair<gdf_dtype, gdf_dtype_extra_info> to_dtype(
    parquet::Type physical, parquet::ConvertedType logical) {
  // Logical type used for actual data interpretation; the legacy converted type
  // is superceded by 'logical' type whenever available.
  switch (logical) {
    case parquet::UINT_8:
    case parquet::INT_8:
      return std::make_pair(GDF_INT8, gdf_dtype_extra_info{TIME_UNIT_NONE});
    case parquet::UINT_16:
    case parquet::INT_16:
      return std::make_pair(GDF_INT16, gdf_dtype_extra_info{TIME_UNIT_NONE});
    case parquet::DATE:
      return std::make_pair(GDF_DATE32, gdf_dtype_extra_info{TIME_UNIT_NONE});
    case parquet::TIMESTAMP_MILLIS:
      return std::make_pair(GDF_DATE64, gdf_dtype_extra_info{TIME_UNIT_ms});
    case parquet::TIMESTAMP_MICROS:
      return std::make_pair(GDF_DATE64, gdf_dtype_extra_info{TIME_UNIT_us});
    default:
      break;
  }

  // Physical storage type supported by Parquet; controls the on-disk storage
  // format in combination with the encoding type.
  switch (physical) {
    case parquet::BOOLEAN:
      return std::make_pair(GDF_INT8, gdf_dtype_extra_info{TIME_UNIT_NONE});
    case parquet::INT32:
      return std::make_pair(GDF_INT32, gdf_dtype_extra_info{TIME_UNIT_NONE});
    case parquet::INT64:
      return std::make_pair(GDF_INT64, gdf_dtype_extra_info{TIME_UNIT_NONE});
    case parquet::FLOAT:
      return std::make_pair(GDF_FLOAT32, gdf_dtype_extra_info{TIME_UNIT_NONE});
    case parquet::DOUBLE:
      return std::make_pair(GDF_FLOAT64, gdf_dtype_extra_info{TIME_UNIT_NONE});
    case parquet::BYTE_ARRAY:
    case parquet::FIXED_LEN_BYTE_ARRAY:
      // Can be mapped to GDF_CATEGORY (32-bit hash) or GDF_STRING (nvstring)
      return std::make_pair(GDF_CATEGORY, gdf_dtype_extra_info{TIME_UNIT_NONE});
    case parquet::INT96:
      // deprecated, only used by legacy implementations
    default:
      break;
  }

  return std::make_pair(GDF_invalid, gdf_dtype_extra_info{TIME_UNIT_NONE});
}

// TODO: Move into metadata or schema class
std::string get_index_col(parquet::FileMetaData md) {
  const auto it =
      std::find_if(md.key_value_metadata.begin(), md.key_value_metadata.end(),
                   [](const auto &item) { return item.key == "pandas"; });

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

// TODO: Move to filemetadata class
void print_metadata(const parquet::FileMetaData &file_md) {
  printf("Metadata:\n");
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
  printf(" num rows = %zd\n", (size_t)file_md.num_rows);
  printf(" num row groups = %zd\n", file_md.row_groups.size());
  printf(" num columns = %zd\n", file_md.row_groups[0].columns.size());
}

// TODO: Remove
void print_gdf_column(gdf_column *col, int index) {
  printf("  [%d] name=%s size=%d type=%d data=%lx valid=%lx\n", index,
         col->col_name, col->size, col->dtype, (int64_t)col->data,
         (int64_t)col->valid);
}
// TODO: Remove
void print_rowgroup(const parquet::RowGroup &rowgroup, int row_start) {
  printf("  [%d] size=%ld rows=%ld cols=%zd\n", row_start,
         rowgroup.total_byte_size, rowgroup.num_rows, rowgroup.columns.size());
}

template <typename T = uint8_t>
T required_bits(uint32_t max_level) {
  return static_cast<T>(parquet::CPReader::NumRequiredBits(max_level));
}
template <typename T = uint8_t>
T required_type_length(parquet::ConvertedType converted_type) {
  if (converted_type == parquet::ConvertedType::INT_8 ||
      converted_type == parquet::ConvertedType::UINT_8) {
    return 1;
  } else if (converted_type == parquet::ConvertedType::INT_16 ||
             converted_type == parquet::ConvertedType::UINT_16) {
    return 2;
  }
  return 0;
}

/**---------------------------------------------------------------------------*
 * @brief A helper class for wrapping CUDA host pinned memory and device memory
 * allocations seen as an array.
 *
 * This abstraction provides functionality for initializing, allocating,
 * and managing a linear pool of memory that required both CPU and GPU access.
 *---------------------------------------------------------------------------**/
class cuio_chunk_array {
  using cuio_chunkdesc = parquet::gpu::ColumnChunkDesc;

 public:
  explicit cuio_chunk_array(size_t num_max_chunks)
      : num_max_chunks(num_max_chunks) {
    cudaMallocHost(&h_chunks, sizeof(cuio_chunkdesc) * num_max_chunks);
    RMM_ALLOC(&d_chunks, sizeof(cuio_chunkdesc) * num_max_chunks, 0);
  }

  ~cuio_chunk_array() {
    if (h_chunks) {
      for (int i = 0; i < num_chunks; i++) {
        RMM_FREE(h_chunks[i].compressed_data, 0);
      }
    }
    RMM_FREE(d_chunks, 0);
    cudaFreeHost(h_chunks);
  }

  gdf_error insert(size_t compressed_size, const uint8_t *compressed_data,
                   size_t num_values, uint16_t datatype,
                   uint16_t datatype_length, uint32_t start_row,
                   uint32_t num_rows, int16_t max_definition_level,
                   int16_t max_repetition_level, int8_t codec) {
    auto &chunk = h_chunks[num_chunks];
    chunk.compressed_data = nullptr;
    chunk.compressed_size = compressed_size;
    chunk.num_values = num_values;
    chunk.start_row = start_row;
    chunk.num_rows = num_rows;
    chunk.max_def_level = max_definition_level;
    chunk.max_rep_level = max_repetition_level;
    chunk.def_level_bits = required_bits(max_definition_level);
    chunk.rep_level_bits = required_bits(max_repetition_level);
    chunk.data_type = datatype | (datatype_length << 3);
    chunk.num_data_pages = 0;
    chunk.num_dict_pages = 0;
    chunk.max_num_pages = 0;
    chunk.page_info = nullptr;
    chunk.str_dict_index = nullptr;
    chunk.valid_map_base = nullptr;
    chunk.column_data_base = nullptr;
    chunk.codec = codec;

    if (chunk.compressed_size != 0) {
      RMM_ALLOC(&chunk.compressed_data, chunk.compressed_size, 0);
      cudaMemcpyAsync(chunk.compressed_data, compressed_data,
                      chunk.compressed_size, cudaMemcpyHostToDevice);
    }
    num_chunks++;
    return GDF_SUCCESS;
  }

  cuio_chunkdesc &operator[](size_t i) const { return h_chunks[i]; }
  cuio_chunkdesc *host_ptr() const { return h_chunks; }
  cuio_chunkdesc *device_ptr() const { return d_chunks; }

  size_t capacity() const { return num_max_chunks; }
  size_t size() const { return num_chunks; }

 private:
  size_t num_max_chunks = 0;
  size_t num_chunks = 0;
  cuio_chunkdesc *h_chunks = 0;
  cuio_chunkdesc *d_chunks = 0;
};

/**---------------------------------------------------------------------------*
 * @brief A helper class that wraps a gdf_column to provide RAII functionality.
 *
 * This abstraction provides functionality for initializing, allocating,
 * and managing a gdf_column and its memory. Like other smart pointers,
 * ownership can be transferred by calling `release()` and and using the raw
 * pointer.
 *---------------------------------------------------------------------------**/
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
    const auto num_rows = std::max(col->size, 1);
    const auto num_masks = gdf_get_num_chars_bitmask(num_rows);
    int column_byte_width = 0;

    // For strings, just store the startpos + length for now
    if (col->dtype == GDF_STRING) {
      column_byte_width = sizeof(parquet::gpu::nvstrdesc_s);
    } else {
      get_column_byte_width(col, &column_byte_width);
    }
    RMM_TRY(RMM_ALLOC(&col->data, num_rows * column_byte_width, 0));
    RMM_TRY(RMM_ALLOC(&col->valid, sizeof(gdf_valid_type) * num_masks, 0));
    CUDA_TRY(cudaMemset(col->valid, 0, sizeof(gdf_valid_type) * num_masks));
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

/*struct MetaDataReader : public parquet::FileMetaData {
  virtual int get_num_cols() {
    if (not row_groups.empty()) {
      return (int)row_groups[0].columns.size();
    }
    return 0;
  }
  virtual int get_num_rows() {
    return num_rows;
  }
  virtual std::vector<std::string> get_col_names() {
    std::vector<std::string> col_names;
    for (auto &col : row_groups[0].columns) {
      col_names.push_back(to_dot_string(col.meta_data.path_in_schema));
    }
    return col_names;
  }

  virtual std::string get_name(int col) {
    if (col < row_groups[0].columns.size()) {
      return to_dot_string(row_groups[0].columns[col].meta_data.path_in_schema);
    }
    return 0;
  }
  virtual std::pair<gdf_dtype, gdf_dtype_extra_info> get_dtype(int col) {
    if (col < row_groups[0].columns.size()) {
      return to_dtype(
          schema[row_groups[0].columns[col].schema_idx].type,
          schema[row_groups[0].columns[col].schema_idx].converted_type);
    }
    return std::make_pair(GDF_invalid, gdf_dtype_extra_info{TIME_UNIT_NONE});
  }
};*/

gdf_error init_metadata(const uint8_t *data, size_t len,
                        parquet::FileMetaData *md) {
  constexpr auto header_len = sizeof(parquet::file_header_s);
  constexpr auto ender_len = sizeof(parquet::file_ender_s);

  if (!data || len < header_len + ender_len) {
    std::cerr << "Invalid PARQUET data:" << std::endl;
    std::cerr << "  data: " << data << "len: " << len << std::endl;
    return GDF_FILE_ERROR;
  }

  auto header = (const parquet::file_header_s *)data;
  auto ender = (const parquet::file_ender_s *)(data + len - ender_len);
  if (header->magic != PARQUET_MAGIC || ender->magic != PARQUET_MAGIC) {
    std::cerr << "Invalid PARQUET magic: " << std::endl;
    std::cerr << "  header: " << header->magic << "footer:" << ender->magic << std::endl;
    return GDF_FILE_ERROR;
  }

  if (ender->footer_len <= 0 ||
      ender->footer_len > len - header_len - ender_len) {
    std::cerr << "Invalid PARQUET footer: " << std::endl;
    std::cerr << "  len: " << ender->footer_len << std::endl;
    return GDF_FILE_ERROR;
  }

  parquet::CPReader cp;
  cp.init(data + len - ender->footer_len - ender_len, ender->footer_len);
  if (not cp.read(md)) {
    std::cerr << "Error reading metadata: " << std::endl;
    return GDF_FILE_ERROR;
  }
  if (not cp.InitSchema(md)) {
    std::cerr << "Error populating metadata: " << std::endl;
    return GDF_FILE_ERROR;
  }
  print_metadata(*md);

  return GDF_SUCCESS;
}

/**---------------------------------------------------------------------------*
 * @brief Reads Apache Parquet data and returns an array of gdf_columns.
 *
 * @param[in,out] args Structure containing input and output args
 *
 * @return gdf_error GDF_SUCCESS if successful, otherwise an error code.
 *---------------------------------------------------------------------------**/
gdf_error read_parquet(pq_read_arg *args) {
  parquet::FileMetaData file_md;

  using pq_gdf_pair = std::pair<const parquet::ColumnChunk *, gdf_column *>;
  std::vector<pq_gdf_pair> chunk_map;

  std::vector<gdf_column_wrapper> columns;
  int num_columns = 0;
  int num_rows = 0;
  int index_col = -1;

  size_t raw_size;
  std::unique_ptr<uint8_t[]> raw_owner(LoadFile(args->source, &raw_size));
  const uint8_t *raw = raw_owner.get();

  GDF_TRY(init_metadata(raw, raw_size, &file_md));

  if (file_md.row_groups.empty() || file_md.row_groups[0].columns.empty()) {
    std::cout << "No data found." << std::endl;
    return GDF_DATASET_EMPTY;
  }

  // Obtain the index column if available
  std::string index_col_name = get_index_col(file_md);

  // Select only columns required (if it exists), otherwise select all
  // For PANDAS behavior, always return index column unless there are no rows
  std::vector<std::pair<int, std::string>> col_names;
  if (args->use_cols) {
    std::vector<std::string> use_names(args->use_cols,
                                       args->use_cols + args->use_cols_len);
    if (file_md.num_rows != 0) {
      use_names.push_back(index_col_name);
    }
    for (const auto &name : use_names) {
      for (int i = 0; i < file_md.row_groups[0].columns.size(); ++i) {
        if (name == to_dot_string(file_md.row_groups[0].columns[i].meta_data.path_in_schema)) {
          col_names.emplace_back(i, name);
        }
      }
    }
  } else {
    for (const auto &col : file_md.row_groups[0].columns) {
      const auto name = to_dot_string(col.meta_data.path_in_schema);
      if (file_md.num_rows != 0 || name != index_col_name) {
        col_names.emplace_back(col_names.size(), name);
      }
    }
  }
  if (col_names.empty()) {
    std::cout << "No matching columns found." << std::endl;
    return GDF_SUCCESS;
  }
  num_columns = col_names.size();

  // Initialize gdf_column metadata
  std::cout << "Selected Columns = " << num_columns << std::endl;
  for (const auto &col : col_names) {
    const auto idx = file_md.row_groups[0].columns[col.first].schema_idx;
    const auto dtype_info =
        to_dtype(file_md.schema[idx].type, file_md.schema[idx].converted_type);

    columns.emplace_back(static_cast<gdf_size_type>(file_md.num_rows),
                         dtype_info.first, dtype_info.second, col.second);

    if (col.second == index_col_name) {
      index_col = columns.size() - 1;
    }
  }

  // Count and initialize gpu chunk description structures
  cuio_chunk_array chunks(file_md.row_groups.size() * num_columns);
  chunk_map.resize(chunks.capacity());

  // Initialize column chunk info
  // TODO: Parallelize for large number of columns
  std::cout << "Selected Rowgroups = " << file_md.row_groups.size()
            << std::endl;
  for (const auto &rowgroup : file_md.row_groups) {
    for (const auto &col : rowgroup.columns) {
      const auto name = to_dot_string(col.meta_data.path_in_schema);

      auto it = std::find_if(
          columns.begin(), columns.end(),
          [&](auto &gdf_col) { return gdf_col->col_name == name; });

      if (it != columns.end()) {
        if (chunks.size() < chunks.capacity()) {
          auto idx = col.schema_idx;
          auto offset = (col.meta_data.dictionary_page_offset != 0)
                            ? std::min(col.meta_data.data_page_offset,
                                       col.meta_data.dictionary_page_offset)
                            : col.meta_data.data_page_offset;
          auto byte_width = 0;
          if (file_md.schema[idx].type == parquet::FIXED_LEN_BYTE_ARRAY) {
            byte_width = file_md.schema[idx].type_length << 3;
          } else {
            get_column_byte_width((*it).get(), &byte_width);
          }
          chunks.insert(col.meta_data.total_compressed_size, raw + offset,
                        col.meta_data.num_values, file_md.schema[idx].type,
                        byte_width, num_rows, rowgroup.num_rows,
                        file_md.schema[idx].max_definition_level,
                        file_md.schema[idx].max_repetition_level,
                        col.meta_data.codec);

          chunk_map[chunks.size() - 1] = std::make_pair(&col, (*it).get());
        } else {
          std::cerr << "Too many column chunks detected" << std::endl;
        }
      }
    }
    num_rows += rowgroup.num_rows;
  }

  // Allocate column data
  // TODO: Parallelize for large number of columns
  for (int i = 0; i < num_columns; ++i) {
    GDF_TRY(columns[i].allocate());
  }

  // Decode page headers to count the number of pages required
  CUDA_TRY(cudaMemcpyAsync(chunks.device_ptr(), chunks.host_ptr(),
                           sizeof(parquet::gpu::ColumnChunkDesc) * chunks.size(),
                           cudaMemcpyHostToDevice));
  parquet::gpu::DecodePageHeaders(chunks.device_ptr(), chunks.size());
  CUDA_TRY(cudaMemcpyAsync(chunks.host_ptr(), chunks.device_ptr(),
                           sizeof(parquet::gpu::ColumnChunkDesc) * chunks.size(),
                           cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaStreamSynchronize(0));
  std::cout << "[GPU] " << chunks.size() << " chunks:" << std::endl;
  size_t total_pages = 0;
  for (size_t c = 0; c < chunks.size(); c++) {
    printf(
        "[%zd] %d rows, %d data pages, %d dictionary pages, data_type=0x%x\n",
        c, chunks[c].num_rows, chunks[c].num_data_pages,
        chunks[c].num_dict_pages, chunks[c].data_type);
    total_pages += chunks[c].num_data_pages + chunks[c].num_dict_pages;
  }

  // Store page info
  if (total_pages > 0) {
    // Allocate the number of pages
    using pageInfo = parquet::gpu::PageInfo;
    pageInfo *page_index = nullptr, *page_index_dev = nullptr;
    RMM_TRY(RMM_ALLOC(&page_index_dev, sizeof(pageInfo) * total_pages, 0));
    CUDA_TRY(cudaMallocHost(&page_index, sizeof(pageInfo) * total_pages));

    // Decode page headers again to store the page info
    for (int32_t chunk = 0, page_cnt = 0; chunk < chunks.size(); chunk++) {
      chunks[chunk].max_num_pages = chunks[chunk].num_data_pages + chunks[chunk].num_dict_pages;
      chunks[chunk].page_info = &page_index_dev[page_cnt];
      page_cnt += chunks[chunk].max_num_pages;
    }
    CUDA_TRY(cudaMemcpyAsync(chunks.device_ptr(), chunks.host_ptr(),
                             sizeof(parquet::gpu::ColumnChunkDesc) * chunks.size(),
                             cudaMemcpyHostToDevice));
    parquet::gpu::DecodePageHeaders(chunks.device_ptr(), chunks.size());
    CUDA_TRY(cudaMemcpyAsync(page_index, page_index_dev,
                             sizeof(pageInfo) * total_pages,
                             cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaStreamSynchronize(0));
    printf("[GPU] %d pages:\n", (int)total_pages);
    for (size_t i = 0; i < total_pages; i++) {
      printf(
          "[%zd] ck=%d, row=%d, flags=%d, num_values=%d, encoding=%d, "
          "size=%d\n",
          i, page_index[i].chunk_idx, page_index[i].chunk_row,
          page_index[i].flags, page_index[i].num_values, page_index[i].encoding,
          page_index[i].uncompressed_page_size);
    }

    // Decompress data by first counting the number of compressed pages
    size_t compressed_pages_count[NUM_SUPPORTED_CODECS]{};
    size_t num_compressed_pages = 0;
    size_t total_decompressed_size = 0;
    auto for_each_codec_page = [&](int8_t codec,
                                   const std::function<void(size_t)> &f) {
      size_t page_cnt = 0;
      for (size_t c = 0; c < chunks.size(); c++) {
        const auto page_stride = chunks[c].max_num_pages;
        if (chunks[c].codec == codec) {
          for (int k = 0; k < page_stride; k++) {
            f(page_cnt + k);
          }
        }
        page_cnt += page_stride;
      }
    };

    for (int i = 0; i < NUM_SUPPORTED_CODECS; i++) {
      for_each_codec_page(g_supportedCodecs[i], [&](size_t page) {
        total_decompressed_size += page_index[page].uncompressed_page_size;
        compressed_pages_count[i]++;
        num_compressed_pages++;
      });
    }

    std::cout << "[GPU]: Total compressed size: " << total_decompressed_size << std::endl;
    std::cout << " gzip pages   = " << compressed_pages_count[0] << std::endl;
    std::cout << " snappy pages = " << compressed_pages_count[1] << std::endl;

    if (total_decompressed_size > 0) {
      uint8_t *decompressed_pages = nullptr;
      RMM_ALLOC(&decompressed_pages, total_decompressed_size, 0);

      using inflate_args = gpu_inflate_input_s;
      using inflate_stat = gpu_inflate_status_s;
      inflate_args *inflate_in = nullptr, *inflate_in_dev = nullptr;
      inflate_stat *inflate_out = nullptr, *inflate_out_dev = nullptr;

      cudaMallocHost(&inflate_in, sizeof(inflate_args) * num_compressed_pages);
      RMM_ALLOC(&inflate_in_dev, sizeof(inflate_args) * num_compressed_pages, 0);
      cudaMallocHost(&inflate_out, sizeof(inflate_stat) * num_compressed_pages);
      RMM_ALLOC(&inflate_out_dev, sizeof(inflate_stat) * num_compressed_pages, 0);

      size_t decompressed_ofs = 0;
      int32_t argc = 0;
      for (int i = 0; i < NUM_SUPPORTED_CODECS; i++) {
        if (compressed_pages_count[i] > 0) {

          int32_t start_pos = argc;

          for_each_codec_page(g_supportedCodecs[i], [&](size_t page) {
            inflate_in[argc].srcDevice = page_index[page].compressed_page_data;
            inflate_in[argc].srcSize = page_index[page].compressed_page_size;
            inflate_in[argc].dstDevice = decompressed_pages + decompressed_ofs;
            inflate_in[argc].dstSize = page_index[page].uncompressed_page_size;

            inflate_out[argc].bytes_written = 0;
            inflate_out[argc].status = -1000;
            inflate_out[argc].reserved = 0;

            page_index[page].compressed_page_data =
                (uint8_t *)inflate_in[argc].dstDevice;
            decompressed_ofs += inflate_in[argc].dstSize;
            argc++;
          });

          cudaMemcpyAsync(inflate_in_dev + start_pos, inflate_in + start_pos,
                          sizeof(inflate_args) * (argc - start_pos),
                          cudaMemcpyHostToDevice);
          cudaMemcpyAsync(inflate_out_dev + start_pos, inflate_out + start_pos,
                          sizeof(inflate_stat) * (argc - start_pos),
                          cudaMemcpyHostToDevice);
          switch (g_supportedCodecs[i]) {
            case parquet::GZIP:
              gpuinflate(inflate_in_dev + start_pos,
                         inflate_out_dev + start_pos, argc - start_pos, 1);
              break;
            case parquet::SNAPPY:
              gpu_unsnap(inflate_in_dev + start_pos,
                         inflate_out_dev + start_pos, argc - start_pos);
              break;
            default:
              printf("This is a bug\n");
              break;
          }
          cudaMemcpyAsync(inflate_out + start_pos, inflate_out_dev + start_pos,
                          sizeof(inflate_stat) * (argc - start_pos),
                          cudaMemcpyDeviceToHost);
        }
      }
      cudaStreamSynchronize(0);

      cudaFreeHost(inflate_out);
      RMM_FREE(inflate_out_dev, 0);
      cudaFreeHost(inflate_in);
      RMM_FREE(inflate_in_dev, 0);

      // Update pages in device memory with the updated value of
      // compressed_page_data, now pointing to the uncompressed data buffer
      cudaMemcpyAsync(page_index_dev, page_index,
                 sizeof(parquet::gpu::PageInfo) * total_pages,
                 cudaMemcpyHostToDevice);
    }

    // Count the number of string dictionary entries
    size_t total_str_indices = 0;
    parquet::gpu::nvstrdesc_s *str_dict_index = 0;
    for (int chunk = 0, page_cnt = 0; chunk < chunks.size(); chunk++) {
      const auto col = chunk_map[chunk].first;
      if (file_md.schema[col->schema_idx].type == parquet::BYTE_ARRAY &&
          chunks[chunk].num_dict_pages > 0) {
        total_str_indices +=
            page_index[page_cnt].num_values;  // NOTE: Assumes first page is
                                              // always the dictionary page
      }
      page_cnt += chunks[chunk].max_num_pages;
    }
    // Build index for string dictionaries since they can't be indexed
    // directly due to variable-sized elements
    if (total_str_indices > 0) {
      RMM_ALLOC(&str_dict_index, total_str_indices * sizeof(parquet::gpu::nvstrdesc_s), 0);
    }

    // Update chunks with pointers to column data
    for (int chunk = 0, page_cnt = 0, str_ofs = 0; chunk < chunks.size();
         chunk++) {
      const auto col = chunk_map[chunk].first;
      const auto gdf = chunk_map[chunk].second;
      if (file_md.schema[col->schema_idx].type == parquet::BYTE_ARRAY &&
          chunks[chunk].num_dict_pages > 0) {
        chunks[chunk].str_dict_index = str_dict_index + str_ofs;
        str_ofs += page_index[page_cnt].num_values;
      }
      chunks[chunk].valid_map_base = (uint32_t *)gdf->valid;
      chunks[chunk].column_data_base = gdf->data;
      page_cnt += chunks[chunk].max_num_pages;
    }
    cudaMemcpyAsync(chunks.device_ptr(), &chunks[0],
                    sizeof(parquet::gpu::ColumnChunkDesc) * chunks.size(),
                    cudaMemcpyHostToDevice);

    // Build any string dictionary indexes for page data
    if (total_str_indices > 0) {
      BuildStringDictionaryIndex(chunks.device_ptr(), chunks.size());
    }

    // Decode page data
    DecodePageData(page_index_dev, (int32_t)total_pages, chunks.device_ptr(),
                   chunks.size(), file_md.num_rows);

    // Synchronize before returning data to user
    cudaStreamSynchronize(0);

    cudaMemcpy(page_index, page_index_dev,
               sizeof(parquet::gpu::PageInfo) * total_pages,
               cudaMemcpyDeviceToHost);
    for (int i = 0; i < (int)total_pages; i++) {
      if (page_index[i].num_rows != 0) {
        printf("page[%d].valid_count = %d/%d\n", i, page_index[i].valid_count,
               page_index[i].num_rows);
        const auto chunk = page_index[i].chunk_idx;
        if (chunk >= 0 && chunk < chunks.size()) {
          chunk_map[chunk].second->null_count +=
              page_index[i].num_rows - page_index[i].valid_count;
        }
      }
    }
  }

  // Transfer ownership to raw pointer output arguments
  args->data = (gdf_column **)malloc(sizeof(gdf_column *) * num_columns);
  for (int i = 0; i < num_columns; ++i) {
    args->data[i] = columns[i].release();
  }
  args->num_cols_out = num_columns;
  args->num_rows_out = num_rows;
  if (index_col != -1) {
    args->index_col = (int *)malloc(sizeof(int));
    *args->index_col = index_col;
  }

  return GDF_SUCCESS;
}
