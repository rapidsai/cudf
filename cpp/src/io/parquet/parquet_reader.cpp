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
#include <utility>

#define GDF_TRY(call)                                               \
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
  printf("  [%d] name=%s size=%d type=%d data=%lx valid=%lx\n",
    index, col->col_name, col->size, col->dtype, (int64_t)col->data, (int64_t)col->valid);
}
// TODO: Remove
void print_rowgroup(const parquet::RowGroup &rowgroup, int row_start) {
  printf("  [%d] size=%ld rows=%ld cols=%zd\n",
    row_start, rowgroup.total_byte_size, rowgroup.num_rows, rowgroup.columns.size());
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

// TODO: Move into class
struct col_chunk_desc : public parquet::gpu::ColumnChunkDesc {
  explicit col_chunk_desc() = default;
  explicit col_chunk_desc(const parquet::ColumnChunk *chunk, parquet::Type type,
                          uint16_t type_length, uint32_t start_row,
                          uint32_t num_rows, int16_t max_definition_level,
                          int16_t max_repetition_level, const gdf_column *col) {
    compressed_data = nullptr;
    compressed_size = chunk->meta_data.total_compressed_size;
    num_values = chunk->meta_data.num_values;
    start_row = start_row;
    num_rows = num_rows;
    max_def_level = max_def_level;
    max_rep_level = max_rep_level;
    def_level_bits = required_bits(max_def_level);
    rep_level_bits = required_bits(max_rep_level);
    data_type = static_cast<uint16_t>(type | (type_length << 3));
    num_data_pages = 0;
    num_dict_pages = 0;
    max_num_pages = 0;
    page_info = nullptr;
    str_dict_index = nullptr;
    valid_map_base = nullptr;
    column_data_base = nullptr;
  }
};

void cuio_chunk_init(parquet::gpu::ColumnChunkDesc *chunk,
                     size_t compressed_size, const uint8_t *compressed_data,
                     size_t num_values, uint16_t datatype,
                     uint16_t datatype_length, uint32_t start_row,
                     uint32_t num_rows, int16_t max_definition_level,
                     int16_t max_repetition_level) {
  chunk->compressed_data = nullptr;
  chunk->compressed_size = compressed_size;
  chunk->num_values = num_values;
  chunk->start_row = start_row;
  chunk->num_rows = num_rows;
  chunk->max_def_level = max_definition_level;
  chunk->max_rep_level = max_repetition_level;
  chunk->def_level_bits = required_bits(max_definition_level);
  chunk->rep_level_bits = required_bits(max_repetition_level);
  chunk->data_type = datatype | (datatype_length << 3);
  chunk->num_data_pages = 0;
  chunk->num_dict_pages = 0;
  chunk->max_num_pages = 0;
  chunk->page_info = nullptr;
  chunk->str_dict_index = nullptr;
  chunk->valid_map_base = nullptr;
  chunk->column_data_base = nullptr;

  if (chunk->compressed_size != 0) {
    RMM_ALLOC(&chunk->compressed_data, chunk->compressed_size, 0);
    cudaMemcpyAsync(chunk->compressed_data, compressed_data,
                    chunk->compressed_size, cudaMemcpyHostToDevice);
  }
}

struct parquet_state {
  parquet::gpu::ColumnChunkDesc *chunk_desc = nullptr;
  parquet::gpu::ColumnChunkDesc *chunk_desc_dev = nullptr;
  parquet::gpu::PageInfo *page_index = nullptr;
  parquet::gpu::PageInfo *page_index_dev = nullptr;

  int max_num_chunks = 0;
  int num_chunks = 0;
  size_t total_pages = 0;
  size_t num_compressed_pages = 0;
  size_t total_decompressed_size = 0;
  size_t compressed_page_cnt[NUM_SUPPORTED_CODECS];

  size_t total_str_indices = 0;
  parquet::gpu::nvstrdesc_s *str_dict_index = nullptr;
  uint8_t *decompressed_pages = nullptr;

  gdf_error alloc_chunks(size_t rowgroups, size_t columns) {
    max_num_chunks = static_cast<int32_t>(rowgroups * columns);
    RMM_TRY(RMM_ALLOC(&chunk_desc_dev, sizeof(parquet::gpu::ColumnChunkDesc) * max_num_chunks, 0));
    CUDA_TRY(cudaMallocHost(&chunk_desc, sizeof(parquet::gpu::ColumnChunkDesc) * max_num_chunks));
  }

  gdf_error alloc_pages() {
    RMM_TRY(RMM_ALLOC(&page_index_dev, sizeof(parquet::gpu::PageInfo) * total_pages, 0));
    CUDA_TRY(cudaMallocHost(&page_index, sizeof(parquet::gpu::PageInfo) * total_pages));
  }

  gdf_error alloc_dictionaries() {
    RMM_TRY(RMM_ALLOC(&str_dict_index, total_str_indices * sizeof(parquet::gpu::nvstrdesc_s), 0));
  }

  template <typename Op, typename ... Args>
  gdf_error dispatch_chunk_op(Op chunk_op, Args&&... args) {
    CUDA_TRY(cudaMemcpyAsync(chunk_desc_dev, chunk_desc,
                    sizeof(parquet::gpu::ColumnChunkDesc) * num_chunks,
                    cudaMemcpyHostToDevice));
    chunk_op(std::forward<Args>(args)...);
    CUDA_TRY(cudaMemcpyAsync(chunk_desc, chunk_desc_dev,
                    sizeof(parquet::gpu::ColumnChunkDesc) * num_chunks,
                    cudaMemcpyDeviceToHost));
    CUDA_TRY(cudaStreamSynchronize(0));
  }

  ~parquet_state() {
    RMM_FREE(str_dict_index, 0);
    RMM_FREE(decompressed_pages, 0);
    cudaFreeHost(page_index);
    RMM_FREE(page_index_dev, 0);
    if (chunk_desc) {
      for (int i = 0; i < num_chunks; i++) {
        RMM_FREE(chunk_desc[i].compressed_data, 0);
      }
    }
    cudaFreeHost(chunk_desc);
    RMM_FREE(chunk_desc_dev, 0);
  }
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
  parquet_state state{};

  using pq_gdf_pair = std::pair<const parquet::ColumnChunk *, gdf_column *>;
  std::vector<pq_gdf_pair> chunk_map;
  size_t compressed_page_cnt[NUM_SUPPORTED_CODECS];

  auto &chunk_desc = state.chunk_desc;
  auto &chunk_desc_dev = state.chunk_desc_dev;
  auto &page_index = state.page_index;
  auto &page_index_dev = state.page_index_dev;

  auto &total_pages = state.total_pages;
  auto &num_compressed_pages = state.num_compressed_pages;
  auto &total_decompressed_size = state.total_decompressed_size;
  auto &total_str_indices = state.total_str_indices;
  auto &str_dict_index = state.str_dict_index;
  auto &decompressed_pages = state.decompressed_pages;

  auto &max_num_chunks = state.max_num_chunks;
  auto &num_chunks = state.num_chunks;

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
    std::vector<std::string> use_names(args->use_cols, args->use_cols + args->use_cols_len);
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
        to_dtype(file_md.schema[idx].type,
                 file_md.schema[idx].converted_type);

    columns.emplace_back(static_cast<gdf_size_type>(file_md.num_rows),
                         dtype_info.first, dtype_info.second, col.second);

    if (col.second == index_col_name) {
      index_col = columns.size() - 1;
    }
  }

  // Count and initialize gpu chunk description structures
  GDF_TRY(state.alloc_chunks(file_md.row_groups.size(), num_columns));
  chunk_map.resize(max_num_chunks);

  // Initialize column chunk info
  // TODO: Parallelize for large number of columns
  std::cout << "Selected Rowgroups = " << file_md.row_groups.size() << std::endl;
  for (const auto &rowgroup : file_md.row_groups) {
    for (const auto &col : rowgroup.columns) {
      const auto name = to_dot_string(col.meta_data.path_in_schema);

      auto it = std::find_if(
          columns.begin(), columns.end(),
          [&](auto &gdf_col) { return gdf_col->col_name == name; });
      if (it != columns.end()) {
        if (num_chunks < max_num_chunks) {
          auto idx = col.schema_idx;
          auto offset = (col.meta_data.dictionary_page_offset != 0)
                            ? std::min(col.meta_data.data_page_offset,
                                       col.meta_data.dictionary_page_offset)
                            : col.meta_data.data_page_offset;
          auto byte_width = 0;
          if (file_md.schema[idx].type == parquet::FIXED_LEN_BYTE_ARRAY)
            byte_width = file_md.schema[idx].type_length << 3;
          else
            get_column_byte_width((*it).get(), &byte_width);

          cuio_chunk_init(&chunk_desc[num_chunks],
                          col.meta_data.total_compressed_size, raw + offset,
                          col.meta_data.num_values, file_md.schema[idx].type,
                          byte_width, num_rows, rowgroup.num_rows,
                          file_md.schema[idx].max_definition_level,
                          file_md.schema[idx].max_repetition_level);

          chunk_map[num_chunks] = std::make_pair(&col, (*it).get());
          num_chunks++;
        } else {
          std::cerr << "Too many column chunks detected" << std::endl;
        }
      }
    }
    num_rows += rowgroup.num_rows;
  }

  // Count the number of pages required
  GDF_TRY(state.dispatch_chunk_op(parquet::gpu::DecodePageHeaders,
                                  chunk_desc_dev, num_chunks, cudaStream_t{0}));

  std::cout << "[GPU] " << num_chunks << "chunks:" << std::endl;
  for (size_t c = 0; c < num_chunks; c++) {
    printf(
        "[%zd] %d rows, %d data pages, %d dictionary pages, data_type=0x%x\n",
        c, chunk_desc[c].num_rows, chunk_desc[c].num_data_pages,
        chunk_desc[c].num_dict_pages, chunk_desc[c].data_type);
    total_pages += chunk_desc[c].num_data_pages + chunk_desc[c].num_dict_pages;
  }

  // Store page info
  if (total_pages > 0) {
    GDF_TRY(state.alloc_pages());

    // Decode page headers again, this time storing page info
    for (int32_t chunk = 0, page_cnt = 0; chunk < num_chunks; chunk++) {
      chunk_desc[chunk].max_num_pages =
          chunk_desc[chunk].num_data_pages + chunk_desc[chunk].num_dict_pages;
      chunk_desc[chunk].page_info = &page_index_dev[page_cnt];
      page_cnt += chunk_desc[chunk].max_num_pages;
    }
    GDF_TRY(state.dispatch_chunk_op(parquet::gpu::DecodePageHeaders,
                                    chunk_desc_dev, num_chunks,
                                    cudaStream_t{0}));

    printf("[GPU] %d pages:\n", (int)total_pages);
    for (size_t i = 0; i < total_pages; i++) {
      printf(
          "[%zd] ck=%d, row=%d, flags=%d, num_values=%d, encoding=%d, "
          "size=%d\n",
          i, page_index[i].chunk_idx, page_index[i].chunk_row,
          page_index[i].flags, page_index[i].num_values, page_index[i].encoding,
          page_index[i].uncompressed_page_size);
    }

    // Decompress pages that are compressed
    memset(&compressed_page_cnt, 0, sizeof(compressed_page_cnt));
    num_compressed_pages = 0;
    total_decompressed_size = 0;
    for (int i = 0; i < NUM_SUPPORTED_CODECS; i++) {
      parquet::Compression codec = g_supportedCodecs[i];
      size_t codec_page_cnt = 0, page_cnt = 0;
      for (int chunk = 0; chunk < num_chunks; chunk++) {
        int32_t max_num_pages = chunk_desc[chunk].max_num_pages;
        if (chunk_map[chunk].first->meta_data.codec == codec) {
          codec_page_cnt += max_num_pages;
          for (int k = 0; k < max_num_pages; k++) {
            state.total_decompressed_size +=
                page_index[page_cnt + k].uncompressed_page_size;
          }
        }
        page_cnt += max_num_pages;
      }
      if (codec_page_cnt != 0) {
        printf("[GPU] %s compression (%zd pages, %zd bytes)\n",
               g_supportedCodecsNames[i], codec_page_cnt,
               total_decompressed_size);
      }
      compressed_page_cnt[i] += codec_page_cnt;
      num_compressed_pages += codec_page_cnt;
    }

    if (num_compressed_pages > 0) {
      gpu_inflate_input_s *inflate_in = nullptr, *inflate_in_dev = nullptr;
      gpu_inflate_status_s *inflate_out = nullptr, *inflate_out_dev = nullptr;
      size_t decompressed_ofs = 0;
      int32_t comp_cnt = 0;
      double uncomp_time = 0;

      cudaMallocHost((void **)&inflate_in,
                     sizeof(gpu_inflate_input_s) * num_compressed_pages);
      RMM_ALLOC((void **)&inflate_in_dev,
                sizeof(gpu_inflate_input_s) * num_compressed_pages, 0);
      cudaMallocHost((void **)&inflate_out,
                     sizeof(gpu_inflate_status_s) * num_compressed_pages);
      RMM_ALLOC((void **)&inflate_out_dev,
                sizeof(gpu_inflate_status_s) * num_compressed_pages, 0);
      RMM_ALLOC((void **)&decompressed_pages, total_decompressed_size, 0);

      for (int codec_idx = 0; codec_idx < NUM_SUPPORTED_CODECS; codec_idx++) {
        parquet::Compression codec = g_supportedCodecs[codec_idx];
        if (compressed_page_cnt[codec_idx] > 0) {
          int32_t start_pos = comp_cnt;

          // Fill in decompression in/out structures & update page ptr to point
          // to the decompressed data
          for (int chunk = 0, page_cnt = 0; chunk < num_chunks; chunk++) {
            if (chunk_map[chunk].first->meta_data.codec == codec) {
              for (int k = 0; k < chunk_desc[chunk].max_num_pages;
                   k++, comp_cnt++) {
                inflate_in[comp_cnt].srcDevice =
                    page_index[page_cnt + k].compressed_page_data;
                inflate_in[comp_cnt].srcSize =
                    page_index[page_cnt + k].compressed_page_size;
                inflate_in[comp_cnt].dstDevice =
                    decompressed_pages + decompressed_ofs;
                inflate_in[comp_cnt].dstSize =
                    page_index[page_cnt + k].uncompressed_page_size;
                inflate_out[comp_cnt].bytes_written = 0;
                inflate_out[comp_cnt].status = -1000;
                inflate_out[comp_cnt].reserved = 0;
                page_index[page_cnt + k].compressed_page_data =
                    decompressed_pages + decompressed_ofs;
                decompressed_ofs +=
                    page_index[page_cnt + k].uncompressed_page_size;
              }
            }
            page_cnt += chunk_desc[chunk].max_num_pages;
          }
          cudaMemcpyAsync(inflate_in_dev + start_pos, inflate_in + start_pos,
                          sizeof(gpu_inflate_input_s) * (comp_cnt - start_pos),
                          cudaMemcpyHostToDevice);
          cudaMemcpyAsync(inflate_out_dev + start_pos, inflate_out + start_pos,
                          sizeof(gpu_inflate_status_s) * (comp_cnt - start_pos),
                          cudaMemcpyHostToDevice);
          switch (codec) {
            case parquet::GZIP:
              gpuinflate(inflate_in_dev + start_pos,
                         inflate_out_dev + start_pos, comp_cnt - start_pos, 1);
              break;
            case parquet::SNAPPY:
              gpu_unsnap(inflate_in_dev + start_pos,
                         inflate_out_dev + start_pos, comp_cnt - start_pos);
              break;
            default:
              printf("This is a bug\n");
              break;
          }
          cudaMemcpyAsync(inflate_out + start_pos, inflate_out_dev + start_pos,
                          sizeof(gpu_inflate_status_s) * (comp_cnt - start_pos),
                          cudaMemcpyDeviceToHost);
        }
      }
      cudaStreamSynchronize(0);

      printf("%zd bytes in %.1fms (%.2fMB/s)\n", total_decompressed_size,
             uncomp_time * 1000.0,
             1.e-6 * total_decompressed_size / uncomp_time);
      for (int i = 0; i < comp_cnt; i++) {
        if (inflate_out[i].status != 0 || inflate_out[i].bytes_written > 100000)
          printf("status[%d] = %d (%zd bytes)\n", i, inflate_out[i].status,
                 (size_t)inflate_out[i].bytes_written);
      }

      cudaFreeHost(inflate_in);
      cudaFreeHost(inflate_out);
      RMM_FREE(inflate_out_dev, 0);
      RMM_FREE(inflate_in_dev, 0);
      // Update pages in device memory with the updated value of
      // compressed_page_data, now pointing to the uncompressed data buffer
      cudaMemcpyAsync(page_index_dev, page_index,
                      sizeof(parquet::gpu::PageInfo) * total_pages,
                      cudaMemcpyHostToDevice);
      cudaStreamSynchronize(0);
    }
  }

  // Allocate column data
  // TODO: Parallelize for large number of columns
  for (int i = 0; i < num_columns; ++i) {
    GDF_TRY(columns[i].allocate());
  }

  // Count the number of string dictionary entries
  total_str_indices = 0;
  for (int chunk = 0, page_cnt = 0; chunk < num_chunks; chunk++) {
    const auto col = chunk_map[chunk].first;
    if (file_md.schema[col->schema_idx].type == parquet::BYTE_ARRAY &&
        chunk_desc[chunk].num_dict_pages > 0) {
      total_str_indices +=
          page_index[page_cnt].num_values;  // NOTE: Assumes first page is
                                            // always the dictionary page
    }
    page_cnt += chunk_desc[chunk].max_num_pages;
  }
  // Build index for string dictionaries since they can't be indexed directly
  // due to variable-sized elements
  if (total_str_indices > 0) {
    state.alloc_dictionaries();
  }
  // Update chunks with pointers to column data
  for (int chunk = 0, page_cnt = 0, str_ofs = 0; chunk < num_chunks; chunk++) {
    const auto col = chunk_map[chunk].first;
    const auto gdf = chunk_map[chunk].second;
    if (file_md.schema[col->schema_idx].type == parquet::BYTE_ARRAY &&
        chunk_desc[chunk].num_dict_pages > 0) {
      chunk_desc[chunk].str_dict_index = str_dict_index + str_ofs;
      str_ofs += page_index[page_cnt].num_values;
    }
    chunk_desc[chunk].valid_map_base = (uint32_t *)gdf->valid;
    chunk_desc[chunk].column_data_base = gdf->data;
    page_cnt += chunk_desc[chunk].max_num_pages;
  }
  cudaMemcpyAsync(chunk_desc_dev, chunk_desc,
                  sizeof(parquet::gpu::ColumnChunkDesc) * num_chunks,
                  cudaMemcpyHostToDevice);
  if (total_str_indices > 0) {
    BuildStringDictionaryIndex(chunk_desc_dev, num_chunks);
    cudaStreamSynchronize(0);
  }

  // Decode page data
  if (total_pages > 0) {
    DecodePageData(page_index_dev, (int32_t)total_pages, chunk_desc_dev,
                   num_chunks, file_md.num_rows);
    cudaMemcpyAsync(page_index, page_index_dev,
                    sizeof(parquet::gpu::PageInfo) * total_pages,
                    cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(0);

    for (int i = 0; i < (int)total_pages; i++) {
      if (page_index[i].num_rows != 0) {
        printf("page[%d].valid_count = %d/%d\n", i, page_index[i].valid_count,
               page_index[i].num_rows);
        const auto chunk = page_index[i].chunk_idx;
        if (chunk >= 0 && chunk < num_chunks) {
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
