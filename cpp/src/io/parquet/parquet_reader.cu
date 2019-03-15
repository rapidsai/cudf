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
#include "io/comp/gpuinflate.h"
#include "utilities/cudf_utils.h"
#include "utilities/error_utils.hpp"

#include <cuda_runtime.h>
#include "NVStrings.h"
#include "rmm/rmm.h"
#include "rmm/thrust_rmm_allocator.h"

#include "parquet.h"
#include "parquet_gpu.h"

#include <array>
#include <cstring>
#include <iostream>
#include <numeric>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define CUDF_TRY(call) CUDF_EXPECTS(call == GDF_SUCCESS, "Failed call")

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
    CUDF_EXPECTS(fd > 0, "Failed file open");

    struct stat st {};
    CUDF_EXPECTS(fstat(fd, &st) == 0, "Failed file size query");

    mapped_size = st.st_size;
    CUDF_EXPECTS(mapped_size > 0, "Found zero-sized file");

    mapped_data = mmap(NULL, mapped_size, PROT_READ, MAP_PRIVATE, fd, 0);
    CUDF_EXPECTS(mapped_data != MAP_FAILED, "Failed memory mapping file");
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
 * @brief Function that translates Parquet datatype to GDF dtype
 **/
constexpr std::pair<gdf_dtype, gdf_dtype_extra_info> to_dtype(
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

/**
 * @brief Function that requires the number of bits to store a given value
 **/
template <typename T = uint8_t>
T required_bits(uint32_t max_level) {
  return static_cast<T>(parquet::CPReader::NumRequiredBits(max_level));
}

/**
 * @brief A helper class that wraps a gdf_column and any associated memory.
 *
 * This abstraction provides functionality for initializing and managing a
 * gdf_column (its fields and its memory) while still allowing direct access.
 * Any free memory is automatically deallocated unless ownership is transferred
 * via releasing and assigning the raw pointer to the underlying gdf_column.
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
    // For strings, just store the startpos + length for now
    const auto num_rows = std::max(col->size, 1);
    const auto column_byte_width = (col->dtype == GDF_STRING)
                                       ? sizeof(parquet::gpu::nvstrdesc_s)
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
 * @brief A type-templated helper class that wraps fixed-length device memory,
 * and a complementary host pinned memory of the same size.
 *
 * This abstraction allocates a specified fixed chunk of device memory that can
 * initialized upfront, or gradually initialized as required.
 * The host-side memory can be used to manipulate data on the CPU before and
 * after operating on the same data on the GPU.
 **/
template <typename T>
class hostdevice_vector {
 public:
  using value_type = T;

  explicit hostdevice_vector(size_t initial_size, size_t max_size)
      : num_elements(initial_size), max_elements(max_size) {
    CUDA_TRY(cudaMallocHost(&h_data, sizeof(T) * max_elements));
    RMM_ALLOC(&d_data, sizeof(T) * max_elements, 0);
  }

  ~hostdevice_vector() {
    RMM_FREE(d_data, 0);
    CUDA_TRY(cudaFreeHost(h_data));
  }

  bool insert(const T &data) {
    if (num_elements < max_elements) {
      h_data[num_elements] = data;
      num_elements++;
      return true;
    }
    return false;
  }

  size_t max_size() const { return max_elements; }
  size_t size() const { return num_elements; }
  size_t memory_size() const { return sizeof(T) * num_elements; }

  T &operator[](size_t i) const { return h_data[i]; }
  T *host_ptr(size_t offset = 0) const { return h_data + offset; }
  T *device_ptr(size_t offset = 0) const { return d_data + offset; }

 private:
  size_t max_elements = 0;
  size_t num_elements = 0;
  T *h_data = nullptr;
  T *d_data = nullptr;
};

/**
 * @brief A unique_ptr with a custom deleter that frees the associated device
 * memory back to RMM. Used to help automatically release device memory of
 * manually allocated pointers.
 **/
template <typename T>
struct rmm_deleter {
  void operator()(T *ptr) { RMM_FREE(ptr, 0); }
};
template <typename T>
using device_ptr = std::unique_ptr<T, rmm_deleter<T>>;

/**
 * @brief A helper wrapper class for the Parquet file metadata
 **/
class ParquetMetadata : public parquet::FileMetaData {
  static std::string to_dot_string(
      std::vector<std::string> const &path_in_schema) {
    size_t n = path_in_schema.size();
    std::string s = (n > 0) ? path_in_schema[0] : "";
    for (size_t i = 1; i < n; i++) {
      s += '.';
      s += path_in_schema[i];
    }
    return s;
  }

 public:
  explicit ParquetMetadata(const uint8_t *data, size_t len) {
    constexpr auto header_len = sizeof(parquet::file_header_s);
    constexpr auto ender_len = sizeof(parquet::file_ender_s);
    const auto header = (const parquet::file_header_s *)data;
    const auto ender = (const parquet::file_ender_s *)(data + len - ender_len);
    CUDF_EXPECTS(
        data && len > header_len + ender_len,
        "Incorrect data source");
    CUDF_EXPECTS(
        header->magic == PARQUET_MAGIC && ender->magic == PARQUET_MAGIC,
        "Corrupted header or footer");
    CUDF_EXPECTS(
        ender->footer_len != 0 && ender->footer_len <= len - header_len - ender_len,
        "Incorrect footer length");

    parquet::CPReader cp;
    cp.init(data + len - ender->footer_len - ender_len, ender->footer_len);
    CUDF_EXPECTS(cp.read(this), "Cannot parse metadata");
    CUDF_EXPECTS(cp.InitSchema(this), "Cannot parse metadata schema");

    print_metadata();
  }

  inline int get_total_rows() const { return num_rows; }
  inline int get_num_rowgroups() const { return row_groups.size(); }
  inline int get_num_columns() const { return row_groups[0].columns.size(); }

  std::vector<std::string> get_column_names() {
    std::vector<std::string> col_names;
    for (auto &col : row_groups[0].columns) {
      col_names.push_back(to_dot_string(col.meta_data.path_in_schema));
    }
    return col_names;
  }
  std::string get_column_name(const std::vector<std::string> &path_in_schema) {
    return to_dot_string(path_in_schema);
  }
  std::string get_index_column_name() {
    auto it =
        std::find_if(key_value_metadata.begin(), key_value_metadata.end(),
                     [](const auto &item) { return item.key == "pandas"; });

    if (it != key_value_metadata.end()) {
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

  void print_metadata() {
    LOG_PRINTF("\n[+] Metadata:\n");
    LOG_PRINTF(" version = %d\n", version);
    LOG_PRINTF(" created_by = \"%s\"\n", created_by.c_str());
    LOG_PRINTF(" schema (%zd entries):\n", schema.size());
    for (size_t i = 0; i < schema.size(); i++) {
      LOG_PRINTF(
          "  [%zd] type=%d, name=\"%s\", num_children=%d, rep_type=%d, "
          "max_def_lvl=%d, max_rep_lvl=%d\n",
          i, schema[i].type, schema[i].name.c_str(), schema[i].num_children,
          schema[i].repetition_type, schema[i].max_definition_level,
          schema[i].max_repetition_level);
    }
    LOG_PRINTF(" num rows = %zd\n", (size_t)num_rows);
    LOG_PRINTF(" num row groups = %zd\n", row_groups.size());
    LOG_PRINTF(" num columns = %zd\n", row_groups[0].columns.size());
  }
};

/**
 * @brief Returns the number of total pages from the given column chunks
 * 
 * @param[in] chunks List of column chunk descriptors
 * @param[in,out] total_pages Total number of pages making up the column chunks
 *
 * @return gdf_error GDF_SUCCESS if successful, otherwise an error code.
 **/
gdf_error count_page_headers(
    const hostdevice_vector<parquet::gpu::ColumnChunkDesc> &chunks,
    size_t *total_pages) {

  CUDA_TRY(cudaMemcpyAsync(chunks.device_ptr(), chunks.host_ptr(),
                           chunks.memory_size(), cudaMemcpyHostToDevice));
  CUDA_TRY(parquet::gpu::DecodePageHeaders(chunks.device_ptr(), chunks.size()));
  CUDA_TRY(cudaMemcpyAsync(chunks.host_ptr(), chunks.device_ptr(),
                           chunks.memory_size(), cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaStreamSynchronize(0));

  LOG_PRINTF("[+] Chunk Information\n");
  for (size_t c = 0; c < chunks.size(); c++) {
    LOG_PRINTF(
        " %2zd: comp_data=%ld, comp_size=%zd, num_values=%zd\n     "
        "start_row=%zd num_rows=%d max_def_level=%d max_rep_level=%d\n     "
        "data_type=%d def_level_bits=%d rep_level_bits=%d\n     "
        "num_data_pages=%d num_dict_pages=%d max_num_pages=%d\n",
        c, (uint64_t)chunks[c].compressed_data, chunks[c].compressed_size,
        chunks[c].num_values, chunks[c].start_row, chunks[c].num_rows,
        chunks[c].max_def_level, chunks[c].max_rep_level, chunks[c].data_type,
        chunks[c].def_level_bits, chunks[c].rep_level_bits,
        chunks[c].num_data_pages, chunks[c].num_dict_pages,
        chunks[c].max_num_pages);
    *total_pages += chunks[c].num_data_pages + chunks[c].num_dict_pages;
  }

  return GDF_SUCCESS;
}

/**
 * @brief Returns the page information from the given column chunks
 *
 * @param[in] chunks List of column chunk descriptors
 * @param[in] pages List of page information
 *
 * @return gdf_error GDF_SUCCESS if successful, otherwise an error code.
 **/
gdf_error decode_page_headers(
    const hostdevice_vector<parquet::gpu::ColumnChunkDesc> &chunks,
    const hostdevice_vector<parquet::gpu::PageInfo> &pages) {

  for (size_t c = 0, page_count = 0; c < chunks.size(); c++) {
    chunks[c].max_num_pages = chunks[c].num_data_pages + chunks[c].num_dict_pages;
    chunks[c].page_info = pages.device_ptr(page_count);
    page_count += chunks[c].max_num_pages;
  }

  CUDA_TRY(cudaMemcpyAsync(chunks.device_ptr(), chunks.host_ptr(),
                           chunks.memory_size(), cudaMemcpyHostToDevice));
  CUDA_TRY(parquet::gpu::DecodePageHeaders(chunks.device_ptr(), chunks.size()));
  CUDA_TRY(cudaMemcpyAsync(pages.host_ptr(), pages.device_ptr(),
                           pages.memory_size(), cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaStreamSynchronize(0));

  LOG_PRINTF("[+] Page Header Information\n");
  for (size_t i = 0; i < pages.size(); i++) {
    LOG_PRINTF(
        " %2zd: comp_size=%d, uncomp_size=%d, num_values=%d, chunk_row=%d, "
        "num_rows=%d\n     chunk_idx=%d, flags=%d, encoding=%d, def_level=%d "
        "rep_level=%d, valid_count=%d\n",
        i, pages[i].compressed_page_size, pages[i].uncompressed_page_size,
        pages[i].num_values, pages[i].chunk_row, pages[i].num_rows,
        pages[i].chunk_idx, pages[i].flags, pages[i].encoding,
        pages[i].definition_level_encoding, pages[i].repetition_level_encoding,
        pages[i].valid_count);
  }

  return GDF_SUCCESS;
}

/**
 * @brief Decompresses the page data, at page granularity
 *
 * @param[in] chunks List of column chunk descriptors
 * @param[in] pages List of page information
 * @param[in,out] page_data List of outstanding page data device allocations
 *
 * @return gdf_error GDF_SUCCESS if successful, otherwise an error code.
 **/
gdf_error decompress_page_data(
    const hostdevice_vector<parquet::gpu::ColumnChunkDesc> &chunks,
    const hostdevice_vector<parquet::gpu::PageInfo> &pages,
    std::vector<device_ptr<void>> *page_data) {

  auto for_each_codec_page = [&](parquet::Compression codec,
                                 const std::function<void(size_t)> &f) {
    for (size_t c = 0, page_count = 0; c < chunks.size(); c++) {
      const auto page_stride = chunks[c].max_num_pages;
      if (chunks[c].codec == codec) {
        for (int k = 0; k < page_stride; k++) {
          f(page_count + k);
        }
      }
      page_count += page_stride;
    }
  };

  // Brotli scratch memory for decoding
  rmm::device_vector<uint8_t> debrotli_scratch;

  // Count the exact number of compressed pages
  size_t num_compressed_pages = 0;
  size_t total_decompressed_size = 0;
  std::array<std::pair<parquet::Compression, size_t>, 3> codecs{
      std::make_pair(parquet::GZIP, 0), std::make_pair(parquet::SNAPPY, 0),
      std::make_pair(parquet::BROTLI, 0)};

  for (auto &codec : codecs) {
    for_each_codec_page(codec.first, [&](size_t page) {
      total_decompressed_size += pages[page].uncompressed_page_size;
      codec.second++;
      num_compressed_pages++;
    });
    if (codec.first == parquet::BROTLI && codec.second > 0) {
      debrotli_scratch.resize(get_gpu_debrotli_scratch_size(codec.second > 0));
    }
  }

  LOG_PRINTF(
      "[+] Compression\n Total compressed size: %zd\n Number of "
      "compressed pages: %zd\n  gzip:    %zd \n  snappy: %zd\n",
      total_decompressed_size, num_compressed_pages, codecs[0].second,
      codecs[1].second);

  // Dispatch batches of pages to decompress for each codec
  uint8_t *decompressed_pages = nullptr;
  RMM_TRY(RMM_ALLOC(&decompressed_pages, total_decompressed_size, 0));
  page_data->emplace_back(decompressed_pages);

  hostdevice_vector<gpu_inflate_input_s> inflate_in(0, num_compressed_pages);
  hostdevice_vector<gpu_inflate_status_s> inflate_out(0, num_compressed_pages);

  size_t decompressed_ofs = 0;
  int32_t argc = 0;
  for (const auto &codec : codecs) {
    if (codec.second > 0) {
      int32_t start_pos = argc;

      for_each_codec_page(codec.first, [&](size_t page) {
        inflate_in[argc].srcDevice = pages[page].page_data;
        inflate_in[argc].srcSize = pages[page].compressed_page_size;
        inflate_in[argc].dstDevice = decompressed_pages + decompressed_ofs;
        inflate_in[argc].dstSize = pages[page].uncompressed_page_size;

        inflate_out[argc].bytes_written = 0;
        inflate_out[argc].status = static_cast<uint32_t>(-1000);
        inflate_out[argc].reserved = 0;

        pages[page].page_data = (uint8_t *)inflate_in[argc].dstDevice;
        decompressed_ofs += inflate_in[argc].dstSize;
        argc++;
      });

      CUDA_TRY(cudaMemcpyAsync(
          inflate_in.device_ptr(start_pos), inflate_in.host_ptr(start_pos),
          sizeof(decltype(inflate_in)::value_type) * (argc - start_pos),
          cudaMemcpyHostToDevice));
      CUDA_TRY(cudaMemcpyAsync(
          inflate_out.device_ptr(start_pos),
          inflate_out.host_ptr(start_pos),
          sizeof(decltype(inflate_out)::value_type) * (argc - start_pos),
          cudaMemcpyHostToDevice));
      switch (codec.first) {
        case parquet::GZIP:
          CUDA_TRY(gpuinflate(inflate_in.device_ptr(start_pos),
                              inflate_out.device_ptr(start_pos),
                              argc - start_pos, 1))
          break;
        case parquet::SNAPPY:
          CUDA_TRY(gpu_unsnap(inflate_in.device_ptr(start_pos),
                              inflate_out.device_ptr(start_pos),
                              argc - start_pos));
          break;
        case parquet::BROTLI:
          CUDA_TRY(gpu_debrotli(inflate_in.device_ptr(start_pos),
                                inflate_out.device_ptr(start_pos),
                                debrotli_scratch.data().get(),
                                debrotli_scratch.size(), argc - start_pos));
          break;
        default:
          std::cerr << "This is a bug" << std::endl;
          break;
      }
      CUDA_TRY(cudaMemcpyAsync(
          inflate_out.host_ptr(start_pos),
          inflate_out.device_ptr(start_pos),
          sizeof(decltype(inflate_out)::value_type) * (argc - start_pos),
          cudaMemcpyDeviceToHost));
    }
  }
  CUDA_TRY(cudaStreamSynchronize(0));

  // Update the page information in device memory with the updated value of
  // page_data; it now points to the uncompressed data buffer
  CUDA_TRY(cudaMemcpyAsync(pages.device_ptr(), pages.host_ptr(),
                           pages.memory_size(), cudaMemcpyHostToDevice));

  return GDF_SUCCESS;
}

/**
 * @brief Converts the page data and outputs to gdf_columns
 *
 * @param[in] chunks List of column chunk descriptors
 * @param[in] pages List of page information
 * @param[in] chunk_map Mapping between column chunk and gdf_column
 * @param[in] total_rows Total number of rows to output
 *
 * @return gdf_error GDF_SUCCESS if successful, otherwise an error code.
 **/
gdf_error decode_page_data(
    const hostdevice_vector<parquet::gpu::ColumnChunkDesc> &chunks,
    const hostdevice_vector<parquet::gpu::PageInfo> &pages,
    const std::vector<gdf_column *> &chunk_map, size_t total_rows) {

  auto is_dict_chunk = [](const parquet::gpu::ColumnChunkDesc &chunk) {
    return (chunk.data_type & 0x7) == parquet::BYTE_ARRAY &&
           chunk.num_dict_pages > 0;
  };

  // Count the number of string dictionary entries
  // NOTE: Assumes first page in the chunk is always the dictionary page
  size_t total_str_dict_indexes = 0;
  for (size_t c = 0, page_count = 0; c < chunks.size(); c++) {
    if (is_dict_chunk(chunks[c])) {
      total_str_dict_indexes += pages[page_count].num_values;
    }
    page_count += chunks[c].max_num_pages;
  }

  // Build index for string dictionaries since they can't be indexed
  // directly due to variable-sized elements
  rmm::device_vector<parquet::gpu::nvstrdesc_s> str_dict_index;
  if (total_str_dict_indexes > 0) {
    str_dict_index.resize(total_str_dict_indexes);
  }

  // Update chunks with pointers to column data
  for (size_t c = 0, page_count = 0, str_ofs = 0; c < chunks.size(); c++) {
    if (is_dict_chunk(chunks[c])) {
      chunks[c].str_dict_index = str_dict_index.data().get() + str_ofs;
      str_ofs += pages[page_count].num_values;
    }
    chunks[c].valid_map_base = (uint32_t *)chunk_map[c]->valid;
    chunks[c].column_data_base = chunk_map[c]->data;
    page_count += chunks[c].max_num_pages;
  }
  CUDA_TRY(cudaMemcpyAsync(chunks.device_ptr(), chunks.host_ptr(),
                           chunks.memory_size(), cudaMemcpyHostToDevice));
  if (total_str_dict_indexes > 0) {
    CUDA_TRY(BuildStringDictionaryIndex(chunks.device_ptr(), chunks.size()));
  }
  CUDA_TRY(DecodePageData(pages.device_ptr(), pages.size(), chunks.device_ptr(),
                          chunks.size(), total_rows));
  CUDA_TRY(cudaMemcpyAsync(pages.host_ptr(), pages.device_ptr(),
                           pages.memory_size(), cudaMemcpyDeviceToHost));
  CUDA_TRY(cudaStreamSynchronize(0));

  LOG_PRINTF("[+] Page Data Information\n");
  for (size_t i = 0; i < pages.size(); i++) {
    if (pages[i].num_rows > 0) {
      LOG_PRINTF(" %2zd: valid_count=%d/%d\n", i, pages[i].valid_count,
                 pages[i].num_rows);
      const size_t c = pages[i].chunk_idx;
      if (c < chunks.size()) {
        chunk_map[c]->null_count += pages[i].num_rows - pages[i].valid_count;
      }
    }
  }

  return GDF_SUCCESS;
}

/**
 * @brief Reads Apache Parquet data and returns an array of gdf_columns.
 *
 * @param[in,out] args Structure containing input and output args
 *
 * @return gdf_error GDF_SUCCESS if successful, otherwise an error code.
 **/
gdf_error read_parquet(pq_read_arg *args) {

  std::vector<gdf_column_wrapper> columns;
  int num_columns = 0;
  int num_rows = 0;
  int index_col = -1;

  DataSource input(args->source);
  const auto raw = input.data();
  const auto raw_size = input.size();

  // Init schema and metadata
  ParquetMetadata md(raw, raw_size);
  CUDF_EXPECTS(md.get_num_rowgroups() > 0, "No row groups found");
  CUDF_EXPECTS(md.get_num_columns() > 0, "No columns found");

  // Obtain the index column if available
  std::string index_col_name = md.get_index_column_name();

  // Select only columns required (if it exists), otherwise select all
  // For PANDAS behavior, always return index column unless there are no rows
  std::vector<std::pair<int, std::string>> col_names;
  if (args->use_cols) {
    std::vector<std::string> use_names(args->use_cols,
                                       args->use_cols + args->use_cols_len);
    if (md.get_total_rows() > 0) {
      use_names.push_back(index_col_name);
    }
    for (const auto &use_name : use_names) {
      size_t index = 0;
      for (const auto name : md.get_column_names()) {
        if (name == use_name) {
          col_names.emplace_back(index, name);
        }
        index++;
      }
    }
  } else {
    for (const auto& name : md.get_column_names()) {
      if (md.get_total_rows() > 0 || name != index_col_name) {
        col_names.emplace_back(col_names.size(), name);
      }
    }
  }
  CUDF_EXPECTS(not col_names.empty(), "Filterd out all columns");
  num_columns = col_names.size();

  // Initialize gdf_columns
  LOG_PRINTF("[+] Selected columns: %d\n", num_columns);
  for (const auto &name : col_names) {
    auto &col_schema = md.schema[md.row_groups[0].columns[name.first].schema_idx];
    auto dtype_info = to_dtype(col_schema.type, col_schema.converted_type);

    columns.emplace_back(static_cast<gdf_size_type>(md.get_total_rows()),
                         dtype_info.first, dtype_info.second, name.second);

    LOG_PRINTF(" %2zd: name=%s size=%zd type=%d data=%lx valid=%lx\n",
               columns.size() - 1, columns.back()->col_name,
               (size_t)columns.back()->size, columns.back()->dtype,
               (uint64_t)columns.back()->data, (uint64_t)columns.back()->valid);

    if (name.second == index_col_name) {
      index_col = columns.size() - 1;
    }
  }

  // Allocate column chunk descriptors
  const auto num_column_chunks = md.get_num_rowgroups() * num_columns;
  hostdevice_vector<parquet::gpu::ColumnChunkDesc> chunks(0, num_column_chunks);
  std::vector<gdf_column *> chunk_map(num_column_chunks);
  std::vector<device_ptr<void>> page_data;

  // Initialize column chunk info
  size_t total_decompressed_size = 0;
  LOG_PRINTF("[+] Column Chunk Description\n");
  for (const auto &rowgroup : md.row_groups) {
    for (size_t i = 0; i < col_names.size(); i++) {
      auto name = col_names[i];
      auto &col_meta = rowgroup.columns[name.first].meta_data;
      auto &col_schema = md.schema[rowgroup.columns[name.first].schema_idx];
      auto &gdf_column = columns[i];

      // Spec requires each row group to contain exactly one chunk for every
      // column. If there are too many or too few, continue with best effort
      if (name.second != md.get_column_name(col_meta.path_in_schema)) {
        std::cerr << "Detected mismatched column chunk" << std::endl;
        continue;
      }
      if (chunks.size() >= chunks.max_size()) {
        std::cerr << "Detected too many column chunks" << std::endl;
        continue;
      }

      int32_t type_width = (col_schema.type == parquet::FIXED_LEN_BYTE_ARRAY)
                               ? (col_schema.type_length << 3)
                               : 0;
      if (gdf_column->dtype == GDF_INT8)
        type_width = 1;  // I32 -> I8
      else if (gdf_column->dtype == GDF_INT16)
        type_width = 2;  // I32 -> I16
      else if (gdf_column->dtype == GDF_CATEGORY)
        type_width = 4;  // str -> hash32

      uint8_t *d_data = nullptr;
      if (col_meta.total_compressed_size != 0) {
        const auto offset = (col_meta.dictionary_page_offset != 0)
                                ? std::min(col_meta.data_page_offset,
                                           col_meta.dictionary_page_offset)
                                : col_meta.data_page_offset;
        RMM_TRY(RMM_ALLOC(&d_data, col_meta.total_compressed_size, 0));
        page_data.emplace_back(d_data);
        CUDA_TRY(cudaMemcpyAsync(d_data, raw + offset,
                                 col_meta.total_compressed_size,
                                 cudaMemcpyHostToDevice));
      }
      chunks.insert(parquet::gpu::ColumnChunkDesc(
          col_meta.total_compressed_size, d_data, col_meta.num_values,
          col_schema.type, type_width, num_rows, rowgroup.num_rows,
          col_schema.max_definition_level, col_schema.max_repetition_level,
          required_bits(col_schema.max_definition_level),
          required_bits(col_schema.max_repetition_level), col_meta.codec));

      LOG_PRINTF(
          " %2d: %s start_row=%d, num_rows=%ld, codec=%d, "
          "num_values=%ld\n",
          name.first, name.second.c_str(), num_rows, rowgroup.num_rows,
          col_meta.codec, col_meta.num_values);
      LOG_PRINTF("     total_compressed_size=%ld total_uncompressed_size=%ld\n",
          col_meta.total_compressed_size, col_meta.total_uncompressed_size);
      LOG_PRINTF(
          "     schema_idx=%d, type=%d, type_width=%d, max_def_level=%d, "
          "max_rep_level=%d\n",
          rowgroup.columns[name.first].schema_idx, chunks[chunks.size()-1].data_type, type_width,
          col_schema.max_definition_level, col_schema.max_repetition_level);
      LOG_PRINTF(
          "     data_page_offset=%zd, index_page_offset=%zd, "
          "dict_page_offset=%zd\n",
          (size_t)col_meta.data_page_offset,
          (size_t)col_meta.index_page_offset,
          (size_t)col_meta.dictionary_page_offset);

      // Map each column chunk to its output gdf_column
      chunk_map[chunks.size() - 1] = gdf_column.get();

      if (col_meta.codec != parquet::Compression::UNCOMPRESSED) {
        total_decompressed_size += col_meta.total_uncompressed_size;
      }
    }
    num_rows += rowgroup.num_rows;
  }

  // Determine how many page headers to allocate
  size_t total_pages = 0;
  CUDF_TRY(count_page_headers(chunks, &total_pages));

  if (total_pages > 0) {
    hostdevice_vector<parquet::gpu::PageInfo> pages(total_pages, total_pages);

    CUDF_TRY(decode_page_headers(chunks, pages));
    if (total_decompressed_size > 0) {
      CUDF_TRY(decompress_page_data(chunks, pages, &page_data));
    }
    for (auto &column : columns) {
      CUDF_TRY(column.allocate());
    }
    CUDF_TRY(decode_page_data(chunks, pages, chunk_map, num_rows));
  } else {
    // Columns are still expected to be allocated for an empty dataframe
    for (auto &column : columns) {
      CUDF_TRY(column.allocate());
    }
  }

  // Transfer ownership to raw pointer output arguments
  args->data = (gdf_column **)malloc(sizeof(gdf_column *) * num_columns);
  for (int i = 0; i < num_columns; ++i) {
    args->data[i] = columns[i].release();

    // For string dtype, allocate and return in an NvStrings container instance
    // The container takes a list of string pointers and lengths, and copies
    // into its own memory so the source memory must not be released yet
    if (args->data[i]->dtype == GDF_STRING) {
      using str_pair = std::pair<const char *, size_t>;
      static_assert(sizeof(str_pair) == sizeof(parquet::gpu::nvstrdesc_s));

      auto str_list = static_cast<str_pair *>(args->data[i]->data);
      args->data[i]->data = NVStrings::create_from_index(str_list, num_rows);
    }
  }
  args->num_cols_out = num_columns;
  args->num_rows_out = num_rows;
  if (index_col != -1) {
    args->index_col = (int *)malloc(sizeof(int));
    *args->index_col = index_col;
  }

  return GDF_SUCCESS;
}
