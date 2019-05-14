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

#include "parquet.h"
#include "parquet_gpu.h"

#include "cudf.h"
#include "io/comp/gpuinflate.h"
#include "io/utilities/datasource.hpp"
#include "io/utilities/wrapper_utils.hpp"
#include "utilities/cudf_utils.h"
#include "utilities/error_utils.hpp"

#include <cuda_runtime.h>
#include <nvstrings/NVStrings.h>
#include <rmm/rmm.h>
#include <rmm/thrust_rmm_allocator.h>

#include <array>
#include <iostream>
#include <numeric>
#include <vector>

#if 0
#define LOG_PRINTF(...) std::printf(__VA_ARGS__)
#else
#define LOG_PRINTF(...) (void)0
#endif

/**
 * @brief Function that translates Parquet datatype to GDF dtype
 **/
constexpr std::pair<gdf_dtype, gdf_dtype_extra_info> to_dtype(
    parquet::Type physical, parquet::ConvertedType logical,
    bool strings_to_categorical) {
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
      return std::make_pair(strings_to_categorical ? GDF_CATEGORY : GDF_STRING,
                            gdf_dtype_extra_info{TIME_UNIT_NONE});
    case parquet::INT96:
      // Convert Spark INT96 timestamp to GDF_DATE64
      return std::make_pair(GDF_DATE64, gdf_dtype_extra_info{TIME_UNIT_ms});
    default:
      break;
  }

  return std::make_pair(GDF_invalid, gdf_dtype_extra_info{TIME_UNIT_NONE});
}

/**
 * @brief Helper that returns the required the number of bits to store a value
 **/
template <typename T = uint8_t>
T required_bits(uint32_t max_level) {
  return static_cast<T>(parquet::CompactProtocolReader::NumRequiredBits(max_level));
}

/**
 * @brief A helper wrapper for Parquet file metadata. Provides some additional
 * convenience methods for initializing and accessing the metadata and schema
 **/
struct ParquetMetadata : public parquet::FileMetaData {
  explicit ParquetMetadata(const DataSource& input) {
    constexpr auto header_len = sizeof(parquet::file_header_s);
    constexpr auto ender_len = sizeof(parquet::file_ender_s);

    const auto len = input.size();
    const auto header_buffer = input.get_buffer(0, header_len);
    const auto header = (const parquet::file_header_s *)header_buffer->data();
    const auto ender_buffer = input.get_buffer(len - ender_len, ender_len);
    const auto ender = (const parquet::file_ender_s *)ender_buffer->data();
    CUDF_EXPECTS(len > header_len + ender_len, "Incorrect data source");
    CUDF_EXPECTS(
        header->magic == PARQUET_MAGIC && ender->magic == PARQUET_MAGIC,
        "Corrupted header or footer");
    CUDF_EXPECTS(ender->footer_len != 0 &&
                     ender->footer_len <= (len - header_len - ender_len),
                 "Incorrect footer length");

    const auto buffer = input.get_buffer(len - ender->footer_len - ender_len, ender->footer_len);
    parquet::CompactProtocolReader cp(buffer->data(), ender->footer_len);
    CUDF_EXPECTS(cp.read(this), "Cannot parse metadata");
    CUDF_EXPECTS(cp.InitSchema(this), "Cannot initialize schema");
    print_metadata();
  }

  inline int get_total_rows() const { return num_rows; }
  inline int get_num_row_groups() const { return row_groups.size(); }
  inline int get_num_columns() const { return row_groups[0].columns.size(); }

  std::string get_column_name(const std::vector<std::string> &path_in_schema) {
    std::string s = (path_in_schema.size() > 0) ? path_in_schema[0] : "";
    for (size_t i = 1; i < path_in_schema.size(); i++) {
      s += "." + path_in_schema[i];
    }
    return s;
  }

  std::vector<std::string> get_column_names() {
    std::vector<std::string> all_names;
    for (const auto &chunk : row_groups[0].columns) {
      all_names.emplace_back(get_column_name(chunk.meta_data.path_in_schema));
    }
    return all_names;
  }

  /**
   * @brief Extracts the column name used for the row indexes in a dataframe
   *
   * PANDAS adds its own metadata to the key_value section when writing out the
   * dataframe to a file to aid in exact reconstruction. The JSON-formatted
   * metadata contains the index column(s) and PANDA-specific datatypes.
   *
   * @return std::string Name of the index column
   **/
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

  /**
   * @brief Filters and reduces down to a selection of row groups
   *
   * @param[in] row_group Index of the row group to select
   * @param[in,out] row_start Starting row of the selection
   * @param[in,out] row_count Total number of rows selected
   *
   * @return List of row group indexes and its starting row
   **/
  auto select_row_groups(int row_group, int &row_start, int &row_count) {
    std::vector<std::pair<int, int>> selection;

    if (row_group != -1) {
      CUDF_EXPECTS(row_group < get_num_row_groups(), "Non-existent row group");
      for (int i = 0; i < row_group; ++i) {
        row_start += row_groups[i].num_rows;
      }
      selection.emplace_back(row_group, row_start);
      row_count = row_groups[row_group].num_rows;
    } else {
      row_start = std::max(row_start, 0);
      if (row_count == -1) {
        row_count = get_total_rows();
      }
      CUDF_EXPECTS(row_count >= 0, "Invalid row count");
      CUDF_EXPECTS(row_start <= get_total_rows(), "Invalid row start");

      for (int i = 0, count = 0; i < (int)row_groups.size(); ++i) {
        count += row_groups[i].num_rows;
        if (count > row_start || count == 0) {
          selection.emplace_back(i, count - row_groups[i].num_rows);
        }
        if (count >= (row_start + row_count)) {
          break;
        }
      }
    }

    return selection;
  }

  /**
   * @brief Filters and reduces down to a selection of columns
   *
   * @param[in] use_cols Array of column names to select
   * @param[in] use_cols_len Length of the column name array
   * @param[in] use_index_col Name of the index column
   *
   * @return A list of column names & Parquet column indexes
   **/
  auto select_columns(
      const char **use_cols, int use_cols_len, const char *use_index_col) {
    std::vector<std::pair<int, std::string>> selection;

    if (use_cols) {
      std::vector<std::string> use_names(use_cols, use_cols + use_cols_len);
      if (get_total_rows() > 0) {
        use_names.push_back(use_index_col);
      }
      for (const auto &use_name : use_names) {
        size_t index = 0;
        for (const auto name : get_column_names()) {
          if (name == use_name) {
            selection.emplace_back(index, name);
            break;
          }
          index++;
        }
      }
    } else {
      for (const auto &name : get_column_names()) {
        if (get_total_rows() > 0 || name != use_index_col) {
          selection.emplace_back(selection.size(), name);
        }
      }
    }
    CUDF_EXPECTS(selection.size() > 0, "Filtered out all columns");

    return selection;
  }

  void print_metadata() const {
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
 *
 * @return size_t The total number of pages
 **/
size_t count_page_headers(
    const hostdevice_vector<parquet::gpu::ColumnChunkDesc> &chunks) {

  size_t total_pages = 0;

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
    total_pages += chunks[c].num_data_pages + chunks[c].num_dict_pages;
  }

  return total_pages;
}

/**
 * @brief Returns the page information from the given column chunks
 *
 * @param[in] chunks List of column chunk descriptors
 * @param[in] pages List of page information
 **/
void decode_page_headers(
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
}

/**
 * @brief Decompresses the page data, at page granularity
 *
 * @param[in] chunks List of column chunk descriptors
 * @param[in] pages List of page information
 *
 * @return device_buffer<uint8_t> Device buffer to decompressed page data
 **/
device_buffer<uint8_t> decompress_page_data(
    const hostdevice_vector<parquet::gpu::ColumnChunkDesc> &chunks,
    const hostdevice_vector<parquet::gpu::PageInfo> &pages) {

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

  // Brotli scratch memory for decompressing
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
      debrotli_scratch.resize(get_gpu_debrotli_scratch_size(codec.second));
    }
  }

  LOG_PRINTF(
      "[+] Compression\n Total compressed size: %zd\n Number of "
      "compressed pages: %zd\n  gzip:    %zd \n  snappy: %zd\n",
      total_decompressed_size, num_compressed_pages, codecs[0].second,
      codecs[1].second);

  // Dispatch batches of pages to decompress for each codec
  device_buffer<uint8_t> decomp_pages(total_decompressed_size);
  hostdevice_vector<gpu_inflate_input_s> inflate_in(0, num_compressed_pages);
  hostdevice_vector<gpu_inflate_status_s> inflate_out(0, num_compressed_pages);

  size_t decomp_offset = 0;
  int32_t argc = 0;
  for (const auto &codec : codecs) {
    if (codec.second > 0) {
      int32_t start_pos = argc;

      for_each_codec_page(codec.first, [&](size_t page) {
        inflate_in[argc].srcDevice = pages[page].page_data;
        inflate_in[argc].srcSize = pages[page].compressed_page_size;
        inflate_in[argc].dstDevice = decomp_pages.data() + decomp_offset;
        inflate_in[argc].dstSize = pages[page].uncompressed_page_size;

        inflate_out[argc].bytes_written = 0;
        inflate_out[argc].status = static_cast<uint32_t>(-1000);
        inflate_out[argc].reserved = 0;

        pages[page].page_data = (uint8_t *)inflate_in[argc].dstDevice;
        decomp_offset += inflate_in[argc].dstSize;
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
          CUDF_EXPECTS(false, "Unexpected decompression dispatch");
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

  return decomp_pages;
}

/**
 * @brief Converts the page data and outputs to gdf_columns
 *
 * @param[in] chunks List of column chunk descriptors
 * @param[in] pages List of page information
 * @param[in] chunk_map Mapping between column chunk and gdf_column
 * @param[in] min_row Minimum number of rows to read from start
 * @param[in] total_rows Total number of rows to output
 **/
void decode_page_data(
    const hostdevice_vector<parquet::gpu::ColumnChunkDesc> &chunks,
    const hostdevice_vector<parquet::gpu::PageInfo> &pages,
    const std::vector<gdf_column *> &chunk_map, size_t min_row,
    size_t total_rows) {

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
                          chunks.size(), total_rows, min_row));
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
}

//
// Implementation for reading Apache ORC data and outputting to cuDF columns.
//
gdf_error read_parquet_arrow(
    pq_read_arg *args,
    std::shared_ptr<arrow::io::RandomAccessFile> rd_file) {

  DataSource input = [&] {
    if (args->source_type == FILE_PATH) {
      return DataSource(args->source);
    } else if (args->source_type == ARROW_RANDOM_ACCESS_FILE) {
      CUDF_EXPECTS(rd_file, "Invalid Arrow file handle");
      return DataSource(rd_file);
    } else {
      return DataSource(args->source, args->buffer_size);
    }
  }();
  ParquetMetadata md(input);

  // Select only row groups required
  int skip_rows = args->skip_rows;
  int num_rows = args->num_rows;
  const auto selected_row_groups =
      md.select_row_groups(args->row_group, skip_rows, num_rows);

  // Select only columns required
  const auto use_index_col = md.get_index_column_name();
  const auto selected_cols = md.select_columns(
      args->use_cols, args->use_cols_len, use_index_col.c_str());
  int num_columns = selected_cols.size();

  // Initialize gdf_columns, but hold off on allocating storage space
  LOG_PRINTF("[+] Selected row groups: %d\n", (int)selected_row_groups.size());
  LOG_PRINTF("[+] Selected columns: %d\n", num_columns);
  LOG_PRINTF("[+] Selected skip_rows: %d num_rows: %d\n", skip_rows, num_rows);
  int index_col = -1;
  std::vector<gdf_column_wrapper> columns;
  for (const auto &col : selected_cols) {
    auto row_group_0 = md.row_groups[selected_row_groups[0].first];
    auto &col_schema = md.schema[row_group_0.columns[col.first].schema_idx];
    auto dtype_info = to_dtype(col_schema.type, col_schema.converted_type,
                               args->strings_to_categorical);

    columns.emplace_back(static_cast<gdf_size_type>(num_rows),
                         dtype_info.first, dtype_info.second, col.second);

    LOG_PRINTF(" %2zd: name=%s size=%zd type=%d data=%lx valid=%lx\n",
               columns.size() - 1, columns.back()->col_name,
               (size_t)columns.back()->size, columns.back()->dtype,
               (uint64_t)columns.back()->data, (uint64_t)columns.back()->valid);

    if (col.second == use_index_col) {
      index_col = columns.size() - 1;
    }
  }

  // Descriptors for all the chunks that make up the selected columns
  const auto num_column_chunks = selected_row_groups.size() * num_columns;
  hostdevice_vector<parquet::gpu::ColumnChunkDesc> chunks(0, num_column_chunks);

  // Association between each column chunk and its gdf_column
  std::vector<gdf_column *> chunk_map(num_column_chunks);

  // Tracker for eventually deallocating compressed and uncompressed data
  std::vector<device_buffer<uint8_t>> page_data;

  // Initialize column chunk info
  LOG_PRINTF("[+] Column Chunk Description\n");
  size_t total_decompressed_size = 0;
  auto remaining_rows = num_rows;
  for (const auto &rg : selected_row_groups) {
    const auto row_group = md.row_groups[rg.first];
    const auto row_group_start = rg.second;
    const auto row_group_rows = std::min(remaining_rows, (int)row_group.num_rows);

    for (size_t i = 0; i < selected_cols.size(); i++) {
      auto col = selected_cols[i];
      auto &col_meta = row_group.columns[col.first].meta_data;
      auto &col_schema = md.schema[row_group.columns[col.first].schema_idx];
      auto &gdf_column = columns[i];

      // Spec requires each row group to contain exactly one chunk for every
      // column. If there are too many or too few, continue with best effort
      if (col.second != md.get_column_name(col_meta.path_in_schema)) {
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

      uint8_t *d_compdata = nullptr;
      if (col_meta.total_compressed_size != 0) {
        const auto offset = (col_meta.dictionary_page_offset != 0)
                                ? std::min(col_meta.data_page_offset,
                                           col_meta.dictionary_page_offset)
                                : col_meta.data_page_offset;
        page_data.emplace_back(col_meta.total_compressed_size);
        d_compdata = page_data.back().data();
        const auto buffer = input.get_buffer(offset, col_meta.total_compressed_size);
        CUDA_TRY(cudaMemcpyAsync(d_compdata, buffer->data(),
                                 col_meta.total_compressed_size,
                                 cudaMemcpyHostToDevice));
        CUDA_TRY(cudaStreamSynchronize(0));
      }
      chunks.insert(parquet::gpu::ColumnChunkDesc(
          col_meta.total_compressed_size, d_compdata, col_meta.num_values,
          col_schema.type, type_width, row_group_start, row_group_rows,
          col_schema.max_definition_level, col_schema.max_repetition_level,
          required_bits(col_schema.max_definition_level),
          required_bits(col_schema.max_repetition_level), col_meta.codec));

      LOG_PRINTF(
          " %2d: %s start_row=%d, num_rows=%d, codec=%d, "
          "num_values=%ld\n     total_compressed_size=%ld "
          "total_uncompressed_size=%ld\n     schema_idx=%d, type=%d, "
          "type_width=%d, max_def_level=%d, "
          "max_rep_level=%d\n     data_page_offset=%zd, index_page_offset=%zd, "
          "dict_page_offset=%zd\n",
          col.first, col.second.c_str(), start_row, row_group_rows,
          col_meta.codec, col_meta.num_values, col_meta.total_compressed_size,
          col_meta.total_uncompressed_size,
          row_group.columns[col.first].schema_idx,
          chunks[chunks.size() - 1].data_type, type_width,
          col_schema.max_definition_level, col_schema.max_repetition_level,
          (size_t)col_meta.data_page_offset, (size_t)col_meta.index_page_offset,
          (size_t)col_meta.dictionary_page_offset);

      // Map each column chunk to its output gdf_column
      chunk_map[chunks.size() - 1] = gdf_column.get();

      if (col_meta.codec != parquet::Compression::UNCOMPRESSED) {
        total_decompressed_size += col_meta.total_uncompressed_size;
      }
    }
    remaining_rows -= row_group.num_rows;
  }
  assert(remaining_rows <= 0);

  // Allocate output memory and convert Parquet format into cuDF format
  const auto total_pages = count_page_headers(chunks);
  if (total_pages > 0) {
    hostdevice_vector<parquet::gpu::PageInfo> pages(total_pages, total_pages);

    decode_page_headers(chunks, pages);
    if (total_decompressed_size > 0) {
      auto decomp_page_data = decompress_page_data(chunks, pages);
      page_data.clear();
      page_data.push_back(std::move(decomp_page_data));
    }
    for (auto &column : columns) {
      CUDF_EXPECTS(column.allocate() == GDF_SUCCESS, "Cannot allocate columns");
    }
    decode_page_data(chunks, pages, chunk_map, skip_rows, num_rows);
  } else {
    // Columns' data's memory is still expected for an empty dataframe
    for (auto &column : columns) {
      CUDF_EXPECTS(column.allocate() == GDF_SUCCESS, "Cannot allocate columns");
    }
  }

  // For string dtype, allocate an NvStrings container instance, deallocating
  // the original string list memory in the process.
  // This container takes a list of string pointers and lengths, and copies
  // into its own memory so the source memory must not be released yet.
  for (auto &column : columns) {
    if (column->dtype == GDF_STRING) {
      using str_pair = std::pair<const char *, size_t>;
      using str_ptr = std::unique_ptr<NVStrings, decltype(&NVStrings::destroy)>;

      auto str_list = static_cast<str_pair *>(column->data);
      str_ptr str_data(NVStrings::create_from_index(str_list, num_rows),
                       &NVStrings::destroy);
      RMM_FREE(std::exchange(column->data, str_data.release()), 0);
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
  } else {
    args->index_col = nullptr;
  }

  return GDF_SUCCESS;
}

gdf_error read_parquet(pq_read_arg *args) {
  return read_parquet_arrow(args, nullptr);
}
