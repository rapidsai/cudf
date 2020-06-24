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

/**
 * @file reader_impl.cu
 * @brief cuDF-IO Parquet reader class implementation
 */

#include "reader_impl.hpp"

#include <io/comp/gpuinflate.h>

#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/device_buffer.hpp>

#include <algorithm>
#include <array>
#include <regex>

namespace cudf {
namespace io {
namespace detail {
namespace parquet {
// Import functionality that's independent of legacy code
using namespace cudf::io::parquet;
using namespace cudf::io;

namespace {
/**
 * @brief Function that translates Parquet datatype to cuDF type enum
 */
constexpr type_id to_type_id(parquet::Type physical,
                             parquet::ConvertedType logical,
                             bool strings_to_categorical,
                             type_id timestamp_type_id,
                             int32_t decimal_scale)
{
  // Logical type used for actual data interpretation; the legacy converted type
  // is superceded by 'logical' type whenever available.
  switch (logical) {
    case parquet::UINT_8: return type_id::UINT8;
    case parquet::INT_8: return type_id::INT8;
    case parquet::UINT_16: return type_id::UINT16;
    case parquet::INT_16: return type_id::INT16;
    case parquet::UINT_32: return type_id::UINT32;
    case parquet::UINT_64: return type_id::UINT64;
    case parquet::DATE: return type_id::TIMESTAMP_DAYS;
    case parquet::TIMESTAMP_MICROS:
      return (timestamp_type_id != type_id::EMPTY) ? timestamp_type_id
                                                   : type_id::TIMESTAMP_MICROSECONDS;
    case parquet::TIMESTAMP_MILLIS:
      return (timestamp_type_id != type_id::EMPTY) ? timestamp_type_id
                                                   : type_id::TIMESTAMP_MILLISECONDS;
    case parquet::DECIMAL:
      if (decimal_scale != 0 || (physical != parquet::INT32 && physical != parquet::INT64)) {
        return type_id::FLOAT64;
      }
      break;
    default: break;
  }

  // Physical storage type supported by Parquet; controls the on-disk storage
  // format in combination with the encoding type.
  switch (physical) {
    case parquet::BOOLEAN: return type_id::BOOL8;
    case parquet::INT32: return type_id::INT32;
    case parquet::INT64: return type_id::INT64;
    case parquet::FLOAT: return type_id::FLOAT32;
    case parquet::DOUBLE: return type_id::FLOAT64;
    case parquet::BYTE_ARRAY:
    case parquet::FIXED_LEN_BYTE_ARRAY:
      // Can be mapped to INT32 (32-bit hash) or STRING
      return strings_to_categorical ? type_id::INT32 : type_id::STRING;
    case parquet::INT96:
      return (timestamp_type_id != type_id::EMPTY) ? timestamp_type_id
                                                   : type_id::TIMESTAMP_NANOSECONDS;
    default: break;
  }

  return type_id::EMPTY;
}

/**
 * @brief Function that translates cuDF time unit to Parquet clock frequency
 */
constexpr int32_t to_clockrate(type_id timestamp_type_id)
{
  switch (timestamp_type_id) {
    case type_id::TIMESTAMP_SECONDS: return 1;
    case type_id::TIMESTAMP_MILLISECONDS: return 1000;
    case type_id::TIMESTAMP_MICROSECONDS: return 1000000;
    case type_id::TIMESTAMP_NANOSECONDS: return 1000000000;
    default: return 0;
  }
}

/**
 * @brief Function that returns the required the number of bits to store a value
 */
template <typename T = uint8_t>
T required_bits(uint32_t max_level)
{
  return static_cast<T>(CompactProtocolReader::NumRequiredBits(max_level));
}

std::tuple<int32_t, int32_t, int8_t> conversion_info(type_id column_type_id,
                                                     type_id timestamp_type_id,
                                                     parquet::Type physical,
                                                     int8_t converted,
                                                     int32_t length)
{
  int32_t type_width = (physical == parquet::FIXED_LEN_BYTE_ARRAY) ? length : 0;
  int32_t clock_rate = 0;
  if (column_type_id == type_id::INT8 or column_type_id == type_id::UINT8) {
    type_width = 1;  // I32 -> I8
  } else if (column_type_id == type_id::INT16 or column_type_id == type_id::UINT16) {
    type_width = 2;  // I32 -> I16
  } else if (column_type_id == type_id::INT32) {
    type_width = 4;  // str -> hash32
  } else if (is_timestamp(data_type{column_type_id})) {
    clock_rate = to_clockrate(timestamp_type_id);
  }

  int8_t converted_type = converted;
  if (converted_type == parquet::DECIMAL && column_type_id != type_id::FLOAT64) {
    converted_type = parquet::UNKNOWN;  // Not converting to float64
  }

  return std::make_tuple(type_width, clock_rate, converted_type);
}

}  // namespace

/**
 * @brief Class for parsing dataset metadata
 */
struct metadata : public FileMetaData {
  explicit metadata(datasource *source)
  {
    constexpr auto header_len = sizeof(file_header_s);
    constexpr auto ender_len  = sizeof(file_ender_s);

    const auto len           = source->size();
    const auto header_buffer = source->host_read(0, header_len);
    const auto header        = (const file_header_s *)header_buffer->data();
    const auto ender_buffer  = source->host_read(len - ender_len, ender_len);
    const auto ender         = (const file_ender_s *)ender_buffer->data();
    CUDF_EXPECTS(len > header_len + ender_len, "Incorrect data source");
    CUDF_EXPECTS(header->magic == PARQUET_MAGIC && ender->magic == PARQUET_MAGIC,
                 "Corrupted header or footer");
    CUDF_EXPECTS(ender->footer_len != 0 && ender->footer_len <= (len - header_len - ender_len),
                 "Incorrect footer length");

    const auto buffer = source->host_read(len - ender->footer_len - ender_len, ender->footer_len);
    CompactProtocolReader cp(buffer->data(), ender->footer_len);
    CUDF_EXPECTS(cp.read(this), "Cannot parse metadata");
    CUDF_EXPECTS(cp.InitSchema(this), "Cannot initialize schema");
  }

  inline int64_t get_total_rows() const { return num_rows; }
  inline int get_num_row_groups() const { return row_groups.size(); }
  inline int get_num_columns() const { return row_groups[0].columns.size(); }

  std::string get_column_name(const std::vector<std::string> &path_in_schema)
  {
    std::string s = (path_in_schema.size() > 0) ? path_in_schema[0] : "";
    for (size_t i = 1; i < path_in_schema.size(); i++) { s += "." + path_in_schema[i]; }
    return s;
  }

  std::vector<std::string> get_column_names()
  {
    std::vector<std::string> all_names;
    if (row_groups.size() != 0) {
      for (const auto &chunk : row_groups[0].columns) {
        all_names.emplace_back(get_column_name(chunk.meta_data.path_in_schema));
      }
    }
    return all_names;
  }

  /**
   * @brief Extracts the pandas "index_columns" section
   *
   * PANDAS adds its own metadata to the key_value section when writing out the
   * dataframe to a file to aid in exact reconstruction. The JSON-formatted
   * metadata contains the index column(s) and PANDA-specific datatypes.
   *
   * @return comma-separated index column names in quotes
   */
  std::string get_pandas_index()
  {
    auto it = std::find_if(key_value_metadata.begin(),
                           key_value_metadata.end(),
                           [](const auto &item) { return item.key == "pandas"; });
    if (it != key_value_metadata.end()) {
      // Captures a list of quoted strings found inside square brackets after `"index_columns":`
      // Inside quotes supports newlines, brackets, escaped quotes, etc.
      // One-liner regex:
      // "index_columns"\s*:\s*\[\s*((?:"(?:|(?:.*?(?![^\\]")).?)[^\\]?",?\s*)*)\]
      // Documented below.
      std::regex index_columns_expr{
        R"("index_columns"\s*:\s*\[\s*)"  // match preamble, opening square bracket, whitespace
        R"(()"                            // Open first capturing group
        R"((?:")"                         // Open non-capturing group match opening quote
        R"((?:|(?:.*?(?![^\\]")).?))"     // match empty string or anything between quotes
        R"([^\\]?")"                      // Match closing non-escaped quote
        R"(,?\s*)"                        // Match optional comma and whitespace
        R"()*)"                           // Close non-capturing group and repeat 0 or more times
        R"())"                            // Close first capturing group
        R"(\])"                           // Match closing square brackets
      };
      std::smatch sm;
      if (std::regex_search(it->value, sm, index_columns_expr)) { return std::move(sm[1].str()); }
    }
    return "";
  }

  /**
   * @brief Extracts the column name(s) used for the row indexes in a dataframe
   *
   * @param names List of column names to load, where index column name(s) will be added
   */
  void add_pandas_index_names(std::vector<std::string> &names)
  {
    auto str = get_pandas_index();
    if (str.length() != 0) {
      std::regex index_name_expr{R"(\"((?:\\.|[^\"])*)\")"};
      std::smatch sm;
      while (std::regex_search(str, sm, index_name_expr)) {
        if (sm.size() == 2) {  // 2 = whole match, first item
          if (std::find(names.begin(), names.end(), sm[1].str()) == names.end()) {
            std::regex esc_quote{R"(\\")"};
            names.emplace_back(std::move(std::regex_replace(sm[1].str(), esc_quote, R"(")")));
          }
        }
        str = sm.suffix();
      }
    }
  }

  /**
   * @brief Filters and reduces down to a selection of row groups
   *
   * @param row_group Index of the row group to select
   * @param max_rowgroup_count Max number of consecutive row groups if > 0
   * @param row_group_indices Arbitrary rowgroup list[max_rowgroup_count] if non-null
   * @param row_start Starting row of the selection
   * @param row_count Total number of rows selected
   *
   * @return List of row group indexes and its starting row
   */
  auto select_row_groups(size_type row_group,
                         size_type max_rowgroup_count,
                         const size_type *row_group_indices,
                         size_type &row_start,
                         size_type &row_count)
  {
    std::vector<std::pair<size_type, size_t>> selection;

    if (row_group_indices) {
      row_count = 0;
      for (size_type i = 0; i < max_rowgroup_count; i++) {
        auto rowgroup_idx = row_group_indices[i];
        CUDF_EXPECTS(rowgroup_idx >= 0 && rowgroup_idx < get_num_row_groups(),
                     "Invalid rowgroup index");
        selection.emplace_back(rowgroup_idx, row_count);
        row_count += row_groups[rowgroup_idx].num_rows;
      }
    } else if (row_group != -1) {
      CUDF_EXPECTS(row_group < get_num_row_groups(), "Non-existent row group");
      row_count = 0;
      do {
        selection.emplace_back(row_group, row_start + row_count);
        row_count += row_groups[row_group].num_rows;
      } while (--max_rowgroup_count > 0 && ++row_group < get_num_row_groups());
    } else {
      row_start = std::max(row_start, 0);
      if (row_count < 0) {
        row_count = static_cast<size_type>(
          std::min<int64_t>(get_total_rows(), std::numeric_limits<size_type>::max()));
      }
      CUDF_EXPECTS(row_count >= 0, "Invalid row count");
      CUDF_EXPECTS(row_start <= get_total_rows(), "Invalid row start");

      for (size_t i = 0, count = 0; i < row_groups.size(); ++i) {
        size_t chunk_start_row = count;
        count += row_groups[i].num_rows;
        if (count > static_cast<size_t>(row_start) || count == 0) {
          selection.emplace_back(i, chunk_start_row);
        }
        if (count >= static_cast<size_t>(row_start) + static_cast<size_t>(row_count)) { break; }
      }
    }

    return selection;
  }

  /**
   * @brief Filters and reduces down to a selection of columns
   *
   * @param use_names List of column names to select
   * @param include_index Whether to always include the PANDAS index column(s)
   *
   * @return List of column names
   */
  auto select_columns(std::vector<std::string> use_names, bool include_index)
  {
    std::vector<std::pair<int, std::string>> selection;

    const auto names = get_column_names();
    if (use_names.empty()) {
      // No columns specified; include all in the dataset
      for (const auto &name : names) { selection.emplace_back(selection.size(), name); }
    } else {
      // Load subset of columns; include PANDAS index unless excluded
      if (include_index) { add_pandas_index_names(use_names); }
      for (const auto &use_name : use_names) {
        for (size_t i = 0; i < names.size(); ++i) {
          if (names[i] == use_name) {
            selection.emplace_back(i, names[i]);
            break;
          }
        }
      }
    }

    return selection;
  }
};

void reader::impl::read_column_chunks(std::vector<rmm::device_buffer> &page_data,
                                      hostdevice_vector<gpu::ColumnChunkDesc> &chunks,
                                      size_t begin_chunk,
                                      size_t end_chunk,
                                      const std::vector<size_t> &column_chunk_offsets,
                                      cudaStream_t stream)
{
  // Transfer chunk data, coalescing adjacent chunks
  for (size_t chunk = begin_chunk; chunk < end_chunk;) {
    const size_t io_offset   = column_chunk_offsets[chunk];
    size_t io_size           = chunks[chunk].compressed_size;
    size_t next_chunk        = chunk + 1;
    const bool is_compressed = (chunks[chunk].codec != parquet::Compression::UNCOMPRESSED);
    while (next_chunk < end_chunk) {
      const size_t next_offset = column_chunk_offsets[next_chunk];
      const bool is_next_compressed =
        (chunks[next_chunk].codec != parquet::Compression::UNCOMPRESSED);
      if (next_offset != io_offset + io_size || is_next_compressed != is_compressed) {
        // Can't merge if not contiguous or mixing compressed and uncompressed
        // Not coalescing uncompressed with compressed chunks is so that compressed buffers can be
        // freed earlier (immediately after decompression stage) to limit peak memory requirements
        break;
      }
      io_size += chunks[next_chunk].compressed_size;
      next_chunk++;
    }
    if (io_size != 0) {
      auto buffer         = _source->host_read(io_offset, io_size);
      page_data[chunk]    = rmm::device_buffer(buffer->data(), buffer->size(), stream);
      uint8_t *d_compdata = reinterpret_cast<uint8_t *>(page_data[chunk].data());
      do {
        chunks[chunk].compressed_data = d_compdata;
        d_compdata += chunks[chunk].compressed_size;
      } while (++chunk != next_chunk);
    } else {
      chunk = next_chunk;
    }
  }
}

size_t reader::impl::count_page_headers(hostdevice_vector<gpu::ColumnChunkDesc> &chunks,
                                        cudaStream_t stream)
{
  size_t total_pages = 0;

  CUDA_TRY(cudaMemcpyAsync(
    chunks.device_ptr(), chunks.host_ptr(), chunks.memory_size(), cudaMemcpyHostToDevice, stream));
  CUDA_TRY(gpu::DecodePageHeaders(chunks.device_ptr(), chunks.size(), stream));
  CUDA_TRY(cudaMemcpyAsync(
    chunks.host_ptr(), chunks.device_ptr(), chunks.memory_size(), cudaMemcpyDeviceToHost, stream));
  CUDA_TRY(cudaStreamSynchronize(stream));

  for (size_t c = 0; c < chunks.size(); c++) {
    total_pages += chunks[c].num_data_pages + chunks[c].num_dict_pages;
  }

  return total_pages;
}

void reader::impl::decode_page_headers(hostdevice_vector<gpu::ColumnChunkDesc> &chunks,
                                       hostdevice_vector<gpu::PageInfo> &pages,
                                       cudaStream_t stream)
{
  for (size_t c = 0, page_count = 0; c < chunks.size(); c++) {
    chunks[c].max_num_pages = chunks[c].num_data_pages + chunks[c].num_dict_pages;
    chunks[c].page_info     = pages.device_ptr(page_count);
    page_count += chunks[c].max_num_pages;
  }

  CUDA_TRY(cudaMemcpyAsync(
    chunks.device_ptr(), chunks.host_ptr(), chunks.memory_size(), cudaMemcpyHostToDevice, stream));
  CUDA_TRY(gpu::DecodePageHeaders(chunks.device_ptr(), chunks.size(), stream));
  CUDA_TRY(cudaMemcpyAsync(
    pages.host_ptr(), pages.device_ptr(), pages.memory_size(), cudaMemcpyDeviceToHost, stream));
  CUDA_TRY(cudaStreamSynchronize(stream));
}

rmm::device_buffer reader::impl::decompress_page_data(
  hostdevice_vector<gpu::ColumnChunkDesc> &chunks,
  hostdevice_vector<gpu::PageInfo> &pages,
  cudaStream_t stream)
{
  auto for_each_codec_page = [&](parquet::Compression codec, const std::function<void(size_t)> &f) {
    for (size_t c = 0, page_count = 0; c < chunks.size(); c++) {
      const auto page_stride = chunks[c].max_num_pages;
      if (chunks[c].codec == codec) {
        for (int k = 0; k < page_stride; k++) { f(page_count + k); }
      }
      page_count += page_stride;
    }
  };

  // Brotli scratch memory for decompressing
  rmm::device_vector<uint8_t> debrotli_scratch;

  // Count the exact number of compressed pages
  size_t num_comp_pages    = 0;
  size_t total_decomp_size = 0;
  std::array<std::pair<parquet::Compression, size_t>, 3> codecs{std::make_pair(parquet::GZIP, 0),
                                                                std::make_pair(parquet::SNAPPY, 0),
                                                                std::make_pair(parquet::BROTLI, 0)};

  for (auto &codec : codecs) {
    for_each_codec_page(codec.first, [&](size_t page) {
      total_decomp_size += pages[page].uncompressed_page_size;
      codec.second++;
      num_comp_pages++;
    });
    if (codec.first == parquet::BROTLI && codec.second > 0) {
      debrotli_scratch.resize(get_gpu_debrotli_scratch_size(codec.second));
    }
  }

  // Dispatch batches of pages to decompress for each codec
  rmm::device_buffer decomp_pages(total_decomp_size, stream);
  hostdevice_vector<gpu_inflate_input_s> inflate_in(0, num_comp_pages, stream);
  hostdevice_vector<gpu_inflate_status_s> inflate_out(0, num_comp_pages, stream);

  size_t decomp_offset = 0;
  int32_t argc         = 0;
  for (const auto &codec : codecs) {
    if (codec.second > 0) {
      int32_t start_pos = argc;

      for_each_codec_page(codec.first, [&](size_t page) {
        auto dst_base              = static_cast<uint8_t *>(decomp_pages.data());
        inflate_in[argc].srcDevice = pages[page].page_data;
        inflate_in[argc].srcSize   = pages[page].compressed_page_size;
        inflate_in[argc].dstDevice = dst_base + decomp_offset;
        inflate_in[argc].dstSize   = pages[page].uncompressed_page_size;

        inflate_out[argc].bytes_written = 0;
        inflate_out[argc].status        = static_cast<uint32_t>(-1000);
        inflate_out[argc].reserved      = 0;

        pages[page].page_data = (uint8_t *)inflate_in[argc].dstDevice;
        decomp_offset += inflate_in[argc].dstSize;
        argc++;
      });

      CUDA_TRY(cudaMemcpyAsync(inflate_in.device_ptr(start_pos),
                               inflate_in.host_ptr(start_pos),
                               sizeof(decltype(inflate_in)::value_type) * (argc - start_pos),
                               cudaMemcpyHostToDevice,
                               stream));
      CUDA_TRY(cudaMemcpyAsync(inflate_out.device_ptr(start_pos),
                               inflate_out.host_ptr(start_pos),
                               sizeof(decltype(inflate_out)::value_type) * (argc - start_pos),
                               cudaMemcpyHostToDevice,
                               stream));
      switch (codec.first) {
        case parquet::GZIP:
          CUDA_TRY(gpuinflate(inflate_in.device_ptr(start_pos),
                              inflate_out.device_ptr(start_pos),
                              argc - start_pos,
                              1,
                              stream))
          break;
        case parquet::SNAPPY:
          CUDA_TRY(gpu_unsnap(inflate_in.device_ptr(start_pos),
                              inflate_out.device_ptr(start_pos),
                              argc - start_pos,
                              stream));
          break;
        case parquet::BROTLI:
          CUDA_TRY(gpu_debrotli(inflate_in.device_ptr(start_pos),
                                inflate_out.device_ptr(start_pos),
                                debrotli_scratch.data().get(),
                                debrotli_scratch.size(),
                                argc - start_pos,
                                stream));
          break;
        default: CUDF_EXPECTS(false, "Unexpected decompression dispatch"); break;
      }
      CUDA_TRY(cudaMemcpyAsync(inflate_out.host_ptr(start_pos),
                               inflate_out.device_ptr(start_pos),
                               sizeof(decltype(inflate_out)::value_type) * (argc - start_pos),
                               cudaMemcpyDeviceToHost,
                               stream));
    }
  }
  CUDA_TRY(cudaStreamSynchronize(stream));

  // Update the page information in device memory with the updated value of
  // page_data; it now points to the uncompressed data buffer
  CUDA_TRY(cudaMemcpyAsync(
    pages.device_ptr(), pages.host_ptr(), pages.memory_size(), cudaMemcpyHostToDevice, stream));

  return decomp_pages;
}

void reader::impl::decode_page_data(hostdevice_vector<gpu::ColumnChunkDesc> &chunks,
                                    hostdevice_vector<gpu::PageInfo> &pages,
                                    size_t min_row,
                                    size_t total_rows,
                                    const std::vector<int> &chunk_map,
                                    std::vector<column_buffer> &out_buffers,
                                    cudaStream_t stream)
{
  auto is_dict_chunk = [](const gpu::ColumnChunkDesc &chunk) {
    return (chunk.data_type & 0x7) == BYTE_ARRAY && chunk.num_dict_pages > 0;
  };

  // Count the number of string dictionary entries
  // NOTE: Assumes first page in the chunk is always the dictionary page
  size_t total_str_dict_indexes = 0;
  for (size_t c = 0, page_count = 0; c < chunks.size(); c++) {
    if (is_dict_chunk(chunks[c])) { total_str_dict_indexes += pages[page_count].num_values; }
    page_count += chunks[c].max_num_pages;
  }

  // Build index for string dictionaries since they can't be indexed
  // directly due to variable-sized elements
  rmm::device_vector<gpu::nvstrdesc_s> str_dict_index;
  if (total_str_dict_indexes > 0) { str_dict_index.resize(total_str_dict_indexes); }

  // Update chunks with pointers to column data
  for (size_t c = 0, page_count = 0, str_ofs = 0; c < chunks.size(); c++) {
    if (is_dict_chunk(chunks[c])) {
      chunks[c].str_dict_index = str_dict_index.data().get() + str_ofs;
      str_ofs += pages[page_count].num_values;
    }
    chunks[c].column_data_base = out_buffers[chunk_map[c]].data();
    chunks[c].valid_map_base   = out_buffers[chunk_map[c]].null_mask();
    page_count += chunks[c].max_num_pages;
  }

  CUDA_TRY(cudaMemcpyAsync(
    chunks.device_ptr(), chunks.host_ptr(), chunks.memory_size(), cudaMemcpyHostToDevice, stream));
  if (total_str_dict_indexes > 0) {
    CUDA_TRY(gpu::BuildStringDictionaryIndex(chunks.device_ptr(), chunks.size(), stream));
  }
  CUDA_TRY(gpu::DecodePageData(pages.device_ptr(),
                               pages.size(),
                               chunks.device_ptr(),
                               chunks.size(),
                               total_rows,
                               min_row,
                               stream));
  CUDA_TRY(cudaMemcpyAsync(
    pages.host_ptr(), pages.device_ptr(), pages.memory_size(), cudaMemcpyDeviceToHost, stream));
  CUDA_TRY(cudaStreamSynchronize(stream));

  for (size_t i = 0; i < pages.size(); i++) {
    if (pages[i].num_rows > 0) {
      const size_t c = pages[i].chunk_idx;
      if (c < chunks.size()) {
        out_buffers[chunk_map[c]].null_count() += pages[i].num_rows - pages[i].valid_count;
      }
    }
  }
}

reader::impl::impl(std::unique_ptr<datasource> source,
                   reader_options const &options,
                   rmm::mr::device_memory_resource *mr)
  : _source(std::move(source)), _mr(mr)
{
  // Open and parse the source dataset metadata
  _metadata = std::make_unique<metadata>(_source.get());

  // Select only columns required by the options
  _selected_columns = _metadata->select_columns(options.columns, options.use_pandas_metadata);

  // Override output timestamp resolution if requested
  if (options.timestamp_type.id() != type_id::EMPTY) { _timestamp_type = options.timestamp_type; }

  // Strings may be returned as either string or categorical columns
  _strings_to_categorical = options.strings_to_categorical;
}

table_with_metadata reader::impl::read(size_type skip_rows,
                                       size_type num_rows,
                                       size_type row_group,
                                       size_type max_rowgroup_count,
                                       const size_type *row_group_indices,
                                       cudaStream_t stream)
{
  std::vector<std::unique_ptr<column>> out_columns;
  table_metadata out_metadata;

  // Select only row groups required
  const auto selected_row_groups = _metadata->select_row_groups(
    row_group, max_rowgroup_count, row_group_indices, skip_rows, num_rows);

  // Get a list of column data types
  std::vector<data_type> column_types;
  if (_metadata->row_groups.size() != 0) {
    for (const auto &col : _selected_columns) {
      auto &col_schema = _metadata->schema[_metadata->row_groups[0].columns[col.first].schema_idx];
      auto col_type    = to_type_id(col_schema.type,
                                 col_schema.converted_type,
                                 _strings_to_categorical,
                                 _timestamp_type.id(),
                                 col_schema.decimal_scale);
      CUDF_EXPECTS(col_type != type_id::EMPTY, "Unknown type");
      column_types.emplace_back(col_type);
    }
  }
  out_columns.reserve(column_types.size());

  if (selected_row_groups.size() != 0 && column_types.size() != 0) {
    // Descriptors for all the chunks that make up the selected columns
    const auto num_columns = _selected_columns.size();
    const auto num_chunks  = selected_row_groups.size() * num_columns;
    hostdevice_vector<gpu::ColumnChunkDesc> chunks(0, num_chunks, stream);

    // Association between each column chunk and its column
    std::vector<int> chunk_map(num_chunks);

    // Tracker for eventually deallocating compressed and uncompressed data
    std::vector<rmm::device_buffer> page_data(num_chunks);

    // Keep track of column chunk file offsets
    std::vector<size_t> column_chunk_offsets(num_chunks);

    // Initialize column chunk information
    size_t total_decompressed_size = 0;
    auto remaining_rows            = num_rows;
    for (const auto &rg : selected_row_groups) {
      const auto &row_group = _metadata->row_groups[rg.first];
      auto row_group_start  = rg.second;
      auto row_group_rows   = std::min<int>(remaining_rows, row_group.num_rows);
      auto io_chunk_idx     = chunks.size();

      for (size_t i = 0; i < num_columns; ++i) {
        auto col         = _selected_columns[i];
        auto &col_meta   = row_group.columns[col.first].meta_data;
        auto &col_schema = _metadata->schema[row_group.columns[col.first].schema_idx];

        // Spec requires each row group to contain exactly one chunk for every
        // column. If there are too many or too few, continue with best effort
        if (col.second != _metadata->get_column_name(col_meta.path_in_schema)) {
          std::cerr << "Detected mismatched column chunk" << std::endl;
          continue;
        }
        if (chunks.size() >= chunks.max_size()) {
          std::cerr << "Detected too many column chunks" << std::endl;
          continue;
        }

        int32_t type_width;
        int32_t clock_rate;
        int8_t converted_type;
        std::tie(type_width, clock_rate, converted_type) =
          conversion_info(column_types[i].id(),
                          _timestamp_type.id(),
                          col_schema.type,
                          col_schema.converted_type,
                          col_schema.type_length);

        column_chunk_offsets[chunks.size()] =
          (col_meta.dictionary_page_offset != 0)
            ? std::min(col_meta.data_page_offset, col_meta.dictionary_page_offset)
            : col_meta.data_page_offset;

        chunks.insert(gpu::ColumnChunkDesc(col_meta.total_compressed_size,
                                           nullptr,
                                           col_meta.num_values,
                                           col_schema.type,
                                           type_width,
                                           row_group_start,
                                           row_group_rows,
                                           col_schema.max_definition_level,
                                           col_schema.max_repetition_level,
                                           required_bits(col_schema.max_definition_level),
                                           required_bits(col_schema.max_repetition_level),
                                           col_meta.codec,
                                           converted_type,
                                           col_schema.decimal_scale,
                                           clock_rate));

        // Map each column chunk to its column index
        chunk_map[chunks.size() - 1] = i;

        if (col_meta.codec != Compression::UNCOMPRESSED) {
          total_decompressed_size += col_meta.total_uncompressed_size;
        }
      }
      // Read compressed chunk data to device memory
      read_column_chunks(
        page_data, chunks, io_chunk_idx, chunks.size(), column_chunk_offsets, stream);

      remaining_rows -= row_group.num_rows;
    }
    assert(remaining_rows <= 0);

    // Process dataset chunk pages into output columns
    const auto total_pages = count_page_headers(chunks, stream);
    if (total_pages > 0) {
      hostdevice_vector<gpu::PageInfo> pages(total_pages, total_pages, stream);
      rmm::device_buffer decomp_page_data;

      decode_page_headers(chunks, pages, stream);
      if (total_decompressed_size > 0) {
        decomp_page_data = decompress_page_data(chunks, pages, stream);
        // Free compressed data
        for (size_t c = 0; c < chunks.size(); c++) {
          if (chunks[c].codec != parquet::Compression::UNCOMPRESSED && page_data[c].size() != 0) {
            page_data[c].resize(0);
            page_data[c].shrink_to_fit();
          }
        }
      }

      std::vector<column_buffer> out_buffers;
      out_buffers.reserve(column_types.size());
      for (size_t i = 0; i < column_types.size(); ++i) {
        auto col = _selected_columns[i];
        auto &col_schema =
          _metadata->schema
            [_metadata->row_groups[selected_row_groups[0].first].columns[col.first].schema_idx];
        bool is_nullable = (col_schema.max_definition_level != 0);
        out_buffers.emplace_back(column_types[i], num_rows, is_nullable, stream, _mr);
      }

      decode_page_data(chunks, pages, skip_rows, num_rows, chunk_map, out_buffers, stream);

      for (size_t i = 0; i < column_types.size(); ++i) {
        out_columns.emplace_back(
          make_column(column_types[i], num_rows, out_buffers[i], stream, _mr));
      }
    }
  }

  // Create empty columns as needed
  for (size_t i = out_columns.size(); i < column_types.size(); ++i) {
    out_columns.emplace_back(make_empty_column(column_types[i]));
  }

  // Return column names (must match order of returned columns)
  out_metadata.column_names.resize(_selected_columns.size());
  for (size_t i = 0; i < _selected_columns.size(); i++) {
    out_metadata.column_names[i] = _selected_columns[i].second;
  }
  // Return user metadata
  for (const auto &kv : _metadata->key_value_metadata) {
    out_metadata.user_data.insert({kv.key, kv.value});
  }

  return {std::make_unique<table>(std::move(out_columns)), std::move(out_metadata)};
}

// Forward to implementation
reader::reader(std::string filepath,
               reader_options const &options,
               rmm::mr::device_memory_resource *mr)
  : _impl(std::make_unique<impl>(datasource::create(filepath), options, mr))
{
}

// Forward to implementation
reader::reader(std::unique_ptr<cudf::io::datasource> source,
               reader_options const &options,
               rmm::mr::device_memory_resource *mr)
  : _impl(std::make_unique<impl>(std::move(source), options, mr))
{
}

// Destructor within this translation unit
reader::~reader() = default;

// Forward to implementation
table_with_metadata reader::read_all(cudaStream_t stream)
{
  return _impl->read(0, -1, -1, -1, nullptr, stream);
}

// Forward to implementation
table_with_metadata reader::read_row_group(size_type row_group,
                                           size_type row_group_count,
                                           cudaStream_t stream)
{
  return _impl->read(0, -1, row_group, row_group_count, nullptr, stream);
}

// Forward to implementation
table_with_metadata reader::read_row_groups(const std::vector<size_type> &row_group_list,
                                            cudaStream_t stream)
{
  return _impl->read(
    0, -1, -1, static_cast<size_type>(row_group_list.size()), row_group_list.data(), stream);
}

// Forward to implementation
table_with_metadata reader::read_rows(size_type skip_rows, size_type num_rows, cudaStream_t stream)
{
  return _impl->read(skip_rows, (num_rows != 0) ? num_rows : -1, -1, -1, nullptr, stream);
}

}  // namespace parquet
}  // namespace detail
}  // namespace io
}  // namespace cudf
