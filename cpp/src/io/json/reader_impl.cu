/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
 * @brief cuDF-IO JSON reader class implementation
 **/

#include "reader_impl.hpp"

#include <rmm/thrust_rmm_allocator.h>

#include <cudf/detail/utilities/trie.cuh>
#include <cudf/io/readers.hpp>
#include <cudf/utilities/error.hpp>

#include <io/comp/io_uncomp.h>
#include <io/utilities/parsing_utils.cuh>
#include <io/utilities/type_conversion.cuh>

#include <cudf/table/table.hpp>

namespace cudf {
namespace io {
namespace detail {
namespace json {
using namespace cudf::io;

namespace {
/**
 * @brief Extract value names from a JSON object
 *
 * @param[in] json_obj Host vector containing the JSON object
 * @param[in] opts Parsing options (e.g. delimiter and quotation character)
 *
 * @return std::vector<std::string> names of JSON object values
 **/
std::vector<std::string> get_names_from_json_object(const std::vector<char> &json_obj,
                                                    const ParseOptions &opts)
{
  enum class ParseState { preColName, colName, postColName };
  std::vector<std::string> names;
  bool quotation = false;
  auto state     = ParseState::preColName;
  int name_start = 0;
  for (size_t pos = 0; pos < json_obj.size(); ++pos) {
    if (state == ParseState::preColName) {
      if (json_obj[pos] == opts.quotechar) {
        name_start = pos + 1;
        state      = ParseState::colName;
        continue;
      }
    } else if (state == ParseState::colName) {
      if (json_obj[pos] == opts.quotechar && json_obj[pos - 1] != '\\') {
        // if found a non-escaped quote character, it's the end of the column name
        names.emplace_back(&json_obj[name_start], &json_obj[pos]);
        state = ParseState::postColName;
        continue;
      }
    } else if (state == ParseState::postColName) {
      // TODO handle complex data types that might include unquoted commas
      if (!quotation && json_obj[pos] == opts.delimiter) {
        state = ParseState::preColName;
        continue;
      } else if (json_obj[pos] == opts.quotechar) {
        quotation = !quotation;
      }
    }
  }
  return names;
}

/**
 * @brief Estimates the maximum expected length or a row, based on the number
 * of columns
 *
 * If the number of columns is not available, it will return a value large
 * enough for most use cases
 *
 * @param[in] num_columns Number of columns in the JSON file (optional)
 *
 * @return Estimated maximum size of a row, in bytes
 **/
constexpr size_t calculate_max_row_size(int num_columns = 0) noexcept
{
  constexpr size_t max_row_bytes = 16 * 1024;  // 16KB
  constexpr size_t column_bytes  = 64;
  constexpr size_t base_padding  = 1024;  // 1KB
  if (num_columns == 0) {
    // Use flat size if the number of columns is not known
    return max_row_bytes;
  } else {
    // Expand the size based on the number of columns, if available
    return base_padding + num_columns * column_bytes;
  }
}

}  // anonymous namespace

/**
 * @brief Ingest input JSON file/buffer, without decompression
 *
 * Sets the source_, byte_range_offset_, and byte_range_size_ data members
 *
 * @param[in] range_offset Number of bytes offset from the start
 * @param[in] range_size Bytes to read; use `0` for all remaining data
 *
 * @return void
 **/
void reader::impl::ingest_raw_input(size_t range_offset, size_t range_size)
{
  size_t map_range_size = 0;
  if (range_size != 0) { map_range_size = range_size + calculate_max_row_size(args_.dtype.size()); }

  // Support delayed opening of the file if using memory mapping datasource
  // This allows only mapping of a subset of the file if using byte range
  if (source_ == nullptr) {
    assert(!filepath_.empty());
    source_ = datasource::create(filepath_, range_offset, map_range_size);
  }

  if (!source_->is_empty()) {
    auto data_size = (map_range_size != 0) ? map_range_size : source_->size();
    buffer_        = source_->host_read(range_offset, data_size);
  }

  byte_range_offset_ = range_offset;
  byte_range_size_   = range_size;
  load_whole_file_   = byte_range_offset_ == 0 && byte_range_size_ == 0;
}

/**
 * @brief Decompress the input data, if needed
 *
 * Sets the uncomp_data_ and uncomp_size_ data members
 * Loads the data into device memory if byte range parameters are not used
 *
 * @return void
 **/
void reader::impl::decompress_input()
{
  const auto compression_type = infer_compression_type(
    args_.compression, filepath_, {{"gz", "gzip"}, {"zip", "zip"}, {"bz2", "bz2"}, {"xz", "xz"}});
  if (compression_type == "none") {
    // Do not use the owner vector here to avoid extra copy
    uncomp_data_ = reinterpret_cast<const char *>(buffer_->data());
    uncomp_size_ = buffer_->size();
  } else {
    getUncompressedHostData(reinterpret_cast<const char *>(buffer_->data()),
                            buffer_->size(),
                            compression_type,
                            uncomp_data_owner_);
    uncomp_data_ = uncomp_data_owner_.data();
    uncomp_size_ = uncomp_data_owner_.size();
  }
  if (load_whole_file_) data_ = rmm::device_buffer(uncomp_data_, uncomp_size_);
}

/**
 * @brief Finds all record starts in the file and stores them in rec_starts_
 *
 * Does not upload the entire file to the GPU
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 *
 * @return void
 **/
void reader::impl::set_record_starts(cudaStream_t stream)
{
  std::vector<char> chars_to_count{'\n'};
  // Currently, ignoring lineterminations within quotes is handled by recording the records of both,
  // and then filtering out the records that is a quotechar or a linetermination within a quotechar
  // pair.
  if (allow_newlines_in_strings_) { chars_to_count.push_back('\"'); }
  // If not starting at an offset, add an extra row to account for the first row in the file
  cudf::size_type prefilter_count = ((byte_range_offset_ == 0) ? 1 : 0);
  if (load_whole_file_) {
    prefilter_count += count_all_from_set(data_, chars_to_count);
  } else {
    prefilter_count += count_all_from_set(uncomp_data_, uncomp_size_, chars_to_count);
  }

  rec_starts_.resize(prefilter_count);

  auto *find_result_ptr = rec_starts_.data().get();
  // Manually adding an extra row to account for the first row in the file
  if (byte_range_offset_ == 0) {
    find_result_ptr++;
    CUDA_TRY(cudaMemsetAsync(rec_starts_.data().get(), 0ull, sizeof(uint64_t), stream));
  }

  std::vector<char> chars_to_find{'\n'};
  if (allow_newlines_in_strings_) { chars_to_find.push_back('\"'); }
  // Passing offset = 1 to return positions AFTER the found character
  if (load_whole_file_) {
    find_all_from_set(data_, chars_to_find, 1, find_result_ptr);
  } else {
    find_all_from_set(uncomp_data_, uncomp_size_, chars_to_find, 1, find_result_ptr);
  }

  // Previous call stores the record pinput_file.typeositions as encountered by all threads
  // Sort the record positions as subsequent processing may require filtering
  // certain rows or other processing on specific records
  thrust::sort(rmm::exec_policy()->on(0), rec_starts_.begin(), rec_starts_.end());

  auto filtered_count = prefilter_count;
  if (allow_newlines_in_strings_) {
    thrust::host_vector<uint64_t> h_rec_starts = rec_starts_;
    bool quotation                             = false;
    for (cudf::size_type i = 1; i < prefilter_count; ++i) {
      if (uncomp_data_[h_rec_starts[i] - 1] == '\"') {
        quotation       = !quotation;
        h_rec_starts[i] = uncomp_size_;
        filtered_count--;
      } else if (quotation) {
        h_rec_starts[i] = uncomp_size_;
        filtered_count--;
      }
    }

    rec_starts_ = h_rec_starts;
    thrust::sort(rmm::exec_policy()->on(0), rec_starts_.begin(), rec_starts_.end());
  }

  // Exclude the ending newline as it does not precede a record start
  if (uncomp_data_[uncomp_size_ - 1] == '\n') { filtered_count--; }

  rec_starts_.resize(filtered_count);
}

/**
 * @brief Uploads the relevant segment of the input json data onto the GPU.
 *
 * Sets the d_data_ data member.
 * Only rows that need to be parsed are copied, based on the byte range
 * Also updates the array of record starts to match the device data offset.
 *
 * @return void
 **/
void reader::impl::upload_data_to_device()
{
  size_t start_offset = 0;
  size_t end_offset   = uncomp_size_;

  // Trim lines that are outside range
  if (byte_range_size_ != 0 || byte_range_offset_ != 0) {
    thrust::host_vector<uint64_t> h_rec_starts = rec_starts_;

    if (byte_range_size_ != 0) {
      auto it = h_rec_starts.end() - 1;
      while (it >= h_rec_starts.begin() && *it > byte_range_size_) {
        end_offset = *it;
        --it;
      }
      h_rec_starts.erase(it + 1, h_rec_starts.end());
    }

    // Resize to exclude rows outside of the range
    // Adjust row start positions to account for the data subcopy
    start_offset = h_rec_starts.front();
    rec_starts_.resize(h_rec_starts.size());
    thrust::transform(rmm::exec_policy()->on(0),
                      rec_starts_.begin(),
                      rec_starts_.end(),
                      thrust::make_constant_iterator(start_offset),
                      rec_starts_.begin(),
                      thrust::minus<uint64_t>());
  }

  const size_t bytes_to_upload = end_offset - start_offset;
  CUDF_EXPECTS(bytes_to_upload <= uncomp_size_,
               "Error finding the record within the specified byte range.\n");

  // Upload the raw data that is within the rows of interest
  data_ = rmm::device_buffer(uncomp_data_ + start_offset, bytes_to_upload);
}

/**
 * @brief Parse the first row to set the column name
 *
 * Sets the column_names_ data member
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 *
 * @return void
 **/
void reader::impl::set_column_names(cudaStream_t stream)
{
  // If file only contains one row, use the file size for the row size
  uint64_t first_row_len = data_.size() / sizeof(char);
  if (rec_starts_.size() > 1) {
    // Set first_row_len to the offset of the second row, if it exists
    CUDA_TRY(cudaMemcpyAsync(&first_row_len,
                             rec_starts_.data().get() + 1,
                             sizeof(uint64_t),
                             cudaMemcpyDeviceToHost,
                             stream));
  }
  std::vector<char> first_row(first_row_len);
  CUDA_TRY(cudaMemcpyAsync(
    first_row.data(), data_.data(), first_row_len * sizeof(char), cudaMemcpyDeviceToHost, stream));
  CUDA_TRY(cudaStreamSynchronize(stream));

  // Determine the row format between:
  //   JSON array - [val1, val2, ...] and
  //   JSON object - {"col1":val1, "col2":val2, ...}
  // based on the top level opening bracket
  const auto first_square_bracket = std::find(first_row.begin(), first_row.end(), '[');
  const auto first_curly_bracket  = std::find(first_row.begin(), first_row.end(), '{');
  CUDF_EXPECTS(first_curly_bracket != first_row.end() || first_square_bracket != first_row.end(),
               "Input data is not a valid JSON file.");
  // If the first opening bracket is '{', assume object format
  const bool is_object = first_curly_bracket < first_square_bracket;
  if (is_object) {
    metadata.column_names = get_names_from_json_object(first_row, opts_);
  } else {
    int cols_found = 0;
    bool quotation = false;
    for (size_t pos = 0; pos < first_row.size(); ++pos) {
      // Flip the quotation flag if current character is a quotechar
      if (first_row[pos] == opts_.quotechar) {
        quotation = !quotation;
      }
      // Check if end of a column/row
      else if (pos == first_row.size() - 1 || (!quotation && first_row[pos] == opts_.delimiter)) {
        metadata.column_names.emplace_back(std::to_string(cols_found++));
      }
    }
  }
}

/**
 * @brief Set the data type array data member
 *
 * If user does not pass the data types, deduces types from the file content
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 *
 * @return void
 **/
void reader::impl::set_data_types(cudaStream_t stream)
{
  if (!args_.dtype.empty()) {
    CUDF_EXPECTS(args_.dtype.size() == metadata.column_names.size(),
                 "Need to specify the type of each column.\n");
    // Assume that the dtype is in dictionary format only if all elements contain a colon
    const bool is_dict =
      std::all_of(args_.dtype.begin(), args_.dtype.end(), [](const std::string &s) {
        return std::find(s.begin(), s.end(), ':') != s.end();
      });
    if (is_dict) {
      std::map<std::string, data_type> col_type_map;

      for (const auto &ts : args_.dtype) {
        const size_t colon_idx = ts.find(":");
        const std::string col_name(ts.begin(), ts.begin() + colon_idx);
        const std::string type_str(ts.begin() + colon_idx + 1, ts.end());

        col_type_map[col_name] = convert_string_to_dtype(type_str);
      }

      // Using the map here allows O(n log n) complexity
      for (size_t col = 0; col < args_.dtype.size(); ++col) {
        dtypes_.push_back(col_type_map[metadata.column_names[col]]);
        // dtypes_extra_info_.push_back(col_type_info_map[column_names_[col]]);
      }
    } else {
      auto dtype_ = std::back_inserter(dtypes_);
      // auto dtype_info_ = std::back_inserter(dtypes_extra_info_);
      for (size_t col = 0; col < args_.dtype.size(); ++col) {
        // std::tie(dtype_, dtype_info_) = convertStringToDtype(args_.dtype[col]);
        dtype_ = convert_string_to_dtype(args_.dtype[col]);
      }
    }
  } else {
    CUDF_EXPECTS(rec_starts_.size() != 0, "No data available for data type inference.\n");
    const auto num_columns = metadata.column_names.size();

    rmm::device_vector<cudf::io::json::ColumnInfo> d_column_infos(num_columns,
                                                                  cudf::io::json::ColumnInfo{});
    cudf::io::json::gpu::detect_data_types(d_column_infos.data().get(),
                                           static_cast<const char *>(data_.data()),
                                           data_.size(),
                                           opts_,
                                           num_columns,
                                           rec_starts_.data().get(),
                                           rec_starts_.size(),
                                           stream);
    thrust::host_vector<cudf::io::json::ColumnInfo> h_column_infos = d_column_infos;

    for (const auto &cinfo : h_column_infos) {
      if (cinfo.null_count == static_cast<int>(rec_starts_.size())) {
        // Entire column is NULL; allocate the smallest amount of memory
        dtypes_.push_back(data_type(type_id::INT8));
      } else if (cinfo.string_count > 0) {
        dtypes_.push_back(data_type(type_id::STRING));
      } else if (cinfo.datetime_count > 0) {
        dtypes_.push_back(data_type(type_id::TIMESTAMP_MILLISECONDS));
      } else if (cinfo.float_count > 0 || (cinfo.int_count > 0 && cinfo.null_count > 0)) {
        dtypes_.push_back(data_type(type_id::FLOAT64));
      } else if (cinfo.int_count > 0) {
        dtypes_.push_back(data_type(type_id::INT64));
      } else if (cinfo.bool_count > 0) {
        dtypes_.push_back(data_type(type_id::BOOL8));
      } else {
        CUDF_FAIL("Data type detection failed.\n");
      }
    }
  }
}

/**
 * @brief Parse the input data and store results a table
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 *
 * @return table_with_metadata struct
 **/
table_with_metadata reader::impl::convert_data_to_table(cudaStream_t stream)
{
  const auto num_columns = dtypes_.size();
  const auto num_records = rec_starts_.size();

  // alloc output buffers.
  std::vector<column_buffer> out_buffers;
  for (size_t col = 0; col < num_columns; ++col) {
    out_buffers.emplace_back(dtypes_[col], num_records, true, stream, mr_);
  }

  thrust::host_vector<data_type> h_dtypes(num_columns);
  thrust::host_vector<void *> h_data(num_columns);
  thrust::host_vector<bitmask_type *> h_valid(num_columns);

  for (size_t i = 0; i < num_columns; ++i) {
    h_dtypes[i] = dtypes_[i];
    h_data[i]   = out_buffers[i].data();
    h_valid[i]  = out_buffers[i].null_mask();
  }

  rmm::device_vector<data_type> d_dtypes           = h_dtypes;
  rmm::device_vector<void *> d_data                = h_data;
  rmm::device_vector<cudf::bitmask_type *> d_valid = h_valid;
  rmm::device_vector<cudf::size_type> d_valid_counts(num_columns, 0);

  cudf::io::json::gpu::convert_json_to_columns(data_,
                                               d_dtypes.data().get(),
                                               d_data.data().get(),
                                               num_records,
                                               num_columns,
                                               rec_starts_.data().get(),
                                               d_valid.data().get(),
                                               d_valid_counts.data().get(),
                                               opts_,
                                               stream);
  CUDA_TRY(cudaStreamSynchronize(stream));
  CUDA_TRY(cudaGetLastError());

  // postprocess columns
  thrust::host_vector<cudf::size_type> h_valid_counts = d_valid_counts;
  std::vector<std::unique_ptr<column>> out_columns;
  for (size_t i = 0; i < num_columns; ++i) {
    out_buffers[i].null_count() = num_records - h_valid_counts[i];

    out_columns.emplace_back(make_column(dtypes_[i], num_records, out_buffers[i]));
  }

  CUDF_EXPECTS(!out_columns.empty(), "Error converting json input into gdf columns.\n");

  return table_with_metadata{std::make_unique<table>(std::move(out_columns)), metadata};
}

reader::impl::impl(std::unique_ptr<datasource> source,
                   std::string filepath,
                   reader_options const &options,
                   rmm::mr::device_memory_resource *mr)
  : source_(std::move(source)), filepath_(filepath), args_(options), mr_(mr)
{
  CUDF_EXPECTS(args_.lines, "Only JSON Lines format is currently supported.\n");

  d_true_trie_         = createSerializedTrie({"true"});
  opts_.trueValuesTrie = d_true_trie_.data().get();

  d_false_trie_         = createSerializedTrie({"false"});
  opts_.falseValuesTrie = d_false_trie_.data().get();

  d_na_trie_         = createSerializedTrie({"null"});
  opts_.naValuesTrie = d_na_trie_.data().get();

  opts_.dayfirst = options.dayfirst;
}

/**
 * @brief Read an entire set or a subset of data from the source
 *
 * @param[in] range_offset Number of bytes offset from the start
 * @param[in] range_size Bytes to read; use `0` for all remaining data
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 *
 * @return Unique pointer to the table data
 **/
table_with_metadata reader::impl::read(size_t range_offset, size_t range_size, cudaStream_t stream)
{
  ingest_raw_input(range_offset, range_size);
  CUDF_EXPECTS(buffer_ != nullptr, "Ingest failed: input data is null.\n");

  decompress_input();
  CUDF_EXPECTS(uncomp_data_ != nullptr, "Ingest failed: uncompressed input data is null.\n");
  CUDF_EXPECTS(uncomp_size_ != 0, "Ingest failed: uncompressed input data has zero size.\n");

  set_record_starts(stream);
  CUDF_EXPECTS(!rec_starts_.empty(), "Error enumerating records.\n");

  upload_data_to_device();
  CUDF_EXPECTS(data_.size() != 0, "Error uploading input data to the GPU.\n");

  set_column_names(stream);
  CUDF_EXPECTS(!metadata.column_names.empty(), "Error determining column names.\n");

  set_data_types(stream);
  CUDF_EXPECTS(!dtypes_.empty(), "Error in data type detection.\n");

  return convert_data_to_table(stream);
}

// Forward to implementation
reader::reader(std::string filepath,
               reader_options const &options,
               rmm::mr::device_memory_resource *mr)
  : _impl(std::make_unique<impl>(nullptr, filepath, options, mr))
{
  // Delay actual instantiation of data source until read to allow for
  // partial memory mapping of file using byte ranges
}

// Forward to implementation
reader::reader(std::unique_ptr<cudf::io::datasource> source,
               reader_options const &options,
               rmm::mr::device_memory_resource *mr)
  : _impl(std::make_unique<impl>(std::move(source), "", options, mr))
{
}

// Destructor within this translation unit
reader::~reader() = default;

// Forward to implementation
table_with_metadata reader::read_all(cudaStream_t stream)
{
  return table_with_metadata{_impl->read(0, 0, stream)};
}

// Forward to implementation
table_with_metadata reader::read_byte_range(size_t offset, size_t size, cudaStream_t stream)
{
  return table_with_metadata{_impl->read(offset, size, stream)};
}

}  // namespace json
}  // namespace detail
}  // namespace io
}  // namespace cudf
