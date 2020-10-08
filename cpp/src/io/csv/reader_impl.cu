/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
 * @brief cuDF-IO CSV reader class implementation
 **/

#include "reader_impl.hpp"

#include <io/comp/io_uncomp.h>
#include <io/utilities/parsing_utils.cuh>
#include <io/utilities/type_conversion.cuh>

#include <cudf/io/types.hpp>
#include <cudf/strings/replace.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <tuple>
#include <unordered_map>

using std::string;
using std::vector;

using cudf::detail::device_span;
using cudf::detail::host_span;

namespace cudf {
namespace io {
namespace detail {
namespace csv {
using namespace cudf::io::csv;
using namespace cudf::io;

/**
 * @brief Estimates the maximum expected length or a row, based on the number
 * of columns
 *
 * If the number of columns is not available, it will return a value large
 * enough for most use cases
 *
 * @param[in] num_columns Number of columns in the CSV file (optional)
 *
 * @return Estimated maximum size of a row, in bytes
 **/
constexpr size_t calculateMaxRowSize(int num_columns = 0) noexcept
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

/**
 * @brief Translates a dtype string and returns its dtype enumeration and any
 * extended dtype flags that are supported by cuIO. Often, this is a column
 * with the same underlying dtype the basic types, but with different parsing
 * interpretations.
 *
 * @param[in] dtype String containing the basic or extended dtype
 *
 * @return Tuple of data_type and flags
 */
std::tuple<data_type, column_parse::flags> get_dtype_info(const std::string &dtype)
{
  if (dtype == "hex" || dtype == "hex64") {
    return std::make_tuple(data_type{cudf::type_id::INT64}, column_parse::as_hexadecimal);
  }
  if (dtype == "hex32") {
    return std::make_tuple(data_type{cudf::type_id::INT32}, column_parse::as_hexadecimal);
  }

  return std::make_tuple(convert_string_to_dtype(dtype), column_parse::as_default);
}

/**
 * @brief Removes the first and Last quote in the string
 */
string removeQuotes(string str, char quotechar)
{
  // Exclude first and last quotation char
  const size_t first_quote = str.find(quotechar);
  if (first_quote != string::npos) { str.erase(first_quote, 1); }
  const size_t last_quote = str.rfind(quotechar);
  if (last_quote != string::npos) { str.erase(last_quote, 1); }

  return str;
}

/**
 * @brief Parse the first row to set the column names in the raw_csv parameter.
 * The first row can be either the header row, or the first data row
 */
std::vector<std::string> setColumnNames(std::vector<char> const &header,
                                        ParseOptions const &opts,
                                        int header_row,
                                        std::string prefix)
{
  std::vector<std::string> col_names;

  // If there is only a single character then it would be the terminator
  if (header.size() <= 1) { return col_names; }

  std::vector<char> first_row = header;
  int num_cols                = 0;

  bool quotation = false;
  for (size_t pos = 0, prev = 0; pos < first_row.size(); ++pos) {
    // Flip the quotation flag if current character is a quotechar
    if (first_row[pos] == opts.quotechar) {
      quotation = !quotation;
    }
    // Check if end of a column/row
    else if (pos == first_row.size() - 1 || (!quotation && first_row[pos] == opts.terminator) ||
             (!quotation && first_row[pos] == opts.delimiter)) {
      // This is the header, add the column name
      if (header_row >= 0) {
        // Include the current character, in case the line is not terminated
        int col_name_len = pos - prev + 1;
        // Exclude the delimiter/terminator is present
        if (first_row[pos] == opts.delimiter || first_row[pos] == opts.terminator) {
          --col_name_len;
        }
        // Also exclude '\r' character at the end of the column name if it's
        // part of the terminator
        if (col_name_len > 0 && opts.terminator == '\n' && first_row[pos] == '\n' &&
            first_row[pos - 1] == '\r') {
          --col_name_len;
        }

        const string new_col_name(first_row.data() + prev, col_name_len);
        col_names.push_back(removeQuotes(new_col_name, opts.quotechar));

        // Stop parsing when we hit the line terminator; relevant when there is
        // a blank line following the header. In this case, first_row includes
        // multiple line terminators at the end, as the new recStart belongs to
        // a line that comes after the blank line(s)
        if (!quotation && first_row[pos] == opts.terminator) { break; }
      } else {
        // This is the first data row, add the automatically generated name
        col_names.push_back(prefix + std::to_string(num_cols));
      }
      num_cols++;

      // Skip adjacent delimiters if delim_whitespace is set
      while (opts.multi_delimiter && pos < first_row.size() && first_row[pos] == opts.delimiter &&
             first_row[pos + 1] == opts.delimiter) {
        ++pos;
      }
      prev = pos + 1;
    }
  }

  return col_names;
}

table_with_metadata reader::impl::read(cudaStream_t stream)
{
  auto range_offset  = opts_.get_byte_range_offset();
  auto range_size    = opts_.get_byte_range_size();
  auto skip_rows     = opts_.get_skiprows();
  auto skip_end_rows = opts_.get_skipfooter();
  auto num_rows      = opts_.get_nrows();

  if (range_offset > 0 || range_size > 0) {
    CUDF_EXPECTS(compression_type_ == "none",
                 "Reading compressed data using `byte range` is unsupported");
  }
  size_t map_range_size = 0;
  if (range_size != 0) {
    const auto num_columns = std::max(opts_.get_names().size(), opts_.get_dtypes().size());
    map_range_size         = range_size + calculateMaxRowSize(num_columns);
  }

  // Support delayed opening of the file if using memory mapping datasource
  // This allows only mapping of a subset of the file if using byte range
  if (source_ == nullptr) {
    assert(!filepath_.empty());
    source_ = datasource::create(filepath_, range_offset, map_range_size);
  }

  // Return an empty dataframe if no data and no column metadata to process
  if (source_->is_empty() && (opts_.get_names().empty() || opts_.get_dtypes().empty())) {
    return {std::make_unique<table>(), {}};
  }

  // Transfer source data to GPU
  if (!source_->is_empty()) {
    auto data_size = (map_range_size != 0) ? map_range_size : source_->size();
    auto buffer    = source_->host_read(range_offset, data_size);

    auto h_data = host_span<char const>(  //
      reinterpret_cast<const char *>(buffer->data()),
      buffer->size());

    std::vector<char> h_uncomp_data_owner;

    if (compression_type_ != "none") {
      h_uncomp_data_owner = get_uncompressed_data(h_data, compression_type_);
      h_data              = h_uncomp_data_owner;
    }
    // None of the parameters for row selection is used, we are parsing the entire file
    const bool load_whole_file = range_offset == 0 && range_size == 0 && skip_rows <= 0 &&
                                 skip_end_rows <= 0 && num_rows == -1;

    // With byte range, find the start of the first data row
    size_t const data_start_offset = (range_offset != 0) ? find_first_row_start(h_data) : 0;

    // TODO: Allow parsing the header outside the mapped range
    CUDF_EXPECTS((range_offset == 0 || opts_.get_header() < 0),
                 "byte_range offset with header not supported");

    // Gather row offsets
    gather_row_offsets(h_data,
                       data_start_offset,
                       (range_size) ? range_size : h_data.size(),
                       (skip_rows > 0) ? skip_rows : 0,
                       num_rows,
                       load_whole_file,
                       stream);

    // Exclude the rows that are to be skipped from the end
    if (skip_end_rows > 0 && static_cast<size_t>(skip_end_rows) < row_offsets_.size()) {
      row_offsets_.resize(row_offsets_.size() - skip_end_rows);
    }

    // Exclude the end-of-data row from number of rows with actual data
    num_records_ = row_offsets_.size();
    num_records_ -= (num_records_ > 0);
  } else {
    num_records_ = 0;
  }

  // Check if the user gave us a list of column names
  if (not opts_.get_names().empty()) {
    h_column_flags_.resize(opts_.get_names().size(), column_parse::enabled);
    col_names_ = opts_.get_names();
  } else {
    col_names_ = setColumnNames(header_, opts, opts_.get_header(), opts_.get_prefix());

    num_actual_cols_ = num_active_cols_ = col_names_.size();

    h_column_flags_.resize(num_actual_cols_, column_parse::enabled);

    // Rename empty column names to "Unnamed: col_index"
    for (size_t col_idx = 0; col_idx < col_names_.size(); ++col_idx) {
      if (col_names_[col_idx].empty()) {
        col_names_[col_idx] = string("Unnamed: ") + std::to_string(col_idx);
      }
    }

    // Looking for duplicates
    std::unordered_map<string, int> col_names_histogram;
    for (auto &col_name : col_names_) {
      // Operator [] inserts a default-initialized value if the given key is not
      // present
      if (++col_names_histogram[col_name] > 1) {
        if (opts_.is_enabled_mangle_dupe_cols()) {
          // Rename duplicates of column X as X.1, X.2, ...; First appearance
          // stays as X
          col_name += "." + std::to_string(col_names_histogram[col_name] - 1);
        } else {
          // All duplicate columns will be ignored; First appearance is parsed
          const auto idx       = &col_name - col_names_.data();
          h_column_flags_[idx] = column_parse::disabled;
        }
      }
    }

    // Update the number of columns to be processed, if some might have been
    // removed
    if (!opts_.is_enabled_mangle_dupe_cols()) { num_active_cols_ = col_names_histogram.size(); }
  }

  // User can specify which columns should be parsed
  if (!opts_.get_use_cols_indexes().empty() || !opts_.get_use_cols_names().empty()) {
    std::fill(h_column_flags_.begin(), h_column_flags_.end(), column_parse::disabled);

    for (const auto index : opts_.get_use_cols_indexes()) {
      h_column_flags_[index] = column_parse::enabled;
    }
    num_active_cols_ = opts_.get_use_cols_indexes().size();

    for (const auto &name : opts_.get_use_cols_names()) {
      const auto it = std::find(col_names_.begin(), col_names_.end(), name);
      if (it != col_names_.end()) {
        h_column_flags_[it - col_names_.begin()] = column_parse::enabled;
        num_active_cols_++;
      }
    }
  }

  // User can specify which columns should be inferred as datetime
  if (!opts_.get_infer_date_indexes().empty() || !opts_.get_infer_date_names().empty()) {
    for (const auto index : opts_.get_infer_date_indexes()) {
      h_column_flags_[index] |= column_parse::as_datetime;
    }

    for (const auto &name : opts_.get_infer_date_names()) {
      auto it = std::find(col_names_.begin(), col_names_.end(), name);
      if (it != col_names_.end()) {
        h_column_flags_[it - col_names_.begin()] |= column_parse::as_datetime;
      }
    }
  }

  // Return empty table rather than exception if nothing to load
  if (num_active_cols_ == 0) { return {std::make_unique<table>(), {}}; }

  auto metadata     = table_metadata{};
  auto out_columns  = std::vector<std::unique_ptr<cudf::column>>();
  auto column_types = gather_column_types(stream);

  out_columns.reserve(column_types.size());

  if (num_records_ != 0) {
    auto out_buffers = decode_data(column_types, stream);
    for (size_t i = 0; i < column_types.size(); ++i) {
      metadata.column_names.emplace_back(out_buffers[i].name);
      if (column_types[i].id() == type_id::STRING && opts.quotechar != '\0' &&
          opts.doublequote == true) {
        // PANDAS' default behavior of enabling doublequote for two consecutive
        // quotechars in quoted fields results in reduction to a single quotechar
        // TODO: Would be much more efficient to perform this operation in-place
        // during the conversion stage
        const std::string quotechar(1, opts.quotechar);
        const std::string dblquotechar(2, opts.quotechar);
        std::unique_ptr<column> col = make_strings_column(out_buffers[i]._strings, stream);
        out_columns.emplace_back(
          cudf::strings::replace(col->view(), dblquotechar, quotechar, -1, mr_));
      } else {
        out_columns.emplace_back(make_column(out_buffers[i], stream, mr_));
      }
    }
  } else {
    // Create empty columns
    for (size_t i = 0; i < column_types.size(); ++i) {
      out_columns.emplace_back(make_empty_column(column_types[i]));
    }
    // Handle empty metadata
    for (int col = 0; col < num_actual_cols_; ++col) {
      if (h_column_flags_[col] & column_parse::enabled) {
        metadata.column_names.emplace_back(col_names_[col]);
      }
    }
  }
  return {std::make_unique<table>(std::move(out_columns)), std::move(metadata)};
}

size_t reader::impl::find_first_row_start(host_span<char const> const data)
{
  // For now, look for the first terminator (assume the first terminator isn't within a quote)
  // TODO: Attempt to infer this from the data
  size_t pos = 0;
  while (pos < data.size() && data[pos] != opts.terminator) { ++pos; }
  return std::min(pos + 1, data.size());
}

void reader::impl::gather_row_offsets(host_span<char const> const data,
                                      size_t range_begin,
                                      size_t range_end,
                                      size_t skip_rows,
                                      int64_t num_rows,
                                      bool load_whole_file,
                                      cudaStream_t stream)
{
  constexpr size_t max_chunk_bytes = 64 * 1024 * 1024;  // 64MB
  size_t buffer_size               = std::min(max_chunk_bytes, data.size());
  size_t max_blocks =
    std::max<size_t>((buffer_size / cudf::io::csv::gpu::rowofs_block_bytes) + 1, 2);
  hostdevice_vector<uint64_t> row_ctx(max_blocks);
  size_t buffer_pos  = std::min(range_begin - std::min(range_begin, sizeof(char)), data.size());
  size_t pos         = std::min(range_begin, data.size());
  size_t header_rows = (opts_.get_header() >= 0) ? opts_.get_header() + 1 : 0;
  uint64_t ctx       = 0;

  // For compatibility with the previous parser, a row is considered in-range if the
  // previous row terminator is within the given range
  range_end += (range_end < data.size());
  data_.resize(0);
  row_offsets_.resize(0);
  data_.reserve((load_whole_file) ? data.size() : std::min(buffer_size * 2, data.size()));
  do {
    size_t target_pos = std::min(pos + max_chunk_bytes, data.size());
    size_t chunk_size = target_pos - pos;

    data_.insert(data_.end(), data.begin() + buffer_pos + data_.size(), data.begin() + target_pos);

    // Pass 1: Count the potential number of rows in each character block for each
    // possible parser state at the beginning of the block.
    uint32_t num_blocks = cudf::io::csv::gpu::gather_row_offsets(opts,
                                                                 row_ctx.device_ptr(),
                                                                 device_span<uint64_t>(),
                                                                 data_,
                                                                 chunk_size,
                                                                 pos,
                                                                 buffer_pos,
                                                                 data.size(),
                                                                 range_begin,
                                                                 range_end,
                                                                 skip_rows,
                                                                 stream);
    CUDA_TRY(cudaMemcpyAsync(row_ctx.host_ptr(),
                             row_ctx.device_ptr(),
                             num_blocks * sizeof(uint64_t),
                             cudaMemcpyDeviceToHost,
                             stream));
    CUDA_TRY(cudaStreamSynchronize(stream));
    // Sum up the rows in each character block, selecting the row count that
    // corresponds to the current input context. Also stores the now known input
    // context per character block that will be needed by the second pass.
    for (uint32_t i = 0; i < num_blocks; i++) {
      uint64_t ctx_next = cudf::io::csv::gpu::select_row_context(ctx, row_ctx[i]);
      row_ctx[i]        = ctx;
      ctx               = ctx_next;
    }
    size_t total_rows = ctx >> 2;
    if (total_rows > skip_rows) {
      // At least one row in range in this batch
      row_offsets_.resize(total_rows - skip_rows);

      CUDA_TRY(cudaMemcpyAsync(row_ctx.device_ptr(),
                               row_ctx.host_ptr(),
                               num_blocks * sizeof(uint64_t),
                               cudaMemcpyHostToDevice,
                               stream));

      // Pass 2: Output row offsets
      cudf::io::csv::gpu::gather_row_offsets(opts,
                                             row_ctx.device_ptr(),
                                             row_offsets_,
                                             data_,
                                             chunk_size,
                                             pos,
                                             buffer_pos,
                                             data.size(),
                                             range_begin,
                                             range_end,
                                             skip_rows,
                                             stream);
      // With byte range, we want to keep only one row out of the specified range
      if (range_end < data.size()) {
        CUDA_TRY(cudaMemcpyAsync(row_ctx.host_ptr(),
                                 row_ctx.device_ptr(),
                                 num_blocks * sizeof(uint64_t),
                                 cudaMemcpyDeviceToHost,
                                 stream));
        CUDA_TRY(cudaStreamSynchronize(stream));
        size_t rows_out_of_range = 0;
        for (uint32_t i = 0; i < num_blocks; i++) { rows_out_of_range += row_ctx[i]; }
        if (rows_out_of_range != 0) {
          // Keep one row out of range (used to infer length of previous row)
          auto new_row_offsets_size =
            row_offsets_.size() - std::min(rows_out_of_range - 1, row_offsets_.size());
          row_offsets_.resize(new_row_offsets_size);
          // Implies we reached the end of the range
          break;
        }
      }
      // num_rows does not include blank rows
      if (num_rows >= 0) {
        if (row_offsets_.size() > header_rows + static_cast<size_t>(num_rows)) {
          size_t num_blanks =
            cudf::io::csv::gpu::count_blank_rows(opts, data_, row_offsets_, stream);
          if (row_offsets_.size() - num_blanks > header_rows + static_cast<size_t>(num_rows)) {
            // Got the desired number of rows
            break;
          }
        }
      }
    } else {
      // Discard data (all rows below skip_rows), keeping one character for history
      size_t discard_bytes = std::max(data_.size(), sizeof(char)) - sizeof(char);
      if (discard_bytes != 0) {
        data_.erase(data_.begin(), data_.begin() + discard_bytes);
        buffer_pos += discard_bytes;
      }
    }
    pos = target_pos;
  } while (pos < data.size());

  // Eliminate blank rows
  if (row_offsets_.size() != 0) {
    cudf::io::csv::gpu::remove_blank_rows(opts, data_, row_offsets_, stream);
  }
  // Remove header rows and extract header
  const size_t header_row_index = std::max<size_t>(header_rows, 1) - 1;
  if (header_row_index + 1 < row_offsets_.size()) {
    CUDA_TRY(cudaMemcpyAsync(row_ctx.host_ptr(),
                             row_offsets_.data().get() + header_row_index,
                             2 * sizeof(uint64_t),
                             cudaMemcpyDeviceToHost,
                             stream));
    CUDA_TRY(cudaStreamSynchronize(stream));
    const auto header_start = buffer_pos + row_ctx[0];
    const auto header_end   = buffer_pos + row_ctx[1];
    CUDF_EXPECTS(header_start <= header_end && header_end <= data.size(),
                 "Invalid csv header location");
    header_.assign(data.begin() + header_start, data.begin() + header_end);
    if (header_rows > 0) {
      row_offsets_.erase(row_offsets_.begin(), row_offsets_.begin() + header_rows);
    }
  }
  // Apply num_rows limit
  if (num_rows >= 0) { row_offsets_.resize(std::min<size_t>(row_offsets_.size(), num_rows + 1)); }
}

std::vector<data_type> reader::impl::gather_column_types(cudaStream_t stream)
{
  std::vector<data_type> dtypes;

  if (opts_.get_dtypes().empty()) {
    if (num_records_ == 0) {
      dtypes.resize(num_active_cols_, data_type{type_id::EMPTY});
    } else {
      d_column_flags_ = h_column_flags_;

      auto column_stats = cudf::io::csv::gpu::detect_column_types(
        opts, data_, d_column_flags_, row_offsets_, num_active_cols_, stream);

      CUDA_TRY(cudaStreamSynchronize(stream));

      for (int col = 0; col < num_active_cols_; col++) {
        unsigned long long countInt = column_stats[col].countInt8 + column_stats[col].countInt16 +
                                      column_stats[col].countInt32 + column_stats[col].countInt64;

        if (column_stats[col].countNULL == num_records_) {
          // Entire column is NULL; allocate the smallest amount of memory
          dtypes.emplace_back(cudf::type_id::INT8);
        } else if (column_stats[col].countString > 0L) {
          dtypes.emplace_back(cudf::type_id::STRING);
        } else if (column_stats[col].countDateAndTime > 0L) {
          dtypes.emplace_back(cudf::type_id::TIMESTAMP_NANOSECONDS);
        } else if (column_stats[col].countBool > 0L) {
          dtypes.emplace_back(cudf::type_id::BOOL8);
        } else if (column_stats[col].countFloat > 0L ||
                   (column_stats[col].countFloat == 0L && countInt > 0L &&
                    column_stats[col].countNULL > 0L)) {
          // The second condition has been added to conform to
          // PANDAS which states that a column of integers with
          // a single NULL record need to be treated as floats.
          dtypes.emplace_back(cudf::type_id::FLOAT64);
        } else {
          // All other integers are stored as 64-bit to conform to PANDAS
          dtypes.emplace_back(cudf::type_id::INT64);
        }
      }
    }
  } else {
    const bool is_dict =
      std::all_of(opts_.get_dtypes().begin(), opts_.get_dtypes().end(), [](const auto &s) {
        return s.find(':') != std::string::npos;
      });

    if (!is_dict) {
      if (opts_.get_dtypes().size() == 1) {
        // If it's a single dtype, assign that dtype to all active columns
        data_type dtype_;
        column_parse::flags col_flags_;
        std::tie(dtype_, col_flags_) = get_dtype_info(opts_.get_dtypes()[0]);
        dtypes.resize(num_active_cols_, dtype_);
        for (int col = 0; col < num_actual_cols_; col++) { h_column_flags_[col] |= col_flags_; }
        CUDF_EXPECTS(dtypes.back().id() != cudf::type_id::EMPTY, "Unsupported data type");
      } else {
        // If it's a list, assign dtypes to active columns in the given order
        CUDF_EXPECTS(static_cast<int>(opts_.get_dtypes().size()) >= num_actual_cols_,
                     "Must specify data types for all columns");

        auto dtype_ = std::back_inserter(dtypes);

        for (int col = 0; col < num_actual_cols_; col++) {
          if (h_column_flags_[col] & column_parse::enabled) {
            column_parse::flags col_flags_;
            std::tie(dtype_, col_flags_) = get_dtype_info(opts_.get_dtypes()[col]);
            h_column_flags_[col] |= col_flags_;
            CUDF_EXPECTS(dtypes.back().id() != cudf::type_id::EMPTY, "Unsupported data type");
          }
        }
      }
    } else {
      // Translate vector of `name : dtype` strings to map
      // NOTE: Incoming pairs can be out-of-order from column names in dataset
      std::unordered_map<std::string, std::string> col_type_map;
      for (const auto &pair : opts_.get_dtypes()) {
        const auto pos     = pair.find_last_of(':');
        const auto name    = pair.substr(0, pos);
        const auto dtype   = pair.substr(pos + 1, pair.size());
        col_type_map[name] = dtype;
      }

      auto dtype_ = std::back_inserter(dtypes);

      for (int col = 0; col < num_actual_cols_; col++) {
        if (h_column_flags_[col] & column_parse::enabled) {
          CUDF_EXPECTS(col_type_map.find(col_names_[col]) != col_type_map.end(),
                       "Must specify data types for all active columns");
          column_parse::flags col_flags_;
          std::tie(dtype_, col_flags_) = get_dtype_info(col_type_map[col_names_[col]]);
          h_column_flags_[col] |= col_flags_;
          CUDF_EXPECTS(dtypes.back().id() != cudf::type_id::EMPTY, "Unsupported data type");
        }
      }
    }
  }

  if (opts_.get_timestamp_type().id() != cudf::type_id::EMPTY) {
    for (auto &type : dtypes) {
      if (cudf::is_timestamp(type)) { type = opts_.get_timestamp_type(); }
    }
  }

  for (size_t i = 0; i < dtypes.size(); i++) {
    // Replace EMPTY dtype with STRING
    if (dtypes[i].id() == type_id::EMPTY) { dtypes[i] = data_type{type_id::STRING}; }
  }

  return dtypes;
}

std::vector<column_buffer> reader::impl::decode_data(std::vector<data_type> const &column_types,
                                                     cudaStream_t stream)
{
  // Alloc output; columns' data memory is still expected for empty dataframe
  std::vector<column_buffer> out_buffers;

  out_buffers.reserve(column_types.size());

  for (int col = 0, active_col = 0; col < num_actual_cols_; ++col) {
    if (h_column_flags_[col] & column_parse::enabled) {
      const bool is_final_allocation = column_types[active_col].id() != type_id::STRING;
      auto out_buffer =
        column_buffer(column_types[active_col],
                      num_records_,
                      true,
                      stream,
                      is_final_allocation ? mr_ : rmm::mr::get_current_device_resource());

      out_buffer.name = col_names_[col];
      out_buffers.emplace_back(std::move(out_buffer));
      active_col++;
    }
  }

  thrust::host_vector<void *> h_data(num_active_cols_);
  thrust::host_vector<bitmask_type *> h_valid(num_active_cols_);

  for (int i = 0; i < num_active_cols_; ++i) {
    h_data[i]  = out_buffers[i].data();
    h_valid[i] = out_buffers[i].null_mask();
  }

  rmm::device_vector<data_type> d_dtypes(column_types);
  rmm::device_vector<void *> d_data          = h_data;
  rmm::device_vector<bitmask_type *> d_valid = h_valid;
  d_column_flags_                            = h_column_flags_;

  cudf::io::csv::gpu::decode_row_column_data(
    opts, data_, d_column_flags_, row_offsets_, d_dtypes, d_data, d_valid, stream);

  CUDA_TRY(cudaStreamSynchronize(stream));

  for (int i = 0; i < num_active_cols_; ++i) { out_buffers[i].null_count() = UNKNOWN_NULL_COUNT; }

  return out_buffers;
}

reader::impl::impl(std::unique_ptr<datasource> source,
                   std::string filepath,
                   csv_reader_options const &options,
                   rmm::mr::device_memory_resource *mr)
  : mr_(mr), source_(std::move(source)), filepath_(filepath), opts_(options)
{
  num_actual_cols_ = opts_.get_names().size();
  num_active_cols_ = num_actual_cols_;

  if (opts_.is_enabled_delim_whitespace()) {
    opts.delimiter       = ' ';
    opts.multi_delimiter = true;
  } else {
    opts.delimiter       = opts_.get_delimiter();
    opts.multi_delimiter = false;
  }
  opts.terminator = opts_.get_lineterminator();
  if (opts_.get_quotechar() != '\0' && opts_.get_quoting() != quote_style::NONE) {
    opts.quotechar   = opts_.get_quotechar();
    opts.keepquotes  = false;
    opts.doublequote = opts_.is_enabled_doublequote();
  } else {
    opts.quotechar   = '\0';
    opts.keepquotes  = true;
    opts.doublequote = false;
  }
  opts.skipblanklines = opts_.is_enabled_skip_blank_lines();
  opts.comment        = opts_.get_comment();
  opts.dayfirst       = opts_.is_enabled_dayfirst();
  opts.decimal        = opts_.get_decimal();
  opts.thousands      = opts_.get_thousands();
  CUDF_EXPECTS(opts.decimal != opts.delimiter, "Decimal point cannot be the same as the delimiter");
  CUDF_EXPECTS(opts.thousands != opts.delimiter,
               "Thousands separator cannot be the same as the delimiter");

  compression_type_ =
    infer_compression_type(opts_.get_compression(),
                           filepath,
                           {{"gz", "gzip"}, {"zip", "zip"}, {"bz2", "bz2"}, {"xz", "xz"}});

  // Handle user-defined false values, whereby field data is substituted with a
  // boolean true or numeric `1` value
  if (opts_.get_true_values().size() != 0) {
    d_trie_true_        = createSerializedTrie(opts_.get_true_values());
    opts.trueValuesTrie = d_trie_true_.data().get();
  }

  // Handle user-defined false values, whereby field data is substituted with a
  // boolean false or numeric `0` value
  if (opts_.get_false_values().size() != 0) {
    d_trie_false_        = createSerializedTrie(opts_.get_false_values());
    opts.falseValuesTrie = d_trie_false_.data().get();
  }

  // Handle user-defined N/A values, whereby field data is treated as null
  if (opts_.get_na_values().size() != 0) {
    d_trie_na_        = createSerializedTrie(opts_.get_na_values());
    opts.naValuesTrie = d_trie_na_.data().get();
  }
}

// Forward to implementation
reader::reader(std::vector<std::string> const &filepaths,
               csv_reader_options const &options,
               rmm::mr::device_memory_resource *mr)
{
  CUDF_EXPECTS(filepaths.size() == 1, "Only a single source is currently supported.");
  // Delay actual instantiation of data source until read to allow for
  // partial memory mapping of file using byte ranges
  _impl = std::make_unique<impl>(nullptr, filepaths[0], options, mr);
}

// Forward to implementation
reader::reader(std::vector<std::unique_ptr<cudf::io::datasource>> &&sources,
               csv_reader_options const &options,
               rmm::mr::device_memory_resource *mr)
{
  CUDF_EXPECTS(sources.size() == 1, "Only a single source is currently supported.");
  _impl = std::make_unique<impl>(std::move(sources[0]), "", options, mr);
}

// Destructor within this translation unit
reader::~reader() = default;

// Forward to implementation
table_with_metadata reader::read(cudaStream_t stream) { return _impl->read(stream); }

}  // namespace csv
}  // namespace detail
}  // namespace io
}  // namespace cudf
