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
 * @brief cuDF-IO CSV reader class implementation
 **/

#include "reader_impl.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <tuple>
#include <unordered_map>

#include <cudf/strings/replace.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>

#include <io/comp/io_uncomp.h>
#include <io/utilities/parsing_utils.cuh>
#include <io/utilities/type_conversion.cuh>

using std::string;
using std::vector;

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

table_with_metadata reader::impl::read(size_t range_offset,
                                       size_t range_size,
                                       int skip_rows,
                                       int skip_end_rows,
                                       int num_rows,
                                       cudaStream_t stream)
{
  std::vector<std::unique_ptr<column>> out_columns;
  table_metadata metadata;

  if (range_offset > 0 || range_size > 0) {
    CUDF_EXPECTS(compression_type_ == "none",
                 "Reading compressed data using `byte range` is unsupported");
  }
  size_t map_range_size = 0;
  if (range_size != 0) {
    const auto num_columns = std::max(args_.names.size(), args_.dtype.size());
    map_range_size         = range_size + calculateMaxRowSize(num_columns);
  }

  // Support delayed opening of the file if using memory mapping datasource
  // This allows only mapping of a subset of the file if using byte range
  if (source_ == nullptr) {
    assert(!filepath_.empty());
    source_ = datasource::create(filepath_, range_offset, map_range_size);
  }

  // Return an empty dataframe if no data and no column metadata to process
  if (source_->is_empty() && (args_.names.empty() || args_.dtype.empty())) {
    return {std::make_unique<table>(std::move(out_columns)), std::move(metadata)};
  }

  // Transfer source data to GPU
  if (!source_->is_empty()) {
    const char *h_uncomp_data = nullptr;
    size_t h_uncomp_size      = 0;

    auto data_size = (map_range_size != 0) ? map_range_size : source_->size();
    auto buffer    = source_->host_read(range_offset, data_size);

    std::vector<char> h_uncomp_data_owner;
    if (compression_type_ == "none") {
      // Do not use the owner vector here to avoid extra copy
      h_uncomp_data = reinterpret_cast<const char *>(buffer->data());
      h_uncomp_size = buffer->size();
    } else {
      getUncompressedHostData(reinterpret_cast<const char *>(buffer->data()),
                              buffer->size(),
                              compression_type_,
                              h_uncomp_data_owner);
      h_uncomp_data = h_uncomp_data_owner.data();
      h_uncomp_size = h_uncomp_data_owner.size();
    }
    // None of the parameters for row selection is used, we are parsing the entire file
    const bool load_whole_file = range_offset == 0 && range_size == 0 && skip_rows <= 0 &&
                                 skip_end_rows <= 0 && num_rows == -1;

    // With byte range, find the start of the first data row
    size_t const data_start_offset =
      (range_offset != 0) ? find_first_row_start(h_uncomp_data, h_uncomp_size) : 0;

    // TODO: Allow parsing the header outside the mapped range
    CUDF_EXPECTS((range_offset == 0 || args_.header < 0),
                 "byte_range offset with header not supported");

    // Gather row offsets
    gather_row_offsets(h_uncomp_data,
                       h_uncomp_size,
                       data_start_offset,
                       (range_size) ? range_size : h_uncomp_size,
                       (skip_rows > 0) ? skip_rows : 0,
                       num_rows,
                       load_whole_file,
                       stream);

    // Exclude the rows that are to be skipped from the end
    if (skip_end_rows > 0 && static_cast<size_t>(skip_end_rows) < row_offsets.size()) {
      row_offsets.resize(row_offsets.size() - skip_end_rows);
    }

    // Exclude the end-of-data row from number of rows with actual data
    num_records = row_offsets.size();
    num_records -= (num_records > 0);
  } else {
    num_records = 0;
  }

  // Check if the user gave us a list of column names
  if (not args_.names.empty()) {
    h_column_flags.resize(args_.names.size(), column_parse::enabled);
    col_names = args_.names;
  } else {
    col_names = setColumnNames(header, opts, args_.header, args_.prefix);

    num_actual_cols = num_active_cols = col_names.size();

    h_column_flags.resize(num_actual_cols, column_parse::enabled);

    // Rename empty column names to "Unnamed: col_index"
    for (size_t col_idx = 0; col_idx < col_names.size(); ++col_idx) {
      if (col_names[col_idx].empty()) {
        col_names[col_idx] = string("Unnamed: ") + std::to_string(col_idx);
      }
    }

    // Looking for duplicates
    std::unordered_map<string, int> col_names_histogram;
    for (auto &col_name : col_names) {
      // Operator [] inserts a default-initialized value if the given key is not
      // present
      if (++col_names_histogram[col_name] > 1) {
        if (args_.mangle_dupe_cols) {
          // Rename duplicates of column X as X.1, X.2, ...; First appearance
          // stays as X
          col_name += "." + std::to_string(col_names_histogram[col_name] - 1);
        } else {
          // All duplicate columns will be ignored; First appearance is parsed
          const auto idx      = &col_name - col_names.data();
          h_column_flags[idx] = column_parse::disabled;
        }
      }
    }

    // Update the number of columns to be processed, if some might have been
    // removed
    if (!args_.mangle_dupe_cols) { num_active_cols = col_names_histogram.size(); }
  }

  // User can specify which columns should be parsed
  if (!args_.use_cols_indexes.empty() || !args_.use_cols_names.empty()) {
    std::fill(h_column_flags.begin(), h_column_flags.end(), column_parse::disabled);

    for (const auto index : args_.use_cols_indexes) {
      h_column_flags[index] = column_parse::enabled;
    }
    num_active_cols = args_.use_cols_indexes.size();

    for (const auto name : args_.use_cols_names) {
      const auto it = std::find(col_names.begin(), col_names.end(), name);
      if (it != col_names.end()) {
        h_column_flags[it - col_names.begin()] = column_parse::enabled;
        num_active_cols++;
      }
    }
  }

  // User can specify which columns should be inferred as datetime
  if (!args_.infer_date_indexes.empty() || !args_.infer_date_names.empty()) {
    for (const auto index : args_.infer_date_indexes) {
      h_column_flags[index] |= column_parse::as_datetime;
    }

    for (const auto name : args_.infer_date_names) {
      auto it = std::find(col_names.begin(), col_names.end(), name);
      if (it != col_names.end()) {
        h_column_flags[it - col_names.begin()] |= column_parse::as_datetime;
      }
    }
  }

  // Return empty table rather than exception if nothing to load
  if (num_active_cols == 0) {
    return {std::make_unique<table>(std::move(out_columns)), std::move(metadata)};
  }

  std::vector<data_type> column_types = gather_column_types(stream);

  // Alloc output; columns' data memory is still expected for empty dataframe
  std::vector<column_buffer> out_buffers;
  out_buffers.reserve(column_types.size());
  for (int col = 0, active_col = 0; col < num_actual_cols; ++col) {
    if (h_column_flags[col] & column_parse::enabled) {
      // Replace EMPTY dtype with STRING
      if (column_types[active_col].id() == type_id::EMPTY) {
        column_types[active_col] = data_type{STRING};
      }
      out_buffers.emplace_back(
        column_types[active_col],
        num_records,
        true,
        stream,
        column_types[active_col].id() != type_id::STRING ? mr_ : rmm::mr::get_default_resource());
      metadata.column_names.emplace_back(col_names[col]);
      active_col++;
    }
  }

  out_columns.reserve(column_types.size());
  if (num_records != 0) {
    decode_data(column_types, out_buffers, stream);

    for (size_t i = 0; i < column_types.size(); ++i) {
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
        out_columns.emplace_back(
          make_column(column_types[i], num_records, out_buffers[i], stream, mr_));
      }
    }
  } else {
    // Create empty columns
    for (size_t i = 0; i < column_types.size(); ++i) {
      out_columns.emplace_back(make_empty_column(column_types[i]));
    }
  }
  return {std::make_unique<table>(std::move(out_columns)), std::move(metadata)};
}

size_t reader::impl::find_first_row_start(const char *h_data, size_t h_size)
{
  // For now, look for the first terminator (assume the first terminator isn't within a quote)
  // TODO: Attempt to infer this from the data
  size_t pos = 0;
  while (pos < h_size && h_data[pos] != opts.terminator) { ++pos; }
  return std::min(pos + 1, h_size);
}

void reader::impl::gather_row_offsets(const char *h_data,
                                      size_t h_size,
                                      size_t range_begin,
                                      size_t range_end,
                                      size_t skip_rows,
                                      int64_t num_rows,
                                      bool load_whole_file,
                                      cudaStream_t stream)
{
  constexpr size_t max_chunk_bytes = 64 * 1024 * 1024;  // 64MB
  size_t buffer_size               = std::min(max_chunk_bytes, h_size);
  size_t max_blocks =
    std::max<size_t>((buffer_size / cudf::io::csv::gpu::rowofs_block_bytes) + 1, 2);
  hostdevice_vector<uint64_t> row_ctx(max_blocks);
  size_t buffer_pos  = std::min(range_begin - std::min(range_begin, sizeof(char)), h_size);
  size_t pos         = std::min(range_begin, h_size);
  size_t header_rows = (args_.header >= 0) ? args_.header + 1 : 0;
  uint64_t ctx       = 0;

  // For compatibility with the previous parser, a row is considered in-range if the
  // previous row terminator is within the given range
  range_end += (range_end < h_size);
  data_.resize(0);
  row_offsets.resize(0);
  data_.reserve((load_whole_file) ? h_size : std::min(buffer_size * 2, h_size));
  do {
    size_t target_pos = std::min(pos + max_chunk_bytes, h_size);
    size_t chunk_size = target_pos - pos;

    data_.insert(data_.end(), h_data + buffer_pos + data_.size(), h_data + target_pos);

    // Pass 1: Count the potential number of rows in each character block for each
    // possible parser state at the beginning of the block.
    uint32_t num_blocks = cudf::io::csv::gpu::gather_row_offsets(row_ctx.device_ptr(),
                                                                 nullptr,
                                                                 data_.data().get(),
                                                                 chunk_size,
                                                                 pos,
                                                                 buffer_pos,
                                                                 h_size,
                                                                 range_begin,
                                                                 range_end,
                                                                 skip_rows,
                                                                 0,
                                                                 opts,
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
      size_t num_row_offsets = total_rows - skip_rows;
      row_offsets.resize(num_row_offsets);
      CUDA_TRY(cudaMemcpyAsync(row_ctx.device_ptr(),
                               row_ctx.host_ptr(),
                               num_blocks * sizeof(uint64_t),
                               cudaMemcpyHostToDevice,
                               stream));
      // Pass 2: Output row offsets
      cudf::io::csv::gpu::gather_row_offsets(row_ctx.device_ptr(),
                                             row_offsets.data().get(),
                                             data_.data().get(),
                                             chunk_size,
                                             pos,
                                             buffer_pos,
                                             h_size,
                                             range_begin,
                                             range_end,
                                             skip_rows,
                                             num_row_offsets,
                                             opts,
                                             stream);
      // With byte range, we want to keep only one row out of the specified range
      if (range_end < h_size) {
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
          num_row_offsets -= std::min(rows_out_of_range - 1, num_row_offsets);
          row_offsets.resize(num_row_offsets);
          // Implies we reached the end of the range
          break;
        }
      }
      // num_rows does not include blank rows
      if (num_rows >= 0) {
        if (num_row_offsets > header_rows + static_cast<size_t>(num_rows)) {
          size_t num_blanks =
            cudf::io::csv::gpu::count_blank_rows(row_offsets, data_, opts, stream);
          if (num_row_offsets - num_blanks > header_rows + static_cast<size_t>(num_rows)) {
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
  } while (pos < h_size);

  // Eliminate blank rows
  if (row_offsets.size() != 0) {
    cudf::io::csv::gpu::remove_blank_rows(row_offsets, data_, opts, stream);
  }
  // Remove header rows and extract header
  const size_t header_row_index = std::max<size_t>(header_rows, 1) - 1;
  if (header_row_index + 1 < row_offsets.size()) {
    CUDA_TRY(cudaMemcpyAsync(row_ctx.host_ptr(),
                             row_offsets.data().get() + header_row_index,
                             2 * sizeof(uint64_t),
                             cudaMemcpyDeviceToHost,
                             stream));
    CUDA_TRY(cudaStreamSynchronize(stream));
    const auto header_start = buffer_pos + row_ctx[0];
    const auto header_end   = buffer_pos + row_ctx[1];
    CUDF_EXPECTS(header_start <= header_end && header_end <= h_size, "Invalid csv header location");
    header.assign(h_data + header_start, h_data + header_end);
    if (header_rows > 0) {
      row_offsets.erase(row_offsets.begin(), row_offsets.begin() + header_rows);
    }
  }
  // Apply num_rows limit
  if (num_rows >= 0) { row_offsets.resize(std::min<size_t>(row_offsets.size(), num_rows + 1)); }
}

std::vector<data_type> reader::impl::gather_column_types(cudaStream_t stream)
{
  std::vector<data_type> dtypes;

  if (args_.dtype.empty()) {
    if (num_records == 0) {
      dtypes.resize(num_active_cols, data_type{EMPTY});
    } else {
      d_column_flags = h_column_flags;

      hostdevice_vector<column_parse::stats> column_stats(num_active_cols);
      CUDA_TRY(cudaMemsetAsync(column_stats.device_ptr(), 0, column_stats.memory_size(), stream));
      CUDA_TRY(cudf::io::csv::gpu::DetectColumnTypes(data_.data().get(),
                                                     row_offsets.data().get(),
                                                     num_records,
                                                     num_actual_cols,
                                                     opts,
                                                     d_column_flags.data().get(),
                                                     column_stats.device_ptr(),
                                                     stream));
      CUDA_TRY(cudaMemcpyAsync(column_stats.host_ptr(),
                               column_stats.device_ptr(),
                               column_stats.memory_size(),
                               cudaMemcpyDeviceToHost,
                               stream));
      CUDA_TRY(cudaStreamSynchronize(stream));

      for (int col = 0; col < num_active_cols; col++) {
        unsigned long long countInt = column_stats[col].countInt8 + column_stats[col].countInt16 +
                                      column_stats[col].countInt32 + column_stats[col].countInt64;

        if (column_stats[col].countNULL == num_records) {
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
    const bool is_dict = std::all_of(args_.dtype.begin(), args_.dtype.end(), [](const auto &s) {
      return s.find(':') != std::string::npos;
    });

    if (!is_dict) {
      if (args_.dtype.size() == 1) {
        // If it's a single dtype, assign that dtype to all active columns
        data_type dtype_;
        column_parse::flags col_flags_;
        std::tie(dtype_, col_flags_) = get_dtype_info(args_.dtype[0]);
        dtypes.resize(num_active_cols, dtype_);
        for (int col = 0; col < num_actual_cols; col++) { h_column_flags[col] |= col_flags_; }
        CUDF_EXPECTS(dtypes.back().id() != cudf::type_id::EMPTY, "Unsupported data type");
      } else {
        // If it's a list, assign dtypes to active columns in the given order
        CUDF_EXPECTS(static_cast<int>(args_.dtype.size()) >= num_actual_cols,
                     "Must specify data types for all columns");

        auto dtype_ = std::back_inserter(dtypes);

        for (int col = 0; col < num_actual_cols; col++) {
          if (h_column_flags[col] & column_parse::enabled) {
            column_parse::flags col_flags_;
            std::tie(dtype_, col_flags_) = get_dtype_info(args_.dtype[col]);
            h_column_flags[col] |= col_flags_;
            CUDF_EXPECTS(dtypes.back().id() != cudf::type_id::EMPTY, "Unsupported data type");
          }
        }
      }
    } else {
      // Translate vector of `name : dtype` strings to map
      // NOTE: Incoming pairs can be out-of-order from column names in dataset
      std::unordered_map<std::string, std::string> col_type_map;
      for (const auto &pair : args_.dtype) {
        const auto pos     = pair.find_last_of(':');
        const auto name    = pair.substr(0, pos);
        const auto dtype   = pair.substr(pos + 1, pair.size());
        col_type_map[name] = dtype;
      }

      auto dtype_ = std::back_inserter(dtypes);

      for (int col = 0; col < num_actual_cols; col++) {
        if (h_column_flags[col] & column_parse::enabled) {
          CUDF_EXPECTS(col_type_map.find(col_names[col]) != col_type_map.end(),
                       "Must specify data types for all active columns");
          column_parse::flags col_flags_;
          std::tie(dtype_, col_flags_) = get_dtype_info(col_type_map[col_names[col]]);
          h_column_flags[col] |= col_flags_;
          CUDF_EXPECTS(dtypes.back().id() != cudf::type_id::EMPTY, "Unsupported data type");
        }
      }
    }
  }

  if (args_.timestamp_type.id() != cudf::type_id::EMPTY) {
    for (auto &type : dtypes) {
      if (cudf::is_timestamp(type)) { type = args_.timestamp_type; }
    }
  }

  return dtypes;
}

void reader::impl::decode_data(const std::vector<data_type> &column_types,
                               std::vector<column_buffer> &out_buffers,
                               cudaStream_t stream)
{
  thrust::host_vector<void *> h_data(num_active_cols);
  thrust::host_vector<bitmask_type *> h_valid(num_active_cols);

  for (int i = 0; i < num_active_cols; ++i) {
    h_data[i]  = out_buffers[i].data();
    h_valid[i] = out_buffers[i].null_mask();
  }

  rmm::device_vector<data_type> d_dtypes(column_types);
  rmm::device_vector<void *> d_data          = h_data;
  rmm::device_vector<bitmask_type *> d_valid = h_valid;
  d_column_flags                             = h_column_flags;

  CUDA_TRY(cudf::io::csv::gpu::DecodeRowColumnData(data_.data().get(),
                                                   row_offsets.data().get(),
                                                   num_records,
                                                   num_actual_cols,
                                                   opts,
                                                   d_column_flags.data().get(),
                                                   d_dtypes.data().get(),
                                                   d_data.data().get(),
                                                   d_valid.data().get(),
                                                   stream));
  CUDA_TRY(cudaStreamSynchronize(stream));

  for (int i = 0; i < num_active_cols; ++i) { out_buffers[i].null_count() = UNKNOWN_NULL_COUNT; }
}

reader::impl::impl(std::unique_ptr<datasource> source,
                   std::string filepath,
                   reader_options const &options,
                   rmm::mr::device_memory_resource *mr)
  : source_(std::move(source)), mr_(mr), filepath_(filepath), args_(options)
{
  num_actual_cols = args_.names.size();
  num_active_cols = args_.names.size();

  if (args_.delim_whitespace) {
    opts.delimiter       = ' ';
    opts.multi_delimiter = true;
  } else {
    opts.delimiter       = args_.delimiter;
    opts.multi_delimiter = false;
  }
  opts.terminator = args_.lineterminator;
  if (args_.quotechar != '\0' && args_.quoting != quote_style::NONE) {
    opts.quotechar   = args_.quotechar;
    opts.keepquotes  = false;
    opts.doublequote = args_.doublequote;
  } else {
    opts.quotechar   = '\0';
    opts.keepquotes  = true;
    opts.doublequote = false;
  }
  opts.skipblanklines = args_.skip_blank_lines;
  opts.comment        = args_.comment;
  opts.dayfirst       = args_.dayfirst;
  opts.decimal        = args_.decimal;
  opts.thousands      = args_.thousands;
  CUDF_EXPECTS(opts.decimal != opts.delimiter, "Decimal point cannot be the same as the delimiter");
  CUDF_EXPECTS(opts.thousands != opts.delimiter,
               "Thousands separator cannot be the same as the delimiter");

  compression_type_ = infer_compression_type(
    args_.compression, filepath, {{"gz", "gzip"}, {"zip", "zip"}, {"bz2", "bz2"}, {"xz", "xz"}});

  // Handle user-defined false values, whereby field data is substituted with a
  // boolean true or numeric `1` value
  if (args_.true_values.size() != 0) {
    d_trueTrie          = createSerializedTrie(args_.true_values);
    opts.trueValuesTrie = d_trueTrie.data().get();
  }

  // Handle user-defined false values, whereby field data is substituted with a
  // boolean false or numeric `0` value
  if (args_.false_values.size() != 0) {
    d_falseTrie          = createSerializedTrie(args_.false_values);
    opts.falseValuesTrie = d_falseTrie.data().get();
  }

  // Handle user-defined N/A values, whereby field data is treated as null
  if (args_.na_values.size() != 0) {
    d_naTrie          = createSerializedTrie(args_.na_values);
    opts.naValuesTrie = d_naTrie.data().get();
  }
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
  return _impl->read(0, 0, 0, 0, -1, stream);
}

// Forward to implementation
table_with_metadata reader::read_byte_range(size_t offset, size_t size, cudaStream_t stream)
{
  return _impl->read(offset, size, 0, 0, -1, stream);
}

// Forward to implementation
table_with_metadata reader::read_rows(size_type num_skip_header,
                                      size_type num_skip_footer,
                                      size_type num_rows,
                                      cudaStream_t stream)
{
  CUDF_EXPECTS(num_rows == -1 || num_skip_footer == 0,
               "Cannot use both `num_rows` and `num_skip_footer`");

  return _impl->read(0, 0, num_skip_header, num_skip_footer, num_rows, stream);
}

}  // namespace csv
}  // namespace detail
}  // namespace io
}  // namespace cudf
