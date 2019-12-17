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

#include "legacy/datetime_parser.cuh"
#include "legacy/type_conversion.cuh"

#include <utilities/legacy/cudf_utils.h>
#include <cudf/legacy/unary.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>

#include <nvstrings/NVStrings.h>

#include <io/comp/io_uncomp.h>
#include <io/utilities/legacy/parsing_utils.cuh>

using std::string;
using std::vector;

namespace cudf {
namespace experimental {
namespace io {
namespace detail {
namespace csv {

using namespace cudf::io::csv;
using namespace cudf::io;

/**---------------------------------------------------------------------------*
 * @brief Estimates the maximum expected length or a row, based on the number
 * of columns
 *
 * If the number of columns is not available, it will return a value large
 * enough for most use cases
 *
 * @param[in] num_columns Number of columns in the CSV file (optional)
 *
 * @return Estimated maximum size of a row, in bytes
 *---------------------------------------------------------------------------**/
constexpr size_t calculateMaxRowSize(int num_columns = 0) noexcept {
  constexpr size_t max_row_bytes = 16 * 1024;  // 16KB
  constexpr size_t column_bytes = 64;
  constexpr size_t base_padding = 1024;  // 1KB
  if (num_columns == 0) {
    // Use flat size if the number of columns is not known
    return max_row_bytes;
  } else {
    // Expand the size based on the number of columns, if available
    return base_padding + num_columns * column_bytes;
  }
}

namespace {

std::string infer_compression_type(
    const compression_type &compression_arg, const std::string &filename,
    const std::vector<std::pair<std::string, std::string>> &ext_to_comp_map) {
  auto str_tolower = [](const auto &begin, const auto &end) {
    std::string out;
    std::transform(begin, end, std::back_inserter(out), ::tolower);
    return out;
  };

  // Attempt to infer from user-supplied argument
  if (compression_arg != compression_type::AUTO) {
    switch (compression_arg) {
      case compression_type::GZIP:
        return "gzip";
      case compression_type::BZIP2:
        return "bz2";
      case compression_type::ZIP:
        return "zip";
      case compression_type::XZ:
        return "xz";
      default:
        break;
    }
  }

  // Attempt to infer from the file extension
  const auto pos = filename.find_last_of('.');
  if (pos != std::string::npos) {
    const auto ext = str_tolower(filename.begin() + pos + 1, filename.end());
    for (const auto &mapping : ext_to_comp_map) {
      if (mapping.first == ext) {
        return mapping.second;
      }
    }
  }

  return "none";
}

data_type convertStringToDtype(const std::string &dtype) {
  if (dtype == "str") return data_type(cudf::type_id::STRING);
  if (dtype == "timestamp[s]")
    return data_type(cudf::type_id::TIMESTAMP_SECONDS);
  // backwards compat: "timestamp" defaults to milliseconds
  if (dtype == "timestamp[ms]" || dtype == "timestamp")
    return data_type(cudf::type_id::TIMESTAMP_MILLISECONDS);
  if (dtype == "timestamp[us]")
    return data_type(cudf::type_id::TIMESTAMP_MICROSECONDS);
  if (dtype == "timestamp[ns]")
    return data_type(cudf::type_id::TIMESTAMP_NANOSECONDS);
  if (dtype == "category") return data_type(cudf::type_id::CATEGORY);
  if (dtype == "date32") return data_type(cudf::type_id::TIMESTAMP_DAYS);
  if (dtype == "bool" || dtype == "boolean")
    return data_type(cudf::type_id::BOOL8);
  if (dtype == "date" || dtype == "date64")
    return data_type(cudf::type_id::TIMESTAMP_MILLISECONDS);
  if (dtype == "float" || dtype == "float32")
    return data_type(cudf::type_id::FLOAT32);
  if (dtype == "double" || dtype == "float64")
    return data_type(cudf::type_id::FLOAT64);
  if (dtype == "byte" || dtype == "int8") return data_type(cudf::type_id::INT8);
  if (dtype == "short" || dtype == "int16")
    return data_type(cudf::type_id::INT16);
  if (dtype == "int" || dtype == "int32")
    return data_type(cudf::type_id::INT32);
  if (dtype == "long" || dtype == "int64")
    return data_type(cudf::type_id::INT64);

  return data_type(cudf::type_id::EMPTY);
}

}  // namespace

/**
 * @brief Translates a dtype string and returns its dtype enumeration and any
 * extended dtype flags that are supported by cuIO. Often, this is a column
 * with the same underlying dtype the basic types, but with different parsing
 * interpretations.
 *
 * @param[in] dtype String containing the basic or extended dtype
 *
 * @return std::pair<gdf_dtype, column_parse::flags> Tuple of dtype and flags
 */
std::tuple<data_type, column_parse::flags> get_dtype_info(
    const std::string &dtype) {
  if (dtype == "hex" || dtype == "hex64") {
    return std::make_tuple(data_type{cudf::type_id::INT64},
                           column_parse::as_hexadecimal);
  }
  if (dtype == "hex32") {
    return std::make_tuple(data_type{cudf::type_id::INT32},
                           column_parse::as_hexadecimal);
  }

  return std::make_tuple(convertStringToDtype(dtype), column_parse::as_default);
}

/**
 * @brief Removes the first and Last quote in the string
 */
string removeQuotes(string str, char quotechar) {
  // Exclude first and last quotation char
  const size_t first_quote = str.find(quotechar);
  if (first_quote != string::npos) {
    str.erase(first_quote, 1);
  }
  const size_t last_quote = str.rfind(quotechar);
  if (last_quote != string::npos) {
    str.erase(last_quote, 1);
  }

  return str;
}

/**
 * @brief Parse the first row to set the column names in the raw_csv parameter.
 * The first row can be either the header row, or the first data row
 */
std::vector<std::string> setColumnNames(std::vector<char> const &header,
                                        ParseOptions const &opts,
                                        int header_row, std::string prefix) {
  std::vector<std::string> col_names;

  // If there is only a single character then it would be the terminator
  if (header.size() <= 1) {
    return col_names;
  }

  std::vector<char> first_row = header;
  int num_cols = 0;

  bool quotation = false;
  for (size_t pos = 0, prev = 0; pos < first_row.size(); ++pos) {
    // Flip the quotation flag if current character is a quotechar
    if (first_row[pos] == opts.quotechar) {
      quotation = !quotation;
    }
    // Check if end of a column/row
    else if (pos == first_row.size() - 1 ||
             (!quotation && first_row[pos] == opts.terminator) ||
             (!quotation && first_row[pos] == opts.delimiter)) {
      // This is the header, add the column name
      if (header_row >= 0) {
        // Include the current character, in case the line is not terminated
        int col_name_len = pos - prev + 1;
        // Exclude the delimiter/terminator is present
        if (first_row[pos] == opts.delimiter ||
            first_row[pos] == opts.terminator) {
          --col_name_len;
        }
        // Also exclude '\r' character at the end of the column name if it's
        // part of the terminator
        if (col_name_len > 0 && opts.terminator == '\n' &&
            first_row[pos] == '\n' && first_row[pos - 1] == '\r') {
          --col_name_len;
        }

        const string new_col_name(first_row.data() + prev, col_name_len);
        col_names.push_back(removeQuotes(new_col_name, opts.quotechar));

        // Stop parsing when we hit the line terminator; relevant when there is
        // a blank line following the header. In this case, first_row includes
        // multiple line terminators at the end, as the new recStart belongs to
        // a line that comes after the blank line(s)
        if (!quotation && first_row[pos] == opts.terminator) {
          break;
        }
      } else {
        // This is the first data row, add the automatically generated name
        col_names.push_back(prefix + std::to_string(num_cols));
      }
      num_cols++;

      // Skip adjacent delimiters if delim_whitespace is set
      while (opts.multi_delimiter && pos < first_row.size() &&
             first_row[pos] == opts.delimiter &&
             first_row[pos + 1] == opts.delimiter) {
        ++pos;
      }
      prev = pos + 1;
    }
  }

  return col_names;
}

table_with_metadata reader::impl::read(size_t range_offset,
                                       size_t range_size, int skip_rows,
                                       int skip_end_rows, int num_rows,
                                       cudaStream_t stream) {
  std::vector<std::unique_ptr<column>> out_columns;
  table_metadata metadata;

  if (range_offset > 0 || range_size > 0) {
    CUDF_EXPECTS(compression_type_ == "none",
                 "Reading compressed data using `byte range` is unsupported");
  }
  size_t map_range_size = 0;
  if (range_size != 0) {
    const auto num_columns = std::max(args_.names.size(), args_.dtype.size());
    map_range_size = range_size + calculateMaxRowSize(num_columns);
  }

  // Support delayed opening of the file if using memory mapping datasource
  // This allows only mapping of a subset of the file if using byte range
  if (source_ == nullptr) {
    assert(!filepath_.empty());
    source_ = datasource::create(filepath_, range_offset, map_range_size);
  }

  // Return an empty dataframe if no data and no column metadata to process
  if (source_->empty() && (args_.names.empty() || args_.dtype.empty())) {
    return { std::make_unique<table>(std::move(out_columns)), std::move(metadata) };
  }

  // Transfer source data to GPU
  if (!source_->empty()) {
    const char *h_uncomp_data = nullptr;
    size_t h_uncomp_size = 0;

    auto data_size = (map_range_size != 0) ? map_range_size : source_->size();
    auto buffer = source_->get_buffer(range_offset, data_size);

    std::vector<char> h_uncomp_data_owner;
    if (compression_type_ == "none") {
      // Do not use the owner vector here to avoid extra copy
      h_uncomp_data = reinterpret_cast<const char *>(buffer->data());
      h_uncomp_size = buffer->size();
    } else {
      CUDF_EXPECTS(
          getUncompressedHostData(
              reinterpret_cast<const char *>(buffer->data()), buffer->size(),
              compression_type_, h_uncomp_data_owner) == GDF_SUCCESS,
          "Cannot decompress data");
      h_uncomp_data = h_uncomp_data_owner.data();
      h_uncomp_size = h_uncomp_data_owner.size();
    }

    gather_row_offsets(h_uncomp_data, h_uncomp_size, range_offset, stream);
    auto row_range = select_rows(h_uncomp_data, h_uncomp_size, range_size,
                                 skip_rows, skip_end_rows, num_rows, stream);

    data_size = row_range.second - row_range.first;
    CUDF_EXPECTS(data_size <= h_uncomp_size, "Row range exceeds data size");

    num_bits = (data_size + 63) / 64;
    data_ = rmm::device_buffer(h_uncomp_data + row_range.first, data_size);
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
          const auto idx = &col_name - col_names.data();
          h_column_flags[idx] = column_parse::disabled;
        }
      }
    }

    // Update the number of columns to be processed, if some might have been
    // removed
    if (!args_.mangle_dupe_cols) {
      num_active_cols = col_names_histogram.size();
    }
  }

  // User can specify which columns should be parsed
  if (!args_.use_cols_indexes.empty() || !args_.use_cols_names.empty()) {
    std::fill(h_column_flags.begin(), h_column_flags.end(),
              column_parse::disabled);

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
    return { std::make_unique<table>(std::move(out_columns)), std::move(metadata) };
  }

  std::vector<data_type> column_types = gather_column_types(stream);

  // Alloc output; columns' data memory is still expected for empty dataframe
  std::vector<column_buffer> out_buffers;
  for (int col = 0, active_col = 0; col < num_actual_cols; ++col) {
    if (h_column_flags[col] & column_parse::enabled) {
      out_buffers.emplace_back(column_types[active_col], num_records, stream,
                               mr_);
      metadata.column_names.emplace_back(col_names[col]);
      active_col++;
    }
  }

  if (num_records != 0) {
    decode_data(column_types, out_buffers, stream);
  }

  for (size_t i = 0; i < column_types.size(); ++i) {
    out_columns.emplace_back(
        make_column(column_types[i], num_records, out_buffers[i]));
  }

  // TODO: String columns need to be reworked to actually copy characters in
  // kernel to allow skipping quotation characters
  /*for (auto &column : columns) {
    column.finalize();

    // PANDAS' default behavior of enabling doublequote for two consecutive
    // quotechars in quoted fields results in reduction to a single quotechar
    if (column->dtype == GDF_STRING &&
        (opts.quotechar != '\0' && opts.doublequote == true)) {
      const std::string quotechar(1, opts.quotechar);
      const std::string dblquotechar(2, opts.quotechar);
      auto str_data = static_cast<NVStrings *>(column->data);
      column->data = str_data->replace(dblquotechar.c_str(), quotechar.c_str());
      NVStrings::destroy(str_data);
    }
  }*/

  return { std::make_unique<table>(std::move(out_columns)), std::move(metadata) };
}

void reader::impl::gather_row_offsets(const char *h_data, size_t h_size,
                                      size_t range_offset,
                                      cudaStream_t stream) {
  // Account for the start and end of row region offsets
  const bool require_first_line_start = (range_offset == 0);
  const bool require_last_line_end = (h_data[h_size - 1] != opts.terminator);

  auto symbols = (opts.quotechar != '\0')
                     ? std::vector<char>{opts.terminator, opts.quotechar}
                     : std::vector<char>{opts.terminator};
  const auto num_rows = countAllFromSet(h_data, h_size, symbols) +
                        (require_first_line_start ? 1 : 0);
  const auto num_offsets = num_rows + (require_last_line_end ? 1 : 0);
  row_offsets.resize(num_offsets);

  auto ptr_first = row_offsets.data().get();
  auto ptr_last = ptr_first + num_rows;
  if (require_first_line_start) {
    ptr_first++;
    const uint64_t first_entry = 0;
    row_offsets.front() = first_entry;
  }
  if (require_last_line_end) {
    const uint64_t last_entry = h_size;
    row_offsets.back() = last_entry;
  }

  // Passing offset = 1 to return positions AFTER the found character
  findAllFromSet(h_data, h_size, symbols, 1, ptr_first);

  // Sort the row info according to ascending start offset
  // Subsequent processing (filtering, etc.) may require row order
  thrust::sort(rmm::exec_policy(stream)->on(stream), ptr_first, ptr_last);
}

std::pair<uint64_t, uint64_t> reader::impl::select_rows(
    const char *h_data, size_t h_size, size_t range_size,
    cudf::size_type skip_rows, cudf::size_type skip_end_rows,
    cudf::size_type num_rows, cudaStream_t stream) {
  thrust::host_vector<uint64_t> h_row_offsets = row_offsets;
  auto it_begin = h_row_offsets.begin();
  auto it_end = h_row_offsets.end();
  assert(std::distance(it_begin, it_end) >= 1);

  // Currently, ignoring lineterminations within quotes is handled by recording
  // the records of both, and then filtering out the records that is a quotechar
  // or a linetermination within a quotechar pair.
  if (opts.quotechar != '\0') {
    auto count = std::distance(it_begin, it_end) - 1;

    auto filtered_count = count;
    bool quotation = false;
    for (int i = 1; i < count; ++i) {
      if (h_data[h_row_offsets[i] - 1] == opts.quotechar) {
        quotation = !quotation;
        h_row_offsets[i] = static_cast<uint64_t>(-1);
        filtered_count--;
      } else if (quotation) {
        h_row_offsets[i] = static_cast<uint64_t>(-1);
        filtered_count--;
      }
    }
    if (filtered_count != count) {
      it_end = std::remove_if(it_begin, it_end, [](uint64_t pos) {
        return (pos == static_cast<uint64_t>(-1));
      });
    }
  }

  // Exclude the rows that are to be skipped from the start
  if (skip_rows != 0 && skip_rows < std::distance(it_begin, it_end)) {
    it_begin += skip_rows;
  }

  // Exclude the rows outside of requested range
  if (range_size != 0) {
    auto it = it_end - 1;
    while (it >= it_begin && *it > static_cast<uint64_t>(range_size)) {
      --it;
    }
    if ((it + 2) < it_end) {
      it_end = it + 2;
    }
  }

  // Exclude the rows without data
  if (opts.skipblanklines || opts.comment != '\0') {
    const auto newline = opts.skipblanklines ? opts.terminator : opts.comment;
    const auto comment = opts.comment != '\0' ? opts.comment : newline;
    const auto carriage =
        (opts.skipblanklines && opts.terminator == '\n') ? '\r' : comment;

    it_end = std::remove_if(it_begin, it_end, [=, &h_data](uint64_t pos) {
      return ((pos != h_size) &&
              (h_data[pos] == newline || h_data[pos] == comment ||
               h_data[pos] == carriage));
    });
  }

  // Exclude the rows before the header row (inclusive)
  if (std::distance(it_begin, it_end) > 1) {
    if (args_.header == -1) {
      header.assign(h_data + *(it_begin), h_data + *(it_begin + 1));
    } else {
      header.assign(h_data + *(it_begin + args_.header),
                    h_data + *(it_begin + args_.header + 1));
      it_begin += args_.header + 1;
    }
  }

  // Exclude the rows that exceed past the requested number
  if (num_rows >= 0 && num_rows < std::distance(it_begin, it_end)) {
    it_end = it_begin + num_rows + 1;
  }

  // Exclude the rows that are to be skipped from the end
  if (skip_end_rows != 0 && skip_end_rows < std::distance(it_begin, it_end)) {
    it_end -= skip_end_rows;
  }

  const uint64_t offset_start = *it_begin;
  const uint64_t offset_end = *(it_end - 1);

  // Copy out the row starts to use for row-column data parsing
  if (offset_start != offset_end) {
    if (offset_start != 0) {
      for (auto it = it_begin; it != it_end; ++it) {
        *it -= offset_start;
      }
    }
    CUDA_TRY(cudaMemcpyAsync(row_offsets.data().get(), &(*it_begin),
                             std::distance(it_begin, it_end) * sizeof(uint64_t),
                             cudaMemcpyHostToDevice, stream));

    // Exclude the end-of-data row from number of rows with actual data
    num_records = std::distance(it_begin, it_end) - 1;
  }

  return std::make_pair(offset_start, offset_end);
}

std::vector<data_type> reader::impl::gather_column_types(cudaStream_t stream) {
  std::vector<data_type> dtypes;

  if (args_.dtype.empty()) {
    if (num_records == 0) {
      dtypes.resize(num_active_cols, data_type{EMPTY});
    } else {
      d_column_flags = h_column_flags;

      hostdevice_vector<column_parse::stats> column_stats(num_active_cols);
      CUDA_TRY(cudaMemsetAsync(column_stats.device_ptr(), 0,
                               column_stats.memory_size(), stream));
      CUDA_TRY(gpu::DetectColumnTypes(
          static_cast<const char *>(data_.data()), row_offsets.data().get(),
          num_records, num_actual_cols, opts, d_column_flags.data().get(),
          column_stats.device_ptr(), stream));
      CUDA_TRY(cudaMemcpyAsync(
          column_stats.host_ptr(), column_stats.device_ptr(),
          column_stats.memory_size(), cudaMemcpyDeviceToHost, stream));
      CUDA_TRY(cudaStreamSynchronize(stream));

      for (int col = 0; col < num_active_cols; col++) {
        unsigned long long countInt =
            column_stats[col].countInt8 + column_stats[col].countInt16 +
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
    const bool is_dict = std::all_of(
        args_.dtype.begin(), args_.dtype.end(),
        [](const auto &s) { return s.find(':') != std::string::npos; });

    if (!is_dict) {
      if (args_.dtype.size() == 1) {
        // If it's a single dtype, assign that dtype to all active columns
        data_type dtype_;
        column_parse::flags col_flags_;
        std::tie(dtype_, col_flags_) = get_dtype_info(args_.dtype[0]);
        dtypes.resize(num_active_cols, dtype_);
        for (int col = 0; col < num_actual_cols; col++) {
          h_column_flags[col] |= col_flags_;
        }
        CUDF_EXPECTS(dtypes.back().id() != cudf::type_id::EMPTY,
                     "Unsupported data type");
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
            CUDF_EXPECTS(dtypes.back().id() != cudf::type_id::EMPTY,
                         "Unsupported data type");
          }
        }
      }
    } else {
      // Translate vector of `name : dtype` strings to map
      // NOTE: Incoming pairs can be out-of-order from column names in dataset
      std::unordered_map<std::string, std::string> col_type_map;
      for (const auto &pair : args_.dtype) {
        const auto pos = pair.find_last_of(':');
        const auto name = pair.substr(0, pos);
        const auto dtype = pair.substr(pos + 1, pair.size());
        col_type_map[name] = dtype;
      }

      auto dtype_ = std::back_inserter(dtypes);

      for (int col = 0; col < num_actual_cols; col++) {
        if (h_column_flags[col] & column_parse::enabled) {
          CUDF_EXPECTS(col_type_map.find(col_names[col]) != col_type_map.end(),
                       "Must specify data types for all active columns");
          column_parse::flags col_flags_;
          std::tie(dtype_, col_flags_) =
              get_dtype_info(col_type_map[col_names[col]]);
          h_column_flags[col] |= col_flags_;
          CUDF_EXPECTS(dtypes.back().id() != cudf::type_id::EMPTY,
                       "Unsupported data type");
        }
      }
    }
  }

  if (args_.timestamp_type.id() != cudf::type_id::EMPTY) {
    for (auto &type : dtypes) {
      if (cudf::is_timestamp(type)) {
        type = args_.timestamp_type;
      }
    }
  }

  return dtypes;
}

void reader::impl::decode_data(const std::vector<data_type> &column_types,
                               std::vector<column_buffer> &out_buffers,
                               cudaStream_t stream) {
  thrust::host_vector<void *> h_data(num_active_cols);
  thrust::host_vector<bitmask_type *> h_valid(num_active_cols);

  for (int i = 0; i < num_active_cols; ++i) {
    h_data[i] = out_buffers[i].data();
    h_valid[i] = out_buffers[i].null_mask();
  }

  rmm::device_vector<data_type> d_dtypes(column_types);
  rmm::device_vector<void *> d_data = h_data;
  rmm::device_vector<bitmask_type *> d_valid = h_valid;
  d_column_flags = h_column_flags;

  CUDA_TRY(gpu::DecodeRowColumnData(
      static_cast<const char *>(data_.data()), row_offsets.data().get(),
      num_records, num_actual_cols, opts, d_column_flags.data().get(),
      d_dtypes.data().get(), d_data.data().get(), d_valid.data().get(),
      stream));
  CUDA_TRY(cudaStreamSynchronize(stream));

  for (int i = 0; i < num_active_cols; ++i) {
    out_buffers[i].null_count() = UNKNOWN_NULL_COUNT;
  }
}

reader::impl::impl(std::unique_ptr<datasource> source, std::string filepath,
                   reader_options const &options,
                   rmm::mr::device_memory_resource *mr)
    : source_(std::move(source)), mr_(mr), filepath_(filepath), args_(options) {
  num_actual_cols = args_.names.size();
  num_active_cols = args_.names.size();

  if (args_.delim_whitespace) {
    opts.delimiter = ' ';
    opts.multi_delimiter = true;
  } else {
    opts.delimiter = args_.delimiter;
    opts.multi_delimiter = false;
  }
  opts.terminator = args_.lineterminator;
  if (args_.quotechar != '\0' && args_.quoting != quote_style::NONE) {
    opts.quotechar = args_.quotechar;
    opts.keepquotes = false;
    opts.doublequote = args_.doublequote;
  } else {
    opts.quotechar = '\0';
    opts.keepquotes = true;
    opts.doublequote = false;
  }
  opts.skipblanklines = args_.skip_blank_lines;
  opts.comment = args_.comment;
  opts.dayfirst = args_.dayfirst;
  opts.decimal = args_.decimal;
  opts.thousands = args_.thousands;
  CUDF_EXPECTS(opts.decimal != opts.delimiter,
               "Decimal point cannot be the same as the delimiter");
  CUDF_EXPECTS(opts.thousands != opts.delimiter,
               "Thousands separator cannot be the same as the delimiter");

  compression_type_ = infer_compression_type(
      args_.compression, filepath,
      {{"gz", "gzip"}, {"zip", "zip"}, {"bz2", "bz2"}, {"xz", "xz"}});

  // Handle user-defined false values, whereby field data is substituted with a
  // boolean true or numeric `1` value
  if (args_.true_values.size() != 0) {
    d_trueTrie = createSerializedTrie(args_.true_values);
    opts.trueValuesTrie = d_trueTrie.data().get();
  }

  // Handle user-defined false values, whereby field data is substituted with a
  // boolean false or numeric `0` value
  if (args_.false_values.size() != 0) {
    d_falseTrie = createSerializedTrie(args_.false_values);
    opts.falseValuesTrie = d_falseTrie.data().get();
  }

  // Handle user-defined N/A values, whereby field data is treated as null
  if (args_.na_values.size() != 0) {
    d_naTrie = createSerializedTrie(args_.na_values);
    opts.naValuesTrie = d_naTrie.data().get();
  }
}

// Forward to implementation
reader::reader(std::string filepath, reader_options const &options,
               rmm::mr::device_memory_resource *mr)
    : _impl(std::make_unique<impl>(nullptr, filepath, options, mr)) {
  // Delay actual instantiation of data source until read to allow for
  // partial memory mapping of file using byte ranges
}

// Forward to implementation
reader::reader(const char *buffer, size_t length, reader_options const &options,
               rmm::mr::device_memory_resource *mr)
    : _impl(std::make_unique<impl>(datasource::create(buffer, length), "",
                                   options, mr)) {}

// Forward to implementation
reader::reader(std::shared_ptr<arrow::io::RandomAccessFile> file,
               reader_options const &options,
               rmm::mr::device_memory_resource *mr)
    : _impl(std::make_unique<impl>(datasource::create(file), "", options, mr)) {
}

// Destructor within this translation unit
reader::~reader() = default;

// Forward to implementation
table_with_metadata reader::read_all(cudaStream_t stream) {
  return _impl->read(0, 0, 0, 0, -1, stream);
}

// Forward to implementation
table_with_metadata reader::read_byte_range(size_t offset, size_t size,
                                            cudaStream_t stream) {
  return _impl->read(offset, size, 0, 0, -1, stream);
}

// Forward to implementation
table_with_metadata reader::read_rows(size_type num_skip_header,
                                      size_type num_skip_footer,
                                      size_type num_rows,
                                      cudaStream_t stream) {
  CUDF_EXPECTS(num_rows == -1 || num_skip_footer == 0,
               "Cannot use both `num_rows` and `num_skip_footer`");

  return _impl->read(0, 0, num_skip_header, num_skip_footer, num_rows, stream);
}

}  // namespace csv
}  // namespace detail
}  // namespace io
}  // namespace experimental
}  // namespace cudf
