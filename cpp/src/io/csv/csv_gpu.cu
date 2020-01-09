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

#include "csv_common.h"
#include "csv_gpu.h"

#include "datetime.cuh"

#include <cudf/null_mask.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/detail/utilities/trie.cuh>

#include <cuda_runtime.h>

namespace cudf {
namespace io {
namespace csv {
namespace gpu {

/**
 * @brief CUDA kernel iterates over the data until the end of the current field
 *
 * Also iterates over (one or more) delimiter characters after the field.
 * Function applies to formats with field delimiters and line terminators.
 *
 * @param data The entire plain text data to read
 * @param opts A set of parsing options
 * @param pos Offset to start the seeking from
 * @param stop Offset of the end of the row
 *
 * @return long The position of the last character in the field, including the
 *  delimiter(s) following the field data
 */
__device__ __inline__ long seek_field_end(const char *data,
                                          ParseOptions const &opts, long pos,
                                          long stop) {
  bool quotation = false;
  while (true) {
    // Use simple logic to ignore control chars between any quote seq
    // Handles nominal cases including doublequotes within quotes, but
    // may not output exact failures as PANDAS for malformed fields
    if (data[pos] == opts.quotechar) {
      quotation = !quotation;
    } else if (quotation == false) {
      if (data[pos] == opts.delimiter) {
        while (opts.multi_delimiter && pos < stop &&
               data[pos + 1] == opts.delimiter) {
          ++pos;
        }
        break;
      } else if (data[pos] == opts.terminator) {
        break;
      } else if (data[pos] == '\r' &&
                 (pos + 1 < stop && data[pos + 1] == '\n')) {
        stop--;
        break;
      }
    }
    if (pos >= stop) break;
    pos++;
  }
  return pos;
}

/**
 * @brief Checks whether the given character is a whitespace character.
 *
 * @param ch The character to check
 *
 * @return True if the input is whitespace, False otherwise
 */
__device__ __inline__ bool is_whitespace(char c) {
  return c == '\t' || c == ' ';
}

/**
 * @brief Scans a character stream within a range, and adjusts the start and end
 * indices of the range to ignore whitespace and quotation characters.
 *
 * @param data The character stream to scan
 * @param start The start index to adjust
 * @param end The end index to adjust
 * @param quotechar The character used to denote quotes
 *
 * @return Adjusted or unchanged start_idx and end_idx
 */
__device__ __inline__ void trim_field_start_end(const char *data, long *start,
                                                long *end,
                                                char quotechar = '\0') {
  while ((*start < *end) && is_whitespace(data[*start])) {
    (*start)++;
  }
  if ((*start < *end) && data[*start] == quotechar) {
    (*start)++;
  }
  while ((*start <= *end) && is_whitespace(data[*end])) {
    (*end)--;
  }
  if ((*start <= *end) && data[*end] == quotechar) {
    (*end)--;
  }
}

/**
 * @brief Returns true is the input character is a valid digit.
 * Supports both decimal and hexadecimal digits (uppercase and lowercase).
 *
 * @param c Chracter to check
 * @param is_hex Whether to check as a hexadecimal
 *
 * @return `true` if it is digit-like, `false` otherwise
 */
__device__ __inline__ bool is_digit(char c, bool is_hex = false) {
  if (c >= '0' && c <= '9') return true;

  if (is_hex) {
    if (c >= 'A' && c <= 'F') return true;
    if (c >= 'a' && c <= 'f') return true;
  }

  return false;
}

/**
 * @brief Checks whether the given character counters indicate a potentially
 * valid date and/or time field.
 *
 * For performance and simplicity, we detect only the most common date
 * formats. Example formats that are detectable:
 *
 *    `2001/02/30`
 *    `2001-02-30 00:00:00`
 *    `2/30/2001 T04:05:60.7`
 *    `2 / 1 / 2011`
 *    `02/January`
 *
 * @param len Number of non special-symbol or numeric characters
 * @param decimal_count Number of '.' characters
 * @param colon_count Number of ':' characters
 * @param dash_count Number of '-' characters
 * @param slash_count Number of '/' characters
 *
 * @return `true` if it is date-like, `false` otherwise
 */
__device__ __inline__ bool is_datetime(long len, long decimal_count,
                                       long colon_count, long dash_count,
                                       long slash_count) {
  // Must not exceed count of longest month (September) plus `T` time indicator
  if (len > 10) {
    return false;
  }
  // Must not exceed more than one decimals or more than two time separators
  if (decimal_count > 1 || colon_count > 2) {
    return false;
  }
  // Must have one or two '-' or '/' but not both as date separators
  if ((dash_count > 0 && dash_count < 3 && slash_count == 0) ||
      (dash_count == 0 && slash_count > 0 && slash_count < 3)) {
    return true;
  }

  return false;
}

/**
 * @brief Returns true if the counters indicate a potentially valid float.
 * False positives are possible because positions are not taken into account.
 * For example, field "e.123-" would match the pattern.
 *
 * @param len Number of non special-symbol or numeric characters
 * @param digit_count Number of digits characters
 * @param decimal_count Number of '.' characters
 * @param dash_count Number of '-' characters
 * @param exponent_count Number of 'e or E' characters
 *
 * @return `true` if it is floating point-like, `false` otherwise
 */
__device__ __inline__ bool is_floatingpoint(long len, long digit_count,
                                            long decimal_count, long dash_count,
                                            long exponent_count) {
  // Can't have more than one exponent and one decimal point
  if (decimal_count > 1) return false;
  if (exponent_count > 1) return false;

  // Without the exponent or a decimal point, this is an integer, not a float
  if (decimal_count == 0 && exponent_count == 0) return false;

  // Can only have one '-' per component
  if (dash_count > 1 + exponent_count) return false;

  // If anything other than these characters is present, it's not a float
  if (digit_count + decimal_count + dash_count + exponent_count != len) {
    return false;
  }

  // Needs at least 1 digit, 2 if exponent is present
  if (digit_count < 1 + exponent_count) return false;

  return true;
}

/**
 * @brief CUDA kernel that parses and converts CSV data into cuDF column data.
 *
 * Data is processed in one row/record at a time, so the number of total
 * threads (tid) is equal to the number of rows.
 *
 * @param raw_csv The entire CSV data to read
 * @param opts A set of parsing options
 * @param num_records The number of lines/rows of CSV data
 * @param num_columns The number of columns of CSV data
 * @param column_flags Per-column parsing behavior flags
 * @param recStart The start the CSV data of interest
 * @param d_columnData The count for each column data type
 */
__global__ void dataTypeDetection(const char *raw_csv, const ParseOptions opts,
                                  size_t num_records, int num_columns,
                                  column_parse::flags *flags,
                                  const uint64_t *recStart,
                                  column_parse::stats *d_columnData) {
  // ThreadIds range per block, so also need the blockId
  // This is entry into the fields; threadId is an element within `num_records`
  long rec_id = threadIdx.x + (blockDim.x * blockIdx.x);

  // we can have more threads than data, make sure we are not past the end of
  // the data
  if (rec_id >= num_records) {
    return;
  }

  long start = recStart[rec_id];
  long stop = recStart[rec_id + 1];

  long pos = start;
  int col = 0;
  int actual_col = 0;

  // Going through all the columns of a given record
  while (col < num_columns) {
    if (start > stop) {
      break;
    }

    pos = seek_field_end(raw_csv, opts, pos, stop);

    // Checking if this is a column that the user wants --- user can filter
    // columns
    if (flags[col] & column_parse::enabled) {
      long tempPos = pos - 1;
      long field_len = pos - start;

      if (field_len <= 0 ||
          serializedTrieContains(opts.naValuesTrie, raw_csv + start,
                                 field_len)) {
        atomicAdd(&d_columnData[actual_col].countNULL, 1);
      } else if (serializedTrieContains(opts.trueValuesTrie, raw_csv + start,
                                        field_len) ||
                 serializedTrieContains(opts.falseValuesTrie, raw_csv + start,
                                        field_len)) {
        atomicAdd(&d_columnData[actual_col].countBool, 1);
      } else {
        long countNumber = 0;
        long countDecimal = 0;
        long countSlash = 0;
        long countDash = 0;
        long countPlus = 0;
        long countColon = 0;
        long countString = 0;
        long countExponent = 0;

        // Modify start & end to ignore whitespace and quotechars
        // This could possibly result in additional empty fields
        trim_field_start_end(raw_csv, &start, &tempPos);
        field_len = tempPos - start + 1;

        for (long startPos = start; startPos <= tempPos; startPos++) {
          if (is_digit(raw_csv[startPos])) {
            countNumber++;
            continue;
          }
          // Looking for unique characters that will help identify column types.
          switch (raw_csv[startPos]) {
            case '.':
              countDecimal++;
              break;
            case '-':
              countDash++;
              break;
            case '+':
              countPlus++;
              break;
            case '/':
              countSlash++;
              break;
            case ':':
              countColon++;
              break;
            case 'e':
            case 'E':
              if (startPos > start && startPos < tempPos) countExponent++;
              break;
            default:
              countString++;
              break;
          }
        }

        // Integers have to have the length of the string
        long int_req_number_cnt = field_len;
        // Off by one if they start with a minus sign
        if ((raw_csv[start] == '-' || raw_csv[start] == '+') && field_len > 1) {
          --int_req_number_cnt;
        }

        if (field_len == 0) {
          // Ignoring whitespace and quotes can result in empty fields
          atomicAdd(&d_columnData[actual_col].countNULL, 1);
        } else if (flags[col] & column_parse::as_datetime) {
          // PANDAS uses `object` dtype if the date is unparseable
          if (is_datetime(countString, countDecimal, countColon, countDash,
                          countSlash)) {
            atomicAdd(&d_columnData[actual_col].countDateAndTime, 1);
          } else {
            atomicAdd(&d_columnData[actual_col].countString, 1);
          }
        } else if (countNumber == int_req_number_cnt) {
          atomicAdd(&d_columnData[actual_col].countInt64, 1);
        } else if (is_floatingpoint(field_len, countNumber, countDecimal,
                                    countDash + countPlus, countExponent)) {
          atomicAdd(&d_columnData[actual_col].countFloat, 1);
        } else {
          atomicAdd(&d_columnData[actual_col].countString, 1);
        }
      }
      actual_col++;
    }
    pos++;
    start = pos;
    col++;
  }
}

/**
 * @brief Returns the numeric value of an ASCII/UTF-8 character. Specialization
 * for integral types. Handles hexadecimal digits, both uppercase and lowercase.
 * If the character is not a valid numeric digit then `0` is returned and
 * valid_flag is set to false.
 *
 * @param c ASCII or UTF-8 character
 * @param valid_flag Set to false if input is not valid. Unchanged otherwise.
 *
 * @return uint8_t Numeric value of the character, or `0`
 */
template <typename T,
          typename std::enable_if_t<std::is_integral<T>::value> * = nullptr>
__device__ __forceinline__ uint8_t decode_digit(char c, bool* valid_flag) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;

  *valid_flag = false;
  return 0;
}

/**
 * @brief Returns the numeric value of an ASCII/UTF-8 character. Specialization
 * for non-integral types. Handles only decimal digits. If the character is not
 * a valid numeric digit then `0` is returned and valid_flag is set to false.
 *
 * @param c ASCII or UTF-8 character
 * @param valid_flag Set to false if input is not valid. Unchanged otherwise.
 *
 * @return uint8_t Numeric value of the character, or `0`
 */
template <typename T,
          typename std::enable_if_t<!std::is_integral<T>::value> * = nullptr>
__device__ __forceinline__ uint8_t decode_digit(char c, bool* valid_flag) {
  if (c >= '0' && c <= '9') return c - '0';

  *valid_flag = false;
  return 0;
}

/**
 * @brief Parses a character string and returns its numeric value.
 *
 * @param data The character string for parse
 * @param start The index within data to start parsing from
 * @param end The end index within data to end parsing
 * @param opts The global parsing behavior options
 * @param base Base (radix) to use for conversion
 *
 * @return The parsed and converted value
 */
template <typename T>
__inline__ __device__ T parse_numeric(const char *data, long start, long end,
                                      ParseOptions const &opts, int base = 10) {
  T value = 0;
  bool all_digits_valid = true;

  // Handle negative values if necessary
  int32_t sign = 1;
  if (data[start] == '-') {
    sign = -1;
    start++;
  }

  // Handle the whole part of the number
  long index = start;
  while (index <= end) {
    if (data[index] == opts.decimal) {
      ++index;
      break;
    } else if (base == 10 && (data[index] == 'e' || data[index] == 'E')) {
      break;
    } else if (data[index] != opts.thousands && data[index] != '+') {
      value = (value * base) + decode_digit<T>(data[index], &all_digits_valid);
    }
    ++index;
  }

  if (std::is_floating_point<T>::value) {
    // Handle fractional part of the number if necessary
    double divisor = 1;
    while (index <= end) {
      if (data[index] == 'e' || data[index] == 'E') {
        ++index;
        break;
      } else if (data[index] != opts.thousands && data[index] != '+') {
        divisor /= base;
        value += decode_digit<T>(data[index], &all_digits_valid) * divisor;
      }
      ++index;
    }

    // Handle exponential part of the number if necessary
    if (index <= end) {
      const int32_t exponent_sign = data[index] == '-' ? -1 : 1;
      if (data[index] == '-' || data[index] == '+') {
        ++index;
      }
      int32_t exponent = 0;
      while (index <= end) {
          exponent = (exponent * 10) + decode_digit<T>(data[index++], &all_digits_valid);
      }
      if (exponent != 0) {
        value *= exp10(double(exponent * exponent_sign));
      }
    }
  }
  if (!all_digits_valid){
    return std::numeric_limits<T>::quiet_NaN();
  }

  return value * sign;
}

template <typename T, int base>
__inline__ __device__ T decode_value(const char *data, long start, long end,
                                     ParseOptions const &opts) {
  return parse_numeric<T>(data, start, end, opts, base);
}

template <typename T>
__inline__ __device__ T decode_value(const char *data, long start, long end,
                                     ParseOptions const &opts) {
  return parse_numeric<T>(data, start, end, opts);
}

template <>
__inline__ __device__ cudf::experimental::bool8 decode_value(
    const char *data, long start, long end, ParseOptions const &opts) {
  using value_type = typename cudf::experimental::bool8::value_type;
  return (parse_numeric<value_type>(data, start, end, opts) != 0)
             ? cudf::experimental::true_v
             : cudf::experimental::false_v;
}

template <>
__inline__ __device__ cudf::timestamp_D decode_value(const char *data,
                                                     long start, long end,
                                                     ParseOptions const &opts) {
  return parseDateFormat(data, start, end, opts.dayfirst);
}

template <>
__inline__ __device__ cudf::timestamp_s decode_value(const char *data,
                                                     long start, long end,
                                                     ParseOptions const &opts) {
  auto milli = parseDateTimeFormat(data, start, end, opts.dayfirst);
  return milli / 1000;
}

template <>
__inline__ __device__ cudf::timestamp_ms decode_value(
    const char *data, long start, long end, ParseOptions const &opts) {
  auto milli = parseDateTimeFormat(data, start, end, opts.dayfirst);
  return milli;
}

template <>
__inline__ __device__ cudf::timestamp_us decode_value(
    const char *data, long start, long end, ParseOptions const &opts) {
  auto milli = parseDateTimeFormat(data, start, end, opts.dayfirst);
  return milli * 1000;
}

template <>
__inline__ __device__ cudf::timestamp_ns decode_value(
    const char *data, long start, long end, ParseOptions const &opts) {
  auto milli = parseDateTimeFormat(data, start, end, opts.dayfirst);
  return milli * 1000000;
}

// The purpose of this is merely to allow compilation ONLY
template <>
__inline__ __device__ cudf::string_view decode_value(const char *data,
                                                     long start, long end,
                                                     ParseOptions const &opts) {
  return cudf::string_view{};
}

/**
 * @brief Functor for converting CSV raw data to typed value.
 */
struct decode_op {
  /**
   * @brief Dispatch for numeric types whose values can be convertible to
   * 0 or 1 to represent boolean false/true, based upon checking against a
   * true/false values list.
   *
   * @return bool Whether the parsed value is valid.
   */
  template <typename T,
            typename std::enable_if_t<
                std::is_integral<T>::value and
                !std::is_same<T, cudf::experimental::bool8>::value> * = nullptr>
  __host__ __device__ __forceinline__ bool operator()(
      const char *data, void *out_buffer, size_t row, long start, long end,
      ParseOptions const &opts, column_parse::flags flags) {
    auto &value{static_cast<T *>(out_buffer)[row]};

    // Check for user-specified true/false values first, where the output is
    // replaced with 1/0 respectively
    const size_t field_len = end - start + 1;
    if (serializedTrieContains(opts.trueValuesTrie, data + start, field_len)) {
      value = 1;
    } else if (serializedTrieContains(opts.falseValuesTrie, data + start,
                                      field_len)) {
      value = 0;
    } else {
      if (flags & column_parse::as_hexadecimal) {
        value = decode_value<T, 16>(data, start, end, opts);
      } else {
        value = decode_value<T>(data, start, end, opts);
      }
    }
    return true;
  }

  /**
   * @brief Dispatch for boolean type types.
   */
  template <typename T, typename std::enable_if_t<std::is_same<
                            T, cudf::experimental::bool8>::value> * = nullptr>
  __host__ __device__ __forceinline__ bool operator()(
      const char *data, void *out_buffer, size_t row, long start, long end,
      ParseOptions const &opts, column_parse::flags flags) {
    auto &value{static_cast<T *>(out_buffer)[row]};

    // Check for user-specified true/false values first, where the output is
    // replaced with 1/0 respectively
    const size_t field_len = end - start + 1;
    if (serializedTrieContains(opts.trueValuesTrie, data + start, field_len)) {
      value = 1;
    } else if (serializedTrieContains(opts.falseValuesTrie, data + start,
                                      field_len)) {
      value = 0;
    } else {
      value = decode_value<T>(data, start, end, opts);
    }
    return true;
  }

  /**
   * @brief Dispatch for floating points, which are set to NaN if the input 
   * is not valid. In such case, the validity mask is set to zero too.
   */
   template <typename T,
             typename std::enable_if_t<std::is_floating_point<T>::value> * = nullptr>
  __host__ __device__ __forceinline__ bool operator()(
      const char *data, void *out_buffer, size_t row, long start, long end,
      ParseOptions const &opts, column_parse::flags flags) {
    auto &value{static_cast<T *>(out_buffer)[row]};

    value = decode_value<T>(data, start, end, opts);
    return !std::isnan(value);
  }

  /**
   * @brief Dispatch for all other types.
   */
  template <typename T,
            typename std::enable_if_t<!std::is_integral<T>::value and 
            !std::is_floating_point<T>::value> * = nullptr>
  __host__ __device__ __forceinline__ bool operator()(
      const char *data, void *out_buffer, size_t row, long start, long end,
      ParseOptions const &opts, column_parse::flags flags) {
    auto &value{static_cast<T *>(out_buffer)[row]};

    value = decode_value<T>(data, start, end, opts);
    return true;
  }
};

/**---------------------------------------------------------------------------*
 * @brief CUDA kernel that parses and converts CSV data into cuDF column data.
 *
 * Data is processed one record at a time
 *
 * @param[in] raw_csv The entire CSV data to read
 * @param[in] opts A set of parsing options
 * @param[in] num_records The number of lines/rows of CSV data
 * @param[in] num_columns The number of columns of CSV data
 * @param[in] column_flags Per-column parsing behavior flags
 * @param[in] recStart The start the CSV data of interest
 * @param[in] dtype The data type of the column
 * @param[out] data The output column data
 * @param[out] valid The bitmaps indicating whether column fields are valid
 * @param[out] num_valid The numbers of valid fields in columns
 *---------------------------------------------------------------------------**/
__global__ void convertCsvToGdf(const char *raw_csv, const ParseOptions opts,
                                size_t num_records, size_t num_columns,
                                const column_parse::flags *flags,
                                const uint64_t *recStart,
                                cudf::data_type *dtype, void **data,
                                cudf::bitmask_type **valid) {
  // thread IDs range per block, so also need the block id
  long rec_id =
      threadIdx.x +
      (blockDim.x * blockIdx.x);  // this is entry into the field array - tid is
                                  // an elements within the num_entries array

  // we can have more threads than data, make sure we are not past the end of
  // the data
  if (rec_id >= num_records) return;

  long start = recStart[rec_id];
  long stop = recStart[rec_id + 1];

  long pos = start;
  int col = 0;
  int actual_col = 0;

  while (col < num_columns) {
    if (start > stop) break;

    pos = seek_field_end(raw_csv, opts, pos, stop);

    if (flags[col] & column_parse::enabled) {
      // check if the entire field is a NaN string - consistent with pandas
      const bool is_na = serializedTrieContains(opts.naValuesTrie,
                                                raw_csv + start, pos - start);

      // Modify start & end to ignore whitespace and quotechars
      long tempPos = pos - 1;
      if (!is_na && dtype[actual_col].id() != cudf::type_id::CATEGORY &&
          dtype[actual_col].id() != cudf::type_id::STRING) {
        trim_field_start_end(raw_csv, &start, &tempPos, opts.quotechar);
      }

      if (!is_na && start <= (tempPos)) {  // Empty fields are not legal values

        // Type dispatcher does not handle GDF_STRINGS
        if (dtype[actual_col].id() == cudf::type_id::STRING) {
          long end = pos;
          if (opts.keepquotes == false) {
            if ((raw_csv[start] == opts.quotechar) &&
                (raw_csv[end - 1] == opts.quotechar)) {
              start++;
              end--;
            }
          }
          auto str_list =
              static_cast<std::pair<const char *, size_t> *>(data[actual_col]);
          str_list[rec_id].first = raw_csv + start;
          str_list[rec_id].second = end - start;
        } else {
          if (cudf::experimental::type_dispatcher(dtype[actual_col], decode_op{},
                                              raw_csv, data[actual_col], rec_id,
                                              start, tempPos, opts, flags[col])){
            // set the valid bitmap - all bits were set to 0 to start
            set_bit(valid[actual_col], rec_id);
          }
        }
      } else if (dtype[actual_col].id() == cudf::type_id::STRING) {
        auto str_list =
            static_cast<std::pair<const char *, size_t> *>(data[actual_col]);
        str_list[rec_id].first = nullptr;
        str_list[rec_id].second = 0;
      }
      actual_col++;
    }
    pos++;
    start = pos;
    col++;
  }
}

cudaError_t __host__ DetectColumnTypes(
    const char *data, const uint64_t *row_starts, size_t num_rows,
    size_t num_columns, const ParseOptions &options, column_parse::flags *flags,
    column_parse::stats *stats, cudaStream_t stream) {
  // Calculate actual block count to use based on records count
  int blockSize = 0;    // suggested thread count to use
  int minGridSize = 0;  // minimum block count required
  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
                                     dataTypeDetection);
  const int gridSize = (num_rows + blockSize - 1) / blockSize;

  dataTypeDetection<<<gridSize, blockSize, 0, stream>>>(
      data, options, num_rows, num_columns, flags, row_starts, stats);

  return cudaSuccess;
}

cudaError_t __host__ DecodeRowColumnData(
    const char *data, const uint64_t *row_starts, size_t num_rows,
    size_t num_columns, const ParseOptions &options,
    const column_parse::flags *flags, cudf::data_type *dtypes, void **columns,
    cudf::bitmask_type **valids, cudaStream_t stream) {
  // Calculate actual block count to use based on records count
  int blockSize = 0;    // suggested thread count to use
  int minGridSize = 0;  // minimum block count required
  cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, convertCsvToGdf);
  const int gridSize = (num_rows + blockSize - 1) / blockSize;

  convertCsvToGdf<<<gridSize, blockSize, 0, stream>>>(
      data, options, num_rows, num_columns, flags, row_starts, dtypes, columns,
      valids);

  return cudaSuccess;
}

}  // namespace gpu
}  // namespace csv
}  // namespace io
}  // namespace cudf
