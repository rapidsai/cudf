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

#include "json_common.h"
#include "json_gpu.h"

#include <rmm/device_buffer.hpp>

#include <cudf/detail/utilities/hash_functions.cuh>
#include <cudf/detail/utilities/trie.cuh>

#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cudf/lists/list_view.cuh>
#include <cudf/strings/string_view.cuh>

#include <io/csv/datetime.cuh>
#include <io/utilities/parsing_utils.cuh>

namespace cudf {
namespace io {
namespace json {
namespace gpu {
using namespace ::cudf;

using string_pair = std::pair<const char *, size_t>;

namespace {
/**
 * @brief CUDA Kernel that modifies the start and stop offsets to exclude
 * the sections outside of the top level brackets.
 *
 * The top level brackets characters are excluded from the resulting range.
 * Parameter stop has the same semantics as end() in STL containers
 * (one past the last element)
 *
 * @param[in] data Pointer to the device buffer containing the data to process
 * @param[in,out] start Offset of the first character in the range
 * @param[in,out] stop Offset of the first character after the range
 *
 * @return void
 **/
__device__ void limit_range_to_brackets(const char *data, long &start, long &stop)
{
  while (start < stop && data[start] != '[' && data[start] != '{') { start++; }
  start++;

  while (start < stop && data[stop - 1] != ']' && data[stop - 1] != '}') { stop--; }
  stop--;
}

enum parse_state { PRE_NAME, NAME, POST_NAME, POST_NAME_QUOTE };

/**
 * @brief CUDA kernel that finds the end position of the next field name,
 * including the colon that separates the name from the field value.
 *
 * Returns the position after the colon that precedes the value token.
 *
 * @param[in] data Pointer to the device buffer containing the data to process
 * @param[in] opts Parsing options (e.g. delimiter and quotation character)
 * @param[in,out] start Offset of the first character in the range
 * @param[in] stop Offset of the first character after the range
 *
 * @return uint32_t Hash value of the field name
 **/
__device__ uint32_t parse_field_name(const char *data,
                                     const ParseOptions opts,
                                     long &start,
                                     long stop)
{
  parse_state state = PRE_NAME;
  auto name_start   = start;
  uint32_t hash_val = 0;
  for (auto pos = start; pos < stop; ++pos) {
    if (state == PRE_NAME && data[pos] == opts.quotechar) {
      state      = NAME;
      name_start = pos + 1;
    } else if (state == NAME && data[pos] == opts.quotechar && data[pos - 1] != '\\') {
      state = POST_NAME;
      hash_val =
        MurmurHash3_32<cudf::string_view>{}(cudf::string_view(data + name_start, pos - name_start));
    } else if (state == POST_NAME && data[pos] == ':') {
      start = pos + 1;
      break;
    }
  }
  return hash_val;
}

/**
 * @brief Decodes a numeric value base on templated cudf type T with specified
 * base.
 *
 * @param data The character string for parse
 * @param start The index within data to start parsing from
 * @param end The end index within data to end parsing
 * @param opts The global parsing behavior options
 *
 * @return The parsed numeric value
 **/
template <typename T, int base>
__inline__ __device__ T
decode_value(const char *data, long start, long end, ParseOptions const &opts)
{
  return cudf::io::gpu::parse_numeric<T, base>(data, start, end, opts);
}

/**
 * @brief Decodes a numeric value base on templated cudf type T
 *
 * @param data The character string for parse
 * @param start The index within data to start parsing from
 * @param end The end index within data to end parsing
 * @param opts The global parsing behavior options
 *
 * @return The parsed numeric value
 **/
template <typename T>
__inline__ __device__ T
decode_value(const char *data, long start, long end, ParseOptions const &opts)
{
  return cudf::io::gpu::parse_numeric<T>(data, start, end, opts);
}

/**
 * @brief Decodes a timestamp_D
 *
 * @param data The character string for parse
 * @param start The index within data to start parsing from
 * @param end The end index within data to end parsing
 * @param opts The global parsing behavior options
 *
 * @return The parsed timestamp_D
 **/
template <>
__inline__ __device__ cudf::timestamp_D decode_value(const char *data,
                                                     long start,
                                                     long end,
                                                     ParseOptions const &opts)
{
  return parseDateFormat(data, start, end, opts.dayfirst);
}

/**
 * @brief Decodes a timestamp_s
 *
 * @param data The character string for parse
 * @param start The index within data to start parsing from
 * @param end The end index within data to end parsing
 * @param opts The global parsing behavior options
 *
 * @return The parsed timestamp_s
 **/
template <>
__inline__ __device__ cudf::timestamp_s decode_value(const char *data,
                                                     long start,
                                                     long end,
                                                     ParseOptions const &opts)
{
  auto milli = parseDateTimeFormat(data, start, end, opts.dayfirst);
  return milli / 1000;
}

/**
 * @brief Decodes a timestamp_ms
 *
 * @param data The character string for parse
 * @param start The index within data to start parsing from
 * @param end The end index within data to end parsing
 * @param opts The global parsing behavior options
 *
 * @return The parsed timestamp_ms
 **/
template <>
__inline__ __device__ cudf::timestamp_ms decode_value(const char *data,
                                                      long start,
                                                      long end,
                                                      ParseOptions const &opts)
{
  auto milli = parseDateTimeFormat(data, start, end, opts.dayfirst);
  return milli;
}

/**
 * @brief Decodes a timestamp_us
 *
 * @param data The character string for parse
 * @param start The index within data to start parsing from
 * @param end The end index within data to end parsing
 * @param opts The global parsing behavior options
 *
 * @return The parsed timestamp_us
 **/
template <>
__inline__ __device__ cudf::timestamp_us decode_value(const char *data,
                                                      long start,
                                                      long end,
                                                      ParseOptions const &opts)
{
  auto milli = parseDateTimeFormat(data, start, end, opts.dayfirst);
  return milli * 1000;
}

/**
 * @brief Decodes a timestamp_ns
 *
 * @param data The character string for parse
 * @param start The index within data to start parsing from
 * @param end The end index within data to end parsing
 * @param opts The global parsing behavior options
 *
 * @return The parsed timestamp_ns
 **/
template <>
__inline__ __device__ cudf::timestamp_ns decode_value(const char *data,
                                                      long start,
                                                      long end,
                                                      ParseOptions const &opts)
{
  auto milli = parseDateTimeFormat(data, start, end, opts.dayfirst);
  return milli * 1000000;
}

// The purpose of this is merely to allow compilation ONLY
// TODO : make this work for json
#ifndef DURATION_DECODE_VALUE
#define DURATION_DECODE_VALUE(Type)                                   \
  template <>                                                         \
  __inline__ __device__ Type decode_value(                            \
    const char *data, long start, long end, ParseOptions const &opts) \
  {                                                                   \
    return Type{};                                                    \
  }
#endif
DURATION_DECODE_VALUE(duration_D)
DURATION_DECODE_VALUE(duration_s)
DURATION_DECODE_VALUE(duration_ms)
DURATION_DECODE_VALUE(duration_us)
DURATION_DECODE_VALUE(duration_ns)

// The purpose of these is merely to allow compilation ONLY
template <>
__inline__ __device__ cudf::string_view decode_value(const char *data,
                                                     long start,
                                                     long end,
                                                     ParseOptions const &opts)
{
  return cudf::string_view{};
}
template <>
__inline__ __device__ cudf::dictionary32 decode_value(const char *data,
                                                      long start,
                                                      long end,
                                                      ParseOptions const &opts)
{
  return cudf::dictionary32{};
}
template <>
__inline__ __device__ cudf::list_view decode_value(const char *data,
                                                   long start,
                                                   long end,
                                                   ParseOptions const &opts)
{
  return cudf::list_view{};
}

/**
 * @brief Functor for converting plain text data to cuDF data type value.
 **/
struct ConvertFunctor {
  /**
   * @brief Template specialization for operator() for types whose values can be
   * convertible to a 0 or 1 to represent false/true. The converting is done by
   * checking against the default and user-specified true/false values list.
   *
   * It is handled here rather than within convertStrToValue() as that function
   * is used by other types (ex. timestamp) that aren't 'booleable'.
   **/
  template <typename T, typename std::enable_if_t<std::is_integral<T>::value> * = nullptr>
  __host__ __device__ __forceinline__ bool operator()(const char *data,
                                                      void *output_columns,
                                                      long row,
                                                      long start,
                                                      long end,
                                                      const ParseOptions &opts)
  {
    T &value{static_cast<T *>(output_columns)[row]};

    // Check for user-specified true/false values first, where the output is
    // replaced with 1/0 respectively
    const size_t field_len = end - start + 1;
    if (serializedTrieContains(opts.trueValuesTrie, data + start, field_len)) {
      value = 1;
    } else if (serializedTrieContains(opts.falseValuesTrie, data + start, field_len)) {
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
  template <typename T, typename std::enable_if_t<std::is_floating_point<T>::value> * = nullptr>
  __host__ __device__ __forceinline__ bool operator()(
    const char *data, void *out_buffer, size_t row, long start, long end, ParseOptions const &opts)
  {
    auto &value{static_cast<T *>(out_buffer)[row]};
    value = decode_value<T>(data, start, end, opts);
    return !std::isnan(value);
  }

  /**
   * @brief Default template operator() dispatch specialization all data types
   * (including wrapper types) that is not covered by above.
   **/
  template <typename T,
            typename std::enable_if_t<!std::is_floating_point<T>::value and
                                      !std::is_integral<T>::value> * = nullptr>
  __host__ __device__ __forceinline__ bool operator()(const char *data,
                                                      void *output_columns,
                                                      long row,
                                                      long start,
                                                      long end,
                                                      const ParseOptions &opts)
  {
    T &value{static_cast<T *>(output_columns)[row]};
    value = decode_value<T>(data, start, end, opts);

    return true;
  }
};

/**
 * @brief Checks whether the given character is a whitespace character.
 *
 * @param[in] ch The character to check
 *
 * @return True if the input is whitespace, False otherwise
 **/
__inline__ __device__ bool is_whitespace(char ch) { return ch == '\t' || ch == ' '; }

/**
 * @brief Scans a character stream within a range, and adjusts the start and end
 * indices of the range to ignore whitespace and quotation characters.
 *
 * @param[in] data The character stream to scan
 * @param[in,out] start The start index to adjust
 * @param[in,out] end The end index to adjust
 * @param[in] quotechar The character used to denote quotes
 *
 * @return Adjusted or unchanged start_idx and end_idx
 **/
__inline__ __device__ void trim_field_start_end(const char *data,
                                                long *start,
                                                long *end,
                                                char quotechar = '\0')
{
  while ((*start < *end) && is_whitespace(data[*start])) { (*start)++; }
  if ((*start < *end) && data[*start] == quotechar) { (*start)++; }
  while ((*start <= *end) && is_whitespace(data[*end])) { (*end)--; }
  if ((*start <= *end) && data[*end] == quotechar) { (*end)--; }
}

/**
 * @brief Returns true is the input character is a valid digit.
 * Supports both decimal and hexadecimal digits (uppercase and lowercase).
 *
 * @param c Character to check
 * @param is_hex Whether to check as a hexadecimal
 *
 * @return `true` if it is digit-like, `false` otherwise
 */
__device__ __inline__ bool is_digit(char c, bool is_hex = false)
{
  if (c >= '0' && c <= '9') return true;

  if (is_hex) {
    if (c >= 'A' && c <= 'F') return true;
    if (c >= 'a' && c <= 'f') return true;
  }

  return false;
}

/**
 * @brief Returns true if the counters indicate a potentially valid float.
 * False positives are possible because positions are not taken into account.
 * For example, field "e.123-" would match the pattern.
 */
__device__ __inline__ bool is_like_float(
  long len, long digit_cnt, long decimal_cnt, long dash_cnt, long exponent_cnt)
{
  // Can't have more than one exponent and one decimal point
  if (decimal_cnt > 1) return false;
  if (exponent_cnt > 1) return false;
  // Without the exponent or a decimal point, this is an integer, not a float
  if (decimal_cnt == 0 && exponent_cnt == 0) return false;

  // Can only have one '-' per component
  if (dash_cnt > 1 + exponent_cnt) return false;

  // If anything other than these characters is present, it's not a float
  if (digit_cnt + decimal_cnt + dash_cnt + exponent_cnt != len) return false;

  // Needs at least 1 digit, 2 if exponent is present
  if (digit_cnt < 1 + exponent_cnt) return false;

  return true;
}

/**
 * @brief CUDA kernel that parses and converts plain text data into cuDF column data.
 *
 * Data is processed one record at a time
 *
 * @param[in] data The entire data to read
 * @param[in] data_size Size of the data buffer, in bytes
 * @param[in] rec_starts The start of each data record
 * @param[in] num_records The number of lines/rows
 * @param[in] dtypes The data type of each column
 * @param[in] opts A set of parsing options
 * @param[out] output_columns The output column data
 * @param[in] num_columns The number of columns
 * @param[out] valid_fields The bitmaps indicating whether column fields are valid
 * @param[out] num_valid_fields The numbers of valid fields in columns
 *
 * @return void
 **/
__global__ void convert_json_to_columns_kernel(const char *data,
                                               size_t data_size,
                                               const uint64_t *rec_starts,
                                               cudf::size_type num_records,
                                               const data_type *dtypes,
                                               ParseOptions opts,
                                               void *const *output_columns,
                                               int num_columns,
                                               bitmask_type *const *valid_fields,
                                               cudf::size_type *num_valid_fields)
{
  const long rec_id = threadIdx.x + (blockDim.x * blockIdx.x);
  if (rec_id >= num_records) return;

  long start = rec_starts[rec_id];
  // has the same semantics as end() in STL containers (one past last element)
  long stop = ((rec_id < num_records - 1) ? rec_starts[rec_id + 1] : data_size);

  limit_range_to_brackets(data, start, stop);
  const bool is_object = (data[start - 1] == '{');

  for (int col = 0; col < num_columns && start < stop; col++) {
    auto dst_col = col;
    if (is_object) { parse_field_name(data, opts, start, stop); }
    // field_end is at the next delimiter/newline
    const long field_end = cudf::io::gpu::seek_field_end(data, opts, start, stop);
    long field_data_last = field_end - 1;
    // Modify start & end to ignore whitespace and quotechars
    trim_field_start_end(data, &start, &field_data_last, opts.quotechar);
    // Empty fields are not legal values
    if (start <= field_data_last &&
        !serializedTrieContains(opts.naValuesTrie, data + start, field_end - start)) {
      // Type dispatcher does not handle strings
      if (dtypes[dst_col].id() == type_id::STRING) {
        auto str_list           = static_cast<string_pair *>(output_columns[dst_col]);
        str_list[rec_id].first  = data + start;
        str_list[rec_id].second = field_data_last - start + 1;

        // set the valid bitmap - all bits were set to 0 to start
        set_bit(valid_fields[dst_col], rec_id);
        atomicAdd(&num_valid_fields[dst_col], 1);
      } else {
        if (cudf::type_dispatcher(dtypes[dst_col],
                                  ConvertFunctor{},
                                  data,
                                  output_columns[dst_col],
                                  rec_id,
                                  start,
                                  field_data_last,
                                  opts)) {
          // set the valid bitmap - all bits were set to 0 to start
          set_bit(valid_fields[dst_col], rec_id);
          atomicAdd(&num_valid_fields[dst_col], 1);
        }
      }
    } else if (dtypes[dst_col].id() == type_id::STRING) {
      auto str_list           = static_cast<string_pair *>(output_columns[dst_col]);
      str_list[rec_id].first  = nullptr;
      str_list[rec_id].second = 0;
    }
    start = field_end + 1;
  }
}

/**
 * @brief CUDA kernel that processes a buffer of data and determines information about the
 * column types within.
 *
 * Data is processed in one row/record at a time, so the number of total
 * threads (tid) is equal to the number of rows.
 *
 * @param[in] data Input data buffer
 * @param[in] data_size Size of the data buffer, in bytes
 * @param[in] opts A set of parsing options
 * @param[in] num_columns The number of columns of input data
 * @param[in] rec_starts The start the input data of interest
 * @param[in] num_records The number of lines/rows of input data
 * @param[out] column_infos The count for each column data type
 *
 * @returns void
 **/
__global__ void detect_json_data_types(const char *data,
                                       size_t data_size,
                                       const ParseOptions opts,
                                       int num_columns,
                                       const uint64_t *rec_starts,
                                       cudf::size_type num_records,
                                       ColumnInfo *column_infos)
{
  long rec_id = threadIdx.x + (blockDim.x * blockIdx.x);
  if (rec_id >= num_records) return;

  long start = rec_starts[rec_id];
  // has the same semantics as end() in STL containers (one past last element)
  long stop = ((rec_id < num_records - 1) ? rec_starts[rec_id + 1] : data_size);

  limit_range_to_brackets(data, start, stop);
  const bool is_object = (data[start - 1] == '{');

  for (int col = 0; col < num_columns; col++) {
    if (is_object) { parse_field_name(data, opts, start, stop); }
    auto field_start     = start;
    const long field_end = cudf::io::gpu::seek_field_end(data, opts, field_start, stop);
    long field_data_last = field_end - 1;
    trim_field_start_end(data, &field_start, &field_data_last);
    const int field_len = field_data_last - field_start + 1;
    // Advance the start offset
    start = field_end + 1;

    // Checking if the field is empty
    if (field_start > field_data_last ||
        serializedTrieContains(opts.naValuesTrie, data + field_start, field_len)) {
      atomicAdd(&column_infos[col].null_count, 1);
      continue;
    }
    // Don't need counts to detect strings, any field in quotes is deduced to be a string
    if (data[field_start] == opts.quotechar && data[field_data_last] == opts.quotechar) {
      atomicAdd(&column_infos[col].string_count, 1);
      continue;
    }

    int digit_count    = 0;
    int decimal_count  = 0;
    int slash_count    = 0;
    int dash_count     = 0;
    int colon_count    = 0;
    int exponent_count = 0;
    int other_count    = 0;

    const bool maybe_hex =
      ((field_len > 2 && data[field_start] == '0' && data[field_start + 1] == 'x') ||
       (field_len > 3 && data[field_start] == '-' && data[field_start + 1] == '0' &&
        data[field_start + 2] == 'x'));
    for (long pos = field_start; pos <= field_data_last; pos++) {
      if (is_digit(data[pos], maybe_hex)) {
        digit_count++;
        continue;
      }
      // Looking for unique characters that will help identify column types
      switch (data[pos]) {
        case '.': decimal_count++; break;
        case '-': dash_count++; break;
        case '/': slash_count++; break;
        case ':': colon_count++; break;
        case 'e':
        case 'E':
          if (!maybe_hex && pos > field_start && pos < field_data_last) exponent_count++;
          break;
        default: other_count++; break;
      }
    }

    // Integers have to have the length of the string
    int int_req_number_cnt = field_len;
    // Off by one if they start with a minus sign
    if (data[field_start] == '-' && field_len > 1) { --int_req_number_cnt; }
    // Off by one if they are a hexadecimal number
    if (maybe_hex) { --int_req_number_cnt; }
    if (serializedTrieContains(opts.trueValuesTrie, data + field_start, field_len) ||
        serializedTrieContains(opts.falseValuesTrie, data + field_start, field_len)) {
      atomicAdd(&column_infos[col].bool_count, 1);
    } else if (digit_count == int_req_number_cnt) {
      atomicAdd(&column_infos[col].int_count, 1);
    } else if (is_like_float(field_len, digit_count, decimal_count, dash_count, exponent_count)) {
      atomicAdd(&column_infos[col].float_count, 1);
    }
    // A date-time field cannot have more than 3 non-special characters
    // A number field cannot have more than one decimal point
    else if (other_count > 3 || decimal_count > 1) {
      atomicAdd(&column_infos[col].string_count, 1);
    } else {
      // A date field can have either one or two '-' or '\'; A legal combination will only have one
      // of them To simplify the process of auto column detection, we are not covering all the
      // date-time formation permutations
      if ((dash_count > 0 && dash_count <= 2 && slash_count == 0) ||
          (dash_count == 0 && slash_count > 0 && slash_count <= 2)) {
        if (colon_count <= 2) {
          atomicAdd(&column_infos[col].datetime_count, 1);
        } else {
          atomicAdd(&column_infos[col].string_count, 1);
        }
      } else {
        // Default field type is string
        atomicAdd(&column_infos[col].string_count, 1);
      }
    }
  }
}

/**
 * @brief TODO
 *
 * @param[in] data Input data buffer
 * @param[in] data_size Size of the data buffer, in bytes
 * @param[in] opts A set of parsing options
 * @param[in] rec_starts The start the input data of interest
 * @param[in] num_records The number of lines/rows of input data
 * @param[out] tag_cnt TODO
 *
 * @returns void
 **/
__global__ void collect_field_names_info_kernel(const char *data,
                                                size_t data_size,
                                                const ParseOptions opts,
                                                const uint64_t *rec_starts,
                                                cudf::size_type num_records,
                                                unsigned long long int *names_cnt,
                                                mutable_table_device_view *names_info)
{
  long rec_id = threadIdx.x + (blockDim.x * blockIdx.x);
  if (rec_id >= num_records) return;

  auto const start = rec_starts[rec_id];
  // has the same semantics as end() in STL containers (one past last element)
  auto const stop = ((rec_id < num_records - 1) ? rec_starts[rec_id + 1] : data_size);

  parse_state st       = PRE_NAME;
  auto last_name_start = start;
  for (auto pos = start; pos < stop; ++pos) {
    if (st == PRE_NAME && data[pos] == opts.quotechar) {
      st              = NAME;
      last_name_start = pos + 1;
    } else if (st == NAME && data[pos] == opts.quotechar && data[pos - 1] != '\\') {
      st       = POST_NAME;
      auto idx = atomicAdd(names_cnt, 1);
      if (nullptr != names_info) {
        auto len                                     = pos - last_name_start;
        names_info->column(0).element<uint64_t>(idx) = last_name_start;
        names_info->column(1).element<uint16_t>(idx) = len;
        names_info->column(2).element<uint32_t>(idx) =
          MurmurHash3_32<cudf::string_view>{}(cudf::string_view(data + last_name_start, len));
      }
    } else if (st == POST_NAME && data[pos] == opts.quotechar) {
      st = POST_NAME_QUOTE;
    } else if (st == POST_NAME_QUOTE && data[pos] == opts.quotechar && data[pos - 1] != '\\') {
      st = POST_NAME;
    } else if (st == POST_NAME && data[pos] == opts.delimiter) {
      st = PRE_NAME;
    }
  }
}

}  // namespace

/**
 * @copydoc cudf::io::json::gpu::convert_json_to_columns
 *
 **/
void convert_json_to_columns(rmm::device_buffer const &input_data,
                             data_type *const dtypes,
                             void *const *output_columns,
                             cudf::size_type num_records,
                             cudf::size_type num_columns,
                             const uint64_t *rec_starts,
                             bitmask_type *const *valid_fields,
                             cudf::size_type *num_valid_fields,
                             ParseOptions const &opts,
                             cudaStream_t stream)
{
  int block_size;
  int min_grid_size;
  CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(
    &min_grid_size, &block_size, convert_json_to_columns_kernel));

  const int grid_size = (num_records + block_size - 1) / block_size;

  convert_json_to_columns_kernel<<<grid_size, block_size, 0, stream>>>(
    static_cast<const char *>(input_data.data()),
    input_data.size(),
    rec_starts,
    num_records,
    dtypes,
    opts,
    output_columns,
    num_columns,
    valid_fields,
    num_valid_fields);

  CUDA_TRY(cudaGetLastError());
}

/**
 * @copydoc cudf::io::json::gpu::detect_data_types
 *
 **/
void detect_data_types(ColumnInfo *column_infos,
                       const char *data,
                       size_t data_size,
                       const ParseOptions &options,
                       int num_columns,
                       const uint64_t *rec_starts,
                       cudf::size_type num_records,
                       cudaStream_t stream)
{
  int block_size;
  int min_grid_size;
  CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, detect_json_data_types));

  // Calculate actual block count to use based on records count
  const int grid_size = (num_records + block_size - 1) / block_size;

  detect_json_data_types<<<grid_size, block_size, 0, stream>>>(
    data, data_size, options, num_columns, rec_starts, num_records, column_infos);

  CUDA_TRY(cudaGetLastError());
}

/**
 * @copydoc cudf::io::json::gpu::gpu_collect_field_names_info
 */
void collect_field_names_info(const char *data,
                              size_t data_size,
                              const ParseOptions &options,
                              const uint64_t *rec_starts,
                              cudf::size_type num_records,
                              unsigned long long int *names_cnt,
                              mutable_table_device_view *names_info,
                              cudaStream_t stream)
{
  int block_size;
  int min_grid_size;
  CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(
    &min_grid_size, &block_size, collect_field_names_info_kernel));

  // Calculate actual block count to use based on records count
  const int grid_size = (num_records + block_size - 1) / block_size;

  collect_field_names_info_kernel<<<grid_size, block_size, 0, stream>>>(
    data, data_size, options, rec_starts, num_records, names_cnt, names_info);

  CUDA_TRY(cudaGetLastError());
}

}  // namespace gpu
}  // namespace json
}  // namespace io
}  // namespace cudf
