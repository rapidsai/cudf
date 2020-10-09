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

#include "csv_common.h"
#include "csv_gpu.h"
#include "datetime.cuh"

#include <io/utilities/block_utils.cuh>
#include <io/utilities/parsing_utils.cuh>

#include <cudf/detail/utilities/trie.cuh>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/lists/list_view.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <thrust/detail/copy.h>
#include <thrust/transform.h>

#include <type_traits>

using namespace ::cudf::io;

using cudf::detail::device_span;

namespace cudf {
namespace io {
namespace csv {
namespace gpu {

/// Block dimension for dtype detection and conversion kernels
constexpr uint32_t csvparse_block_dim = 128;

/*
 * @brief Checks whether the given character is a whitespace character.
 *
 * @param ch The character to check
 *
 * @return True if the input is whitespace, False otherwise
 */
__device__ __inline__ bool is_whitespace(char c) { return c == '\t' || c == ' '; }

/*
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
__device__ __inline__ void trim_field_start_end(const char *data,
                                                long *start,
                                                long *end,
                                                char quotechar = '\0')
{
  while ((*start < *end) && is_whitespace(data[*start])) { (*start)++; }
  if ((*start < *end) && data[*start] == quotechar) { (*start)++; }
  while ((*start <= *end) && is_whitespace(data[*end])) { (*end)--; }
  if ((*start <= *end) && data[*end] == quotechar) { (*end)--; }
}

/*
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

/*
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
__device__ __inline__ bool is_datetime(
  long len, long decimal_count, long colon_count, long dash_count, long slash_count)
{
  // Must not exceed count of longest month (September) plus `T` time indicator
  if (len > 10) { return false; }
  // Must not exceed more than one decimals or more than two time separators
  if (decimal_count > 1 || colon_count > 2) { return false; }
  // Must have one or two '-' or '/' but not both as date separators
  if ((dash_count > 0 && dash_count < 3 && slash_count == 0) ||
      (dash_count == 0 && slash_count > 0 && slash_count < 3)) {
    return true;
  }

  return false;
}

/*
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
__device__ __inline__ bool is_floatingpoint(
  long len, long digit_count, long decimal_count, long dash_count, long exponent_count)
{
  // Can't have more than one exponent and one decimal point
  if (decimal_count > 1) return false;
  if (exponent_count > 1) return false;

  // Without the exponent or a decimal point, this is an integer, not a float
  if (decimal_count == 0 && exponent_count == 0) return false;

  // Can only have one '-' per component
  if (dash_count > 1 + exponent_count) return false;

  // If anything other than these characters is present, it's not a float
  if (digit_count + decimal_count + dash_count + exponent_count != len) { return false; }

  // Needs at least 1 digit, 2 if exponent is present
  if (digit_count < 1 + exponent_count) return false;

  return true;
}

/*
 * @brief CUDA kernel that parses and converts CSV data into cuDF column data.
 *
 * Data is processed in one row/record at a time, so the number of total
 * threads (tid) is equal to the number of rows.
 *
 * @param opts A set of parsing options
 * @param csv_text The entire CSV data to read
 * @param column_flags Per-column parsing behavior flags
 * @param row_offsets The start the CSV data of interest
 * @param d_columnData The count for each column data type
 */
__global__ void __launch_bounds__(csvparse_block_dim)
  data_type_detection(ParseOptions const opts,
                      device_span<char const> const csv_text,
                      device_span<column_parse::flags const> const column_flags,
                      device_span<uint64_t const> const row_offsets,
                      device_span<column_parse::stats> d_columnData)
{
  auto raw_csv = csv_text.data();

  // ThreadIds range per block, so also need the blockId
  // This is entry into the fields; threadId is an element within `num_records`
  long rec_id      = threadIdx.x + (blockDim.x * blockIdx.x);
  long rec_id_next = rec_id + 1;

  // we can have more threads than data, make sure we are not past the end of
  // the data
  if (rec_id_next >= row_offsets.size()) { return; }

  long start = row_offsets[rec_id];
  long stop  = row_offsets[rec_id_next];

  long pos       = start;
  int col        = 0;
  int actual_col = 0;

  // Going through all the columns of a given record
  while (col < column_flags.size()) {
    if (start > stop) { break; }

    pos = cudf::io::gpu::seek_field_end(raw_csv + pos, raw_csv + stop, opts) - raw_csv;

    // Checking if this is a column that the user wants --- user can filter
    // columns
    if (column_flags[col] & column_parse::enabled) {
      long tempPos   = pos - 1;
      long field_len = pos - start;

      if (field_len <= 0 || serializedTrieContains(opts.naValuesTrie, raw_csv + start, field_len)) {
        atomicAdd(&d_columnData[actual_col].countNULL, 1);
      } else if (serializedTrieContains(opts.trueValuesTrie, raw_csv + start, field_len) ||
                 serializedTrieContains(opts.falseValuesTrie, raw_csv + start, field_len)) {
        atomicAdd(&d_columnData[actual_col].countBool, 1);
      } else if (cudf::io::gpu::is_infinity(raw_csv + start, raw_csv + tempPos)) {
        atomicAdd(&d_columnData[actual_col].countFloat, 1);
      } else {
        long countNumber   = 0;
        long countDecimal  = 0;
        long countSlash    = 0;
        long countDash     = 0;
        long countPlus     = 0;
        long countColon    = 0;
        long countString   = 0;
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
            case '.': countDecimal++; break;
            case '-': countDash++; break;
            case '+': countPlus++; break;
            case '/': countSlash++; break;
            case ':': countColon++; break;
            case 'e':
            case 'E':
              if (startPos > start && startPos < tempPos) countExponent++;
              break;
            default: countString++; break;
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
        } else if (column_flags[col] & column_parse::as_datetime) {
          // PANDAS uses `object` dtype if the date is unparseable
          if (is_datetime(countString, countDecimal, countColon, countDash, countSlash)) {
            atomicAdd(&d_columnData[actual_col].countDateAndTime, 1);
          } else {
            atomicAdd(&d_columnData[actual_col].countString, 1);
          }
        } else if (countNumber == int_req_number_cnt) {
          atomicAdd(&d_columnData[actual_col].countInt64, 1);
        } else if (is_floatingpoint(
                     field_len, countNumber, countDecimal, countDash + countPlus, countExponent)) {
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

template <typename T, int base>
__inline__ __device__ T decode_value(char const *begin, char const *end, ParseOptions const &opts)
{
  return cudf::io::gpu::parse_numeric<T, base>(begin, end, opts);
}

template <typename T>
__inline__ __device__ T decode_value(char const *begin, char const *end, ParseOptions const &opts)
{
  return cudf::io::gpu::parse_numeric<T>(begin, end, opts);
}

template <>
__inline__ __device__ cudf::timestamp_D decode_value(char const *begin,
                                                     char const *end,
                                                     ParseOptions const &opts)
{
  return timestamp_D{cudf::duration_D{parseDateFormat(begin, end, opts.dayfirst)}};
}

template <>
__inline__ __device__ cudf::timestamp_s decode_value(char const *begin,
                                                     char const *end,
                                                     ParseOptions const &opts)
{
  auto milli = parseDateTimeFormat(begin, end, opts.dayfirst);
  return timestamp_s{cudf::duration_s{milli / 1000}};
}

template <>
__inline__ __device__ cudf::timestamp_ms decode_value(char const *begin,
                                                      char const *end,
                                                      ParseOptions const &opts)
{
  auto milli = parseDateTimeFormat(begin, end, opts.dayfirst);
  return timestamp_ms{cudf::duration_ms{milli}};
}

template <>
__inline__ __device__ cudf::timestamp_us decode_value(char const *begin,
                                                      char const *end,
                                                      ParseOptions const &opts)
{
  auto milli = parseDateTimeFormat(begin, end, opts.dayfirst);
  return timestamp_us{cudf::duration_us{milli * 1000}};
}

template <>
__inline__ __device__ cudf::timestamp_ns decode_value(char const *begin,
                                                      char const *end,
                                                      ParseOptions const &opts)
{
  auto milli = parseDateTimeFormat(begin, end, opts.dayfirst);
  return timestamp_ns{cudf::duration_ns{milli * 1000000}};
}

#ifndef DURATION_DECODE_VALUE
#define DURATION_DECODE_VALUE(Type)                                 \
  template <>                                                       \
  __inline__ __device__ Type decode_value(                          \
    const char *begin, const char *end, ParseOptions const &opts)   \
  {                                                                 \
    return Type{parseTimeDeltaFormat<Type>(begin, 0, end - begin)}; \
  }
#endif
DURATION_DECODE_VALUE(duration_D)
DURATION_DECODE_VALUE(duration_s)
DURATION_DECODE_VALUE(duration_ms)
DURATION_DECODE_VALUE(duration_us)
DURATION_DECODE_VALUE(duration_ns)

// The purpose of this is merely to allow compilation ONLY
// TODO : make this work for csv
template <>
__inline__ __device__ cudf::string_view decode_value(char const *begin,
                                                     char const *end,
                                                     ParseOptions const &opts)
{
  return cudf::string_view{};
}

// The purpose of this is merely to allow compilation ONLY
template <>
__inline__ __device__ cudf::dictionary32 decode_value(char const *begin,
                                                      char const *end,
                                                      ParseOptions const &opts)
{
  return cudf::dictionary32{};
}

// The purpose of this is merely to allow compilation ONLY
// TODO : make this work for csv
template <>
__inline__ __device__ cudf::list_view decode_value(char const *begin,
                                                   char const *end,
                                                   ParseOptions const &opts)
{
  return cudf::list_view{};
}

// The purpose of this is merely to allow compilation ONLY
// TODO : make this work for csv
template <>
__inline__ __device__ numeric::decimal32 decode_value(char const *begin,
                                                      char const *end,
                                                      ParseOptions const &opts)
{
  return numeric::decimal32{};
}

// The purpose of this is merely to allow compilation ONLY
// TODO : make this work for csv
template <>
__inline__ __device__ numeric::decimal64 decode_value(char const *begin,
                                                      char const *end,
                                                      ParseOptions const &opts)
{
  return numeric::decimal64{};
}

// The purpose of this is merely to allow compilation ONLY
// TODO : make this work for csv
template <>
__inline__ __device__ cudf::struct_view decode_value(char const *begin,
                                                     char const *end,
                                                     ParseOptions const &opts)
{
  return cudf::struct_view{};
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
            typename std::enable_if_t<std::is_integral<T>::value and !std::is_same<T, bool>::value>
              * = nullptr>
  __host__ __device__ __forceinline__ bool operator()(void *out_buffer,
                                                      size_t row,
                                                      char const *begin,
                                                      char const *end,
                                                      ParseOptions const &opts,
                                                      column_parse::flags flags)
  {
    static_cast<T *>(out_buffer)[row] = [&]() {
      // Check for user-specified true/false values first, where the output is
      // replaced with 1/0 respectively
      const size_t field_len = end - begin + 1;
      if (serializedTrieContains(opts.trueValuesTrie, begin, field_len)) {
        return static_cast<T>(1);
      } else if (serializedTrieContains(opts.falseValuesTrie, begin, field_len)) {
        return static_cast<T>(0);
      } else {
        if (flags & column_parse::as_hexadecimal) {
          return decode_value<T, 16>(begin, end, opts);
        } else {
          return decode_value<T>(begin, end, opts);
        }
      }
    }();

    return true;
  }

  /**
   * @brief Dispatch for boolean type types.
   */
  template <typename T, typename std::enable_if_t<std::is_same<T, bool>::value> * = nullptr>
  __host__ __device__ __forceinline__ bool operator()(void *out_buffer,
                                                      size_t row,
                                                      char const *begin,
                                                      char const *end,
                                                      ParseOptions const &opts,
                                                      column_parse::flags flags)
  {
    auto &value{static_cast<T *>(out_buffer)[row]};

    // Check for user-specified true/false values first, where the output is
    // replaced with 1/0 respectively
    const size_t field_len = end - begin + 1;
    if (serializedTrieContains(opts.trueValuesTrie, begin, field_len)) {
      value = 1;
    } else if (serializedTrieContains(opts.falseValuesTrie, begin, field_len)) {
      value = 0;
    } else {
      value = decode_value<T>(begin, end, opts);
    }
    return true;
  }

  /**
   * @brief Dispatch for floating points, which are set to NaN if the input
   * is not valid. In such case, the validity mask is set to zero too.
   */
  template <typename T, typename std::enable_if_t<std::is_floating_point<T>::value> * = nullptr>
  __host__ __device__ __forceinline__ bool operator()(void *out_buffer,
                                                      size_t row,
                                                      char const *begin,
                                                      char const *end,
                                                      ParseOptions const &opts,
                                                      column_parse::flags flags)
  {
    auto &value{static_cast<T *>(out_buffer)[row]};

    value = decode_value<T>(begin, end, opts);
    return !std::isnan(value);
  }

  /**
   * @brief Dispatch for all other types.
   */
  template <typename T,
            typename std::enable_if_t<!std::is_integral<T>::value and
                                      !std::is_floating_point<T>::value> * = nullptr>
  __host__ __device__ __forceinline__ bool operator()(void *out_buffer,
                                                      size_t row,
                                                      char const *begin,
                                                      char const *end,
                                                      ParseOptions const &opts,
                                                      column_parse::flags flags)
  {
    auto &value{static_cast<T *>(out_buffer)[row]};

    value = decode_value<T>(begin, end, opts);
    return true;
  }
};

/**
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
 **/
__global__ void __launch_bounds__(csvparse_block_dim)
  convert_csv_to_cudf(cudf::io::ParseOptions options,
                      device_span<char const> data,
                      device_span<column_parse::flags const> column_flags,
                      device_span<uint64_t const> row_offsets,
                      device_span<cudf::data_type const> dtypes,
                      device_span<void *> columns,
                      device_span<cudf::bitmask_type *> valids)
{
  auto raw_csv = data.data();
  // thread IDs range per block, so also need the block id.
  // this is entry into the field array - tid is an elements within the num_entries array
  long rec_id      = threadIdx.x + (blockDim.x * blockIdx.x);
  long rec_id_next = rec_id + 1;

  // we can have more threads than data, make sure we are not past the end of
  // the data
  if (rec_id_next >= row_offsets.size()) return;

  long start = row_offsets[rec_id];
  long stop  = row_offsets[rec_id_next];

  long pos       = start;
  int col        = 0;
  int actual_col = 0;

  while (col < column_flags.size()) {
    if (start > stop) break;

    pos = cudf::io::gpu::seek_field_end(raw_csv + pos, raw_csv + stop, options) - raw_csv;

    if (column_flags[col] & column_parse::enabled) {
      // check if the entire field is a NaN string - consistent with pandas
      const bool is_na = serializedTrieContains(options.naValuesTrie, raw_csv + start, pos - start);

      // Modify start & end to ignore whitespace and quotechars
      long tempPos = pos - 1;
      if (!is_na && dtypes[actual_col].id() != cudf::type_id::STRING) {
        trim_field_start_end(raw_csv, &start, &tempPos, options.quotechar);
      }

      if (!is_na && start <= (tempPos)) {  // Empty fields are not legal values

        // Type dispatcher does not handle STRING
        if (dtypes[actual_col].id() == cudf::type_id::STRING) {
          long end = pos;
          if (options.keepquotes == false) {
            if ((raw_csv[start] == options.quotechar) && (raw_csv[end - 1] == options.quotechar)) {
              start++;
              end--;
            }
          }
          auto str_list = static_cast<std::pair<const char *, size_t> *>(columns[actual_col]);
          str_list[rec_id].first  = raw_csv + start;
          str_list[rec_id].second = end - start;
        } else {
          if (cudf::type_dispatcher(dtypes[actual_col],
                                    decode_op{},
                                    columns[actual_col],
                                    rec_id,
                                    raw_csv + start,
                                    raw_csv + tempPos,
                                    options,
                                    column_flags[col])) {
            // set the valid bitmap - all bits were set to 0 to start
            set_bit(valids[actual_col], rec_id);
          }
        }
      } else if (dtypes[actual_col].id() == cudf::type_id::STRING) {
        auto str_list = static_cast<std::pair<const char *, size_t> *>(columns[actual_col]);
        str_list[rec_id].first  = nullptr;
        str_list[rec_id].second = 0;
      }
      actual_col++;
    }
    pos++;
    start = pos;
    col++;
  }
}

/*
 * @brief Merge two packed row contexts (each corresponding to a block of characters)
 * and return the packed row context corresponding to the merged character block
 **/
inline __device__ packed_rowctx_t merge_row_contexts(packed_rowctx_t first_ctx,
                                                     packed_rowctx_t second_ctx)
{
  uint32_t id0 = get_row_context(first_ctx, ROW_CTX_NONE) & 3;
  uint32_t id1 = get_row_context(first_ctx, ROW_CTX_QUOTE) & 3;
  uint32_t id2 = get_row_context(first_ctx, ROW_CTX_COMMENT) & 3;
  return (first_ctx & ~pack_row_contexts(3, 3, 3)) +
         pack_row_contexts(get_row_context(second_ctx, id0),
                           get_row_context(second_ctx, id1),
                           get_row_context(second_ctx, id2));
}

/*
 * @brief Per-character context:
 * 1-bit count (0 or 1) per context in the lower 4 bits
 * 2-bit output context id per input context in bits 8..15
 **/
constexpr __device__ uint32_t make_char_context(uint32_t id0,
                                                uint32_t id1,
                                                uint32_t id2 = ROW_CTX_COMMENT,
                                                uint32_t c0  = 0,
                                                uint32_t c1  = 0,
                                                uint32_t c2  = 0)
{
  return (id0 << 8) | (id1 << 10) | (id2 << 12) | (ROW_CTX_EOF << 14) | (c0) | (c1 << 1) |
         (c2 << 2);
}

/*
 * @brief Merge a 1-character context to keep track of bitmasks where new rows occur
 * Merges a single-character "block" row context at position pos with the current
 * block's row context (the current block contains 32-pos characters)
 *
 * @param ctx Current block context and new rows bitmaps
 * @param char_ctx state transitions associated with new character
 * @param pos Position within the current 32-character block
 *
 * NOTE: This is probably the most performance-critical piece of the row gathering kernel.
 * The char_ctx value should be created via make_char_context, and its value should
 * have been evaluated at compile-time.
 *
 **/
inline __device__ void merge_char_context(uint4 &ctx, uint32_t char_ctx, uint32_t pos)
{
  uint32_t id0 = (ctx.w >> 0) & 3;
  uint32_t id1 = (ctx.w >> 2) & 3;
  uint32_t id2 = (ctx.w >> 4) & 3;
  // Set the newrow bit in the bitmap at the corresponding position
  ctx.x |= ((char_ctx >> id0) & 1) << pos;
  ctx.y |= ((char_ctx >> id1) & 1) << pos;
  ctx.z |= ((char_ctx >> id2) & 1) << pos;
  // Update the output context ids
  ctx.w = ((char_ctx >> (8 + id0 * 2)) & 0x03) | ((char_ctx >> (6 + id1 * 2)) & 0x0c) |
          ((char_ctx >> (4 + id2 * 2)) & 0x30) | (ROW_CTX_EOF << 6);
}

/*
 * Convert the context-with-row-bitmaps version to a packed row context
 **/
inline __device__ packed_rowctx_t pack_rowmaps(uint4 ctx_map)
{
  return pack_row_contexts(make_row_context(__popc(ctx_map.x), (ctx_map.w >> 0) & 3),
                           make_row_context(__popc(ctx_map.y), (ctx_map.w >> 2) & 3),
                           make_row_context(__popc(ctx_map.z), (ctx_map.w >> 4) & 3));
}

/*
 * Selects the row bitmap corresponding to the given parser state
 **/
inline __device__ uint32_t select_rowmap(uint4 ctx_map, uint32_t ctxid)
{
  return (ctxid == ROW_CTX_NONE)
           ? ctx_map.x
           : (ctxid == ROW_CTX_QUOTE) ? ctx_map.y : (ctxid == ROW_CTX_COMMENT) ? ctx_map.z : 0;
}

/**
 * @brief Single pair-wise 512-wide row context merge transform
 *
 * Merge row context blocks and record the merge operation in a context
 * tree so that the transform is reversible.
 * The tree is organized such that the left and right children of node n
 * are located at indices n*2 and n*2+1, the root node starting at index 1
 *
 * @tparam lanemask mask to specify source of packed row context
 * @tparam tmask mask to specify principle thread for merging row context
 * @tparam base start location for writing into packed row context tree
 * @tparam level_scale level of the node in the tree
 * @param ctxtree[out] packed row context tree
 * @param ctxb[in] packed row context for the current character block
 * @param t thread id (leaf node id)
 *
 */
template <uint32_t lanemask, uint32_t tmask, uint32_t base, uint32_t level_scale>
inline __device__ void ctx_merge(uint64_t *ctxtree, packed_rowctx_t *ctxb, uint32_t t)
{
  uint64_t tmp = SHFL_XOR(*ctxb, lanemask);
  if (!(t & tmask)) {
    *ctxb                              = merge_row_contexts(*ctxb, tmp);
    ctxtree[base + (t >> level_scale)] = *ctxb;
  }
}

/**
 * @brief Single 512-wide row context inverse merge transform
 *
 * Walks the context tree starting from a root node
 *
 * @tparam rmask Mask to specify which threads write input row context
 * @param[in] base Start read location of the merge transform tree
 * @param[in] ctxtree Merge transform tree
 * @param[in] ctx Input context
 * @param[in] brow4 output row in block *4
 * @param[in] t thread id (leaf node id)
 */
template <uint32_t rmask>
inline __device__ void ctx_unmerge(
  uint32_t base, uint64_t *ctxtree, uint32_t *ctx, uint32_t *brow4, uint32_t t)
{
  rowctx32_t ctxb_left, ctxb_right, ctxb_sum;
  ctxb_sum   = get_row_context(ctxtree[base], *ctx);
  ctxb_left  = get_row_context(ctxtree[(base)*2 + 0], *ctx);
  ctxb_right = get_row_context(ctxtree[(base)*2 + 1], ctxb_left & 3);
  if (t & (rmask)) {
    *brow4 += (ctxb_sum & ~3) - (ctxb_right & ~3);
    *ctx = ctxb_left & 3;
  }
}

/*
 * @brief 512-wide row context merge transform
 *
 * Repeatedly merge row context blocks, keeping track of each merge operation
 * in a context tree so that the transform is reversible
 * The tree is organized such that the left and right children of node n
 * are located at indices n*2 and n*2+1, the root node starting at index 1
 *
 * Each node contains the counts and output contexts corresponding to the
 * possible input contexts.
 * Each parent node's count is obtained by adding the corresponding counts
 * from the left child node with the right child node's count selected from
 * the left child node's output context:
 *   parent.count[k] = left.count[k] + right.count[left.outctx[k]]
 *   parent.outctx[k] = right.outctx[left.outctx[k]]
 *
 * @param ctxtree[out] packed row context tree
 * @param ctxb[in] packed row context for the current character block
 * @param t thread id (leaf node id)
 *
 **/
static inline __device__ void rowctx_merge_transform(uint64_t ctxtree[1024],
                                                     packed_rowctx_t ctxb,
                                                     uint32_t t)
{
  ctxtree[512 + t] = ctxb;
  ctx_merge<1, 0x1, 256, 1>(ctxtree, &ctxb, t);
  ctx_merge<2, 0x3, 128, 2>(ctxtree, &ctxb, t);
  ctx_merge<4, 0x7, 64, 3>(ctxtree, &ctxb, t);
  ctx_merge<8, 0xf, 32, 4>(ctxtree, &ctxb, t);
  __syncthreads();
  if (t < 32) {
    ctxb = ctxtree[32 + t];
    ctx_merge<1, 0x1, 16, 1>(ctxtree, &ctxb, t);
    ctx_merge<2, 0x3, 8, 2>(ctxtree, &ctxb, t);
    ctx_merge<4, 0x7, 4, 3>(ctxtree, &ctxb, t);
    ctx_merge<8, 0xf, 2, 4>(ctxtree, &ctxb, t);
    // Final stage
    uint64_t tmp = SHFL_XOR(ctxb, 16);
    if (t == 0) { ctxtree[1] = merge_row_contexts(ctxb, tmp); }
  }
}

/*
 * @brief 512-wide row context inverse merge transform
 *
 * Walks the context tree starting from the root node (index 1) using
 * the starting context in node index 0.
 * The return value is the starting row and input context for the given leaf node
 *
 * @param[in] ctxtree Merge transform tree
 * @param[in] t thread id (leaf node id)
 *
 * @return Final row context and count (row_position*4 + context_id format)
 **/
static inline __device__ rowctx32_t rowctx_inverse_merge_transform(uint64_t ctxtree[1024],
                                                                   uint32_t t)
{
  uint32_t ctx     = ctxtree[0] & 3;  // Starting input context
  rowctx32_t brow4 = 0;               // output row in block *4

  ctx_unmerge<256>(1, ctxtree, &ctx, &brow4, t);
  ctx_unmerge<128>(2 + (t >> 8), ctxtree, &ctx, &brow4, t);
  ctx_unmerge<64>(4 + (t >> 7), ctxtree, &ctx, &brow4, t);
  ctx_unmerge<32>(8 + (t >> 6), ctxtree, &ctx, &brow4, t);
  ctx_unmerge<16>(16 + (t >> 5), ctxtree, &ctx, &brow4, t);
  ctx_unmerge<8>(32 + (t >> 4), ctxtree, &ctx, &brow4, t);
  ctx_unmerge<4>(64 + (t >> 3), ctxtree, &ctx, &brow4, t);
  ctx_unmerge<2>(128 + (t >> 2), ctxtree, &ctx, &brow4, t);
  ctx_unmerge<1>(256 + (t >> 1), ctxtree, &ctx, &brow4, t);

  return brow4 + ctx;
}

/**
 * @brief Gather row offsets from CSV character data split into 16KB chunks
 *
 * This is done in two phases: the first phase returns the possible row counts
 * per 16K character block for each possible parsing context at the start of the block,
 * along with the resulting parsing context at the end of the block.
 * The caller can then compute the actual parsing context at the beginning of each
 * individual block and total row count.
 * The second phase outputs the location of each row in the block, using the parsing
 * context and initial row counter accumulated from the results of the previous phase.
 * Row parsing context will be updated after phase 2 such that the value contains
 * the number of rows starting at byte_range_end or beyond.
 *
 * @param row_ctx Row parsing context (output of phase 1 or input to phase 2)
 * @param offsets_out Row offsets (nullptr for phase1, non-null indicates phase 2)
 * @param data Base pointer of character data (all row offsets are relative to this)
 * @param chunk_size Total number of characters to parse
 * @param parse_pos Current parsing position in the file
 * @param start_offset Position of the start of the character buffer in the file
 * @param data_size CSV file size
 * @param byte_range_start Ignore rows starting before this position in the file
 * @param byte_range_end In phase 2, store the number of rows beyond range in row_ctx
 * @param skip_rows Number of rows to skip (ignored in phase 1)
 * @param terminator Line terminator character
 * @param delimiter Column delimiter character
 * @param quotechar Quote character
 * @param escapechar Delimiter escape character
 * @param commentchar Comment line character (skip rows starting with this character)
 **/
__global__ void __launch_bounds__(rowofs_block_dim)
  gather_row_offsets_gpu(uint64_t *row_ctx,
                         device_span<uint64_t> offsets_out,
                         device_span<char const> const data,
                         size_t chunk_size,
                         size_t parse_pos,
                         size_t start_offset,
                         size_t data_size,
                         size_t byte_range_start,
                         size_t byte_range_end,
                         size_t skip_rows,
                         int terminator,
                         int delimiter,
                         int quotechar,
                         int escapechar,
                         int commentchar)
{
  auto start = data.begin();
  __shared__ __align__(8) uint64_t ctxtree[rowofs_block_dim * 2];
  using warp_reduce      = typename cub::WarpReduce<uint32_t>;
  using half_warp_reduce = typename cub::WarpReduce<uint32_t, 16>;
  __shared__ union {
    typename warp_reduce::TempStorage full;
    typename half_warp_reduce::TempStorage half[rowofs_block_dim / 32];
  } temp_storage;

  const char *end = start + (min(parse_pos + chunk_size, data_size) - start_offset);
  uint32_t t      = threadIdx.x;
  size_t block_pos =
    (parse_pos - start_offset) + blockIdx.x * static_cast<size_t>(rowofs_block_bytes) + t * 32;
  const char *cur = start + block_pos;

  // Initial state is neutral context (no state transitions), zero rows
  uint4 ctx_map = {
    .x = 0,
    .y = 0,
    .z = 0,
    .w = (ROW_CTX_NONE << 0) | (ROW_CTX_QUOTE << 2) | (ROW_CTX_COMMENT << 4) | (ROW_CTX_EOF << 6)};
  int c, c_prev = (cur > start && cur <= end) ? cur[-1] : terminator;
  // Loop through all 32 bytes and keep a bitmask of row starts for each possible input context
  for (uint32_t pos = 0; pos < 32; pos++, cur++, c_prev = c) {
    uint32_t ctx;
    if (cur < end) {
      c = cur[0];
      if (c_prev == terminator) {
        if (c == commentchar) {
          // Start of a new comment row
          ctx = make_char_context(ROW_CTX_COMMENT, ROW_CTX_QUOTE, ROW_CTX_COMMENT, 1, 0, 1);
        } else if (c == quotechar) {
          // Quoted string on newrow, or quoted string ending in terminator
          ctx = make_char_context(ROW_CTX_QUOTE, ROW_CTX_NONE, ROW_CTX_QUOTE, 1, 0, 1);
        } else {
          // Start of a new row unless within a quote
          ctx = make_char_context(ROW_CTX_NONE, ROW_CTX_QUOTE, ROW_CTX_NONE, 1, 0, 1);
        }
      } else if (c == quotechar) {
        if (c_prev == delimiter || c_prev == quotechar) {
          // Quoted string after delimiter, quoted string ending in delimiter, or double-quote
          ctx = make_char_context(ROW_CTX_QUOTE, ROW_CTX_NONE);
        } else {
          // Closing or ignored quote
          ctx = make_char_context(ROW_CTX_NONE, ROW_CTX_NONE);
        }
      } else {
        // Neutral character
        ctx = make_char_context(ROW_CTX_NONE, ROW_CTX_QUOTE);
      }
    } else {
      const char *data_end = start + data_size - start_offset;
      if (cur <= end && cur == data_end) {
        // Add a newline at data end (need the extra row offset to infer length of previous row)
        ctx = make_char_context(ROW_CTX_EOF, ROW_CTX_EOF, ROW_CTX_EOF, 1, 1, 1);
      } else {
        // Pass-through context (beyond chunk_size or data_end)
        ctx = make_char_context(ROW_CTX_NONE, ROW_CTX_QUOTE, ROW_CTX_COMMENT);
      }
    }
    // Merge with current context, keeping track of where new rows occur
    merge_char_context(ctx_map, ctx, pos);
  }

  // Eliminate rows that start before byte_range_start
  if (start_offset + block_pos < byte_range_start) {
    uint32_t dist_minus1 = min(byte_range_start - (start_offset + block_pos) - 1, UINT64_C(31));
    uint32_t mask        = 0xfffffffe << dist_minus1;
    ctx_map.x &= mask;
    ctx_map.y &= mask;
    ctx_map.z &= mask;
  }

  // Convert the long-form {rowmap,outctx}[inctx] version into packed version
  // {rowcount,ouctx}[inctx], then merge the row contexts of the 32-character blocks into
  // a single 16K-character block context
  rowctx_merge_transform(ctxtree, pack_rowmaps(ctx_map), t);

  // If this is the second phase, get the block's initial parser state and row counter
  if (offsets_out.data()) {
    if (t == 0) { ctxtree[0] = row_ctx[blockIdx.x]; }
    __syncthreads();

    // Walk back the transform tree with the known initial parser state
    rowctx32_t ctx             = rowctx_inverse_merge_transform(ctxtree, t);
    uint64_t row               = (ctxtree[0] >> 2) + (ctx >> 2);
    uint32_t rows_out_of_range = 0;
    uint32_t rowmap            = select_rowmap(ctx_map, ctx & 3);
    // Output row positions
    while (rowmap != 0) {
      uint32_t pos = __ffs(rowmap);
      block_pos += pos;
      if (row >= skip_rows && row - skip_rows < offsets_out.size()) {
        // Output byte offsets are relative to the base of the input buffer
        offsets_out[row - skip_rows] = block_pos - 1;
        rows_out_of_range += (start_offset + block_pos - 1 >= byte_range_end);
      }
      row++;
      rowmap >>= pos;
    }
    // Return the number of rows out of range
    rows_out_of_range = half_warp_reduce(temp_storage.half[t / 32]).Sum(rows_out_of_range);
    __syncthreads();
    if (!(t & 0xf)) { ctxtree[t >> 4] = rows_out_of_range; }
    __syncthreads();
    if (t < 32) {
      rows_out_of_range = warp_reduce(temp_storage.full).Sum(static_cast<uint32_t>(ctxtree[t]));
      if (t == 0) { row_ctx[blockIdx.x] = rows_out_of_range; }
    }
  } else {
    // Just store the row counts and output contexts
    if (t == 0) { row_ctx[blockIdx.x] = ctxtree[1]; }
  }
}

size_t __host__ count_blank_rows(const cudf::io::ParseOptions &opts,
                                 device_span<char const> const data,
                                 device_span<uint64_t const> const row_offsets,
                                 cudaStream_t stream)
{
  const auto newline  = opts.skipblanklines ? opts.terminator : opts.comment;
  const auto comment  = opts.comment != '\0' ? opts.comment : newline;
  const auto carriage = (opts.skipblanklines && opts.terminator == '\n') ? '\r' : comment;
  return thrust::count_if(
    rmm::exec_policy(stream)->on(stream),
    row_offsets.begin(),
    row_offsets.end(),
    [data = data, newline, comment, carriage] __device__(const uint64_t pos) {
      return ((pos != data.size()) &&
              (data[pos] == newline || data[pos] == comment || data[pos] == carriage));
    });
}

void __host__ remove_blank_rows(cudf::io::ParseOptions const &options,
                                device_span<char const> const data,
                                rmm::device_vector<uint64_t> &row_offsets,
                                cudaStream_t stream)
{
  size_t d_size       = data.size();
  const auto newline  = options.skipblanklines ? options.terminator : options.comment;
  const auto comment  = options.comment != '\0' ? options.comment : newline;
  const auto carriage = (options.skipblanklines && options.terminator == '\n') ? '\r' : comment;
  auto new_end        = thrust::remove_if(
    rmm::exec_policy(stream)->on(stream),
    row_offsets.begin(),
    row_offsets.end(),
    [data = data, d_size, newline, comment, carriage] __device__(const uint64_t pos) {
      return ((pos != d_size) &&
              (data[pos] == newline || data[pos] == comment || data[pos] == carriage));
    });
  row_offsets.resize(new_end - row_offsets.begin());
}

thrust::host_vector<column_parse::stats> detect_column_types(
  cudf::io::ParseOptions const &options,
  device_span<char const> const data,
  device_span<column_parse::flags const> const column_flags,
  device_span<uint64_t const> const row_starts,
  size_t const num_active_columns,
  cudaStream_t stream)
{
  // Calculate actual block count to use based on records count
  const int block_size = csvparse_block_dim;
  const int grid_size  = (row_starts.size() + block_size - 1) / block_size;

  auto d_stats = rmm::device_vector<column_parse::stats>(num_active_columns);

  data_type_detection<<<grid_size, block_size, 0, stream>>>(
    options, data, column_flags, row_starts, d_stats);

  return thrust::host_vector<column_parse::stats>(d_stats);
}

void __host__ decode_row_column_data(cudf::io::ParseOptions const &options,
                                     device_span<char const> const data,
                                     device_span<column_parse::flags const> const column_flags,
                                     device_span<uint64_t const> const row_offsets,
                                     device_span<cudf::data_type const> const dtypes,
                                     device_span<void *> const columns,
                                     device_span<cudf::bitmask_type *> const valids,
                                     cudaStream_t stream)
{
  // Calculate actual block count to use based on records count
  auto const block_size = csvparse_block_dim;
  auto const num_rows   = row_offsets.size() - 1;
  auto const grid_size  = (num_rows + block_size - 1) / block_size;

  convert_csv_to_cudf<<<grid_size, block_size, 0, stream>>>(
    options, data, column_flags, row_offsets, dtypes, columns, valids);
}

uint32_t __host__ gather_row_offsets(const ParseOptions &options,
                                     uint64_t *row_ctx,
                                     device_span<uint64_t> const offsets_out,
                                     device_span<char const> const data,
                                     size_t chunk_size,
                                     size_t parse_pos,
                                     size_t start_offset,
                                     size_t data_size,
                                     size_t byte_range_start,
                                     size_t byte_range_end,
                                     size_t skip_rows,
                                     cudaStream_t stream)
{
  uint32_t dim_grid = 1 + (chunk_size / rowofs_block_bytes);

  gather_row_offsets_gpu<<<dim_grid, rowofs_block_dim, 0, stream>>>(
    row_ctx,
    offsets_out,
    data,
    chunk_size,
    parse_pos,
    start_offset,
    data_size,
    byte_range_start,
    byte_range_end,
    skip_rows,
    options.terminator,
    options.delimiter,
    (options.quotechar) ? options.quotechar : 0x100,
    /*(options.escapechar) ? options.escapechar :*/ 0x100,
    (options.comment) ? options.comment : 0x100);

  return dim_grid;
}

}  // namespace gpu
}  // namespace csv
}  // namespace io
}  // namespace cudf
