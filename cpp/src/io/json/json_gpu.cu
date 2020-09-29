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

#include <io/csv/datetime.cuh>
#include <io/utilities/parsing_utils.cuh>

#include <cudf/detail/utilities/hash_functions.cuh>
#include <cudf/detail/utilities/trie.cuh>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/lists/list_view.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>

#include <thrust/detail/copy.h>
#include <thrust/find.h>

using cudf::detail::device_span;

namespace cudf {
namespace io {
namespace json {
namespace gpu {
using namespace ::cudf;

using string_pair = std::pair<const char *, size_t>;

namespace {
/**
 * @brief CUDA Kernel that adjusts the row range to exclude the character outside of the top level
 * brackets.
 *
 * The top level brackets characters are excluded from the resulting range.
 *
 * @param[in] begin Pointer to the first character in the row
 * @param[in] end pointer to the first character after the row
 */
__device__ std::pair<char const *, char const *> limit_range_to_brackets(char const *begin,
                                                                         char const *end)
{
  begin = thrust::find_if(
    thrust::seq, begin, end, [] __device__(auto c) { return c == '[' || c == '{'; });
  end = thrust::find_if(thrust::seq,
                        thrust::make_reverse_iterator(end),
                        thrust::make_reverse_iterator(++begin),
                        [](auto c) { return c == ']' || c == '}'; })
          .base();
  return {begin, --end};
}

/**
 * @brief Find the first JSON object key in the range.
 *
 * Assumes that begin is not in the middle of a field.
 *
 * @param[in] begin Pointer to the first character in the parsing range
 * @param[in] end pointer to the first character after the parsing range
 * @param[in] quotechar The character used to denote quotes
 *
 * @return Begin and end iterators of the key name; (`end`, `end`) if a key is not found
 */
__device__ std::pair<char const *, char const *> get_next_key(char const *begin,
                                                              char const *end,
                                                              char quotechar)
{
  // Key starts after the first quote
  auto const key_begin = thrust::find(thrust::seq, begin, end, quotechar) + 1;
  if (key_begin > end) return {end, end};

  // Key ends after the next unescaped quote
  auto const key_end_pair = thrust::mismatch(
    thrust::seq, key_begin, end - 1, key_begin + 1, [quotechar] __device__(auto prev_ch, auto ch) {
      return !(ch == quotechar && prev_ch != '\\');
    });

  return {key_begin, key_end_pair.second};
}

/**
 * @brief Decodes a numeric value base on templated cudf type T with specified
 * base.
 *
 * @param[in] begin Beginning of the character string
 * @param[in] end End of the character string
 * @param opts The global parsing behavior options
 *
 * @return The parsed numeric value
 */
template <typename T, int base>
__inline__ __device__ T decode_value(const char *begin, uint64_t end, ParseOptions const &opts)
{
  return cudf::io::gpu::parse_numeric<T, base>(begin, end, opts);
}

/**
 * @brief Decodes a numeric value base on templated cudf type T
 *
 * @param[in] begin Beginning of the character string
 * @param[in] end End of the character string
 * @param opts The global parsing behavior options
 *
 * @return The parsed numeric value
 */
template <typename T>
__inline__ __device__ T decode_value(const char *begin, const char *end, ParseOptions const &opts)
{
  return cudf::io::gpu::parse_numeric<T>(begin, end, opts);
}

/**
 * @brief Decodes a timestamp_D
 *
 * @param[in] begin Beginning of the character string
 * @param[in] end End of the character string
 * @param opts The global parsing behavior options
 *
 * @return The parsed timestamp_D
 */
template <>
__inline__ __device__ cudf::timestamp_D decode_value(const char *begin,
                                                     const char *end,
                                                     ParseOptions const &opts)
{
  return cudf::timestamp_D{cudf::duration_D{parseDateFormat(begin, end, opts.dayfirst)}};
}

/**
 * @brief Decodes a timestamp_s
 *
 * @param[in] begin Beginning of the character string
 * @param[in] end End of the character string
 * @param opts The global parsing behavior options
 *
 * @return The parsed timestamp_s
 */
template <>
__inline__ __device__ cudf::timestamp_s decode_value(const char *begin,
                                                     const char *end,
                                                     ParseOptions const &opts)
{
  auto milli = parseDateTimeFormat(begin, end, opts.dayfirst);
  return cudf::timestamp_s{cudf::duration_s{milli / 1000}};
}

/**
 * @brief Decodes a timestamp_ms
 *
 * @param[in] begin Beginning of the character string
 * @param[in] end End of the character string
 * @param opts The global parsing behavior options
 *
 * @return The parsed timestamp_ms
 */
template <>
__inline__ __device__ cudf::timestamp_ms decode_value(const char *begin,
                                                      const char *end,
                                                      ParseOptions const &opts)
{
  auto milli = parseDateTimeFormat(begin, end, opts.dayfirst);
  return cudf::timestamp_ms{cudf::duration_ms{milli}};
}

/**
 * @brief Decodes a timestamp_us
 *
 * @param[in] begin Beginning of the character string
 * @param[in] end End of the character string
 * @param opts The global parsing behavior options
 *
 * @return The parsed timestamp_us
 */
template <>
__inline__ __device__ cudf::timestamp_us decode_value(const char *begin,
                                                      const char *end,
                                                      ParseOptions const &opts)
{
  auto milli = parseDateTimeFormat(begin, end, opts.dayfirst);
  return cudf::timestamp_us{cudf::duration_us{milli * 1000}};
}

/**
 * @brief Decodes a timestamp_ns
 *
 * @param[in] begin Beginning of the character string
 * @param[in] end End of the character string
 * @param opts The global parsing behavior options
 *
 * @return The parsed timestamp_ns
 */
template <>
__inline__ __device__ cudf::timestamp_ns decode_value(const char *begin,
                                                      const char *end,
                                                      ParseOptions const &opts)
{
  auto milli = parseDateTimeFormat(begin, end, opts.dayfirst);
  return cudf::timestamp_ns{cudf::duration_ns{milli * 1000000}};
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

// The purpose of these is merely to allow compilation ONLY
template <>
__inline__ __device__ cudf::string_view decode_value(const char *begin,
                                                     const char *end,
                                                     ParseOptions const &opts)
{
  return cudf::string_view{};
}

template <>
__inline__ __device__ cudf::dictionary32 decode_value(const char *begin,
                                                      const char *end,
                                                      ParseOptions const &opts)
{
  return cudf::dictionary32{};
}

template <>
__inline__ __device__ cudf::list_view decode_value(const char *begin,
                                                   const char *end,
                                                   ParseOptions const &opts)
{
  return cudf::list_view{};
}
template <>
__inline__ __device__ cudf::struct_view decode_value(const char *begin,
                                                     const char *end,
                                                     ParseOptions const &opts)
{
  return cudf::struct_view{};
}

template <>
__inline__ __device__ numeric::decimal32 decode_value(const char *begin,
                                                      const char *end,
                                                      ParseOptions const &opts)
{
  return numeric::decimal32{};
}

template <>
__inline__ __device__ numeric::decimal64 decode_value(const char *begin,
                                                      const char *end,
                                                      ParseOptions const &opts)
{
  return numeric::decimal64{};
}

/**
 * @brief Functor for converting plain text data to cuDF data type value.
 */
struct ConvertFunctor {
  /**
   * @brief Template specialization for operator() for types whose values can be
   * convertible to a 0 or 1 to represent false/true. The converting is done by
   * checking against the default and user-specified true/false values list.
   *
   * It is handled here rather than within convertStrToValue() as that function
   * is used by other types (ex. timestamp) that aren't 'booleable'.
   */
  template <typename T, typename std::enable_if_t<std::is_integral<T>::value> * = nullptr>
  __host__ __device__ __forceinline__ bool operator()(char const *begin,
                                                      char const *end,
                                                      void *output_column,
                                                      cudf::size_type row,
                                                      const ParseOptions &opts)
  {
    T &value{static_cast<T *>(output_column)[row]};

    // Check for user-specified true/false values first, where the output is
    // replaced with 1/0 respectively
    value = [&opts, end, begin]() -> T {
      if (serializedTrieContains(opts.trueValuesTrie, begin, end - begin)) {
        return 1;
      } else if (serializedTrieContains(opts.falseValuesTrie, begin, end - begin)) {
        return 0;
      } else {
        return decode_value<T>(begin, end - 1, opts);
      }
    }();

    return true;
  }

  /**
   * @brief Dispatch for floating points, which are set to NaN if the input
   * is not valid. In such case, the validity mask is set to zero too.
   */
  template <typename T, typename std::enable_if_t<std::is_floating_point<T>::value> * = nullptr>
  __host__ __device__ __forceinline__ bool operator()(
    char const *begin, char const *end, void *out_buffer, size_t row, ParseOptions const &opts)
  {
    auto &value{static_cast<T *>(out_buffer)[row]};
    value = decode_value<T>(begin, end - 1, opts);
    return !std::isnan(value);
  }

  /**
   * @brief Default template operator() dispatch specialization all data types
   * (including wrapper types) that is not covered by above.
   */
  template <typename T,
            typename std::enable_if_t<!std::is_floating_point<T>::value and
                                      !std::is_integral<T>::value> * = nullptr>
  __host__ __device__ __forceinline__ bool operator()(char const *begin,
                                                      char const *end,
                                                      void *output_column,
                                                      cudf::size_type row,
                                                      const ParseOptions &opts)
  {
    T &value{static_cast<T *>(output_column)[row]};
    value = decode_value<T>(begin, end - 1, opts);

    return true;
  }
};

/**
 * @brief Checks whether the given character is a whitespace character.
 *
 * @param[in] ch The character to check
 *
 * @return True if the input is whitespace, False otherwise
 */
__inline__ __device__ bool is_whitespace(char ch) { return ch == '\t' || ch == ' '; }

/**
 * @brief Adjusts the range to ignore starting/trailing whitespace and quotation characters.
 *
 * @param[in] begin Pointer to the first character in the parsing range
 * @param[in] end pointer to the first character after the parsing range
 * @param[in] quotechar The character used to denote quotes; '\0' if none
 *
 * @return Trimmed range
 */
__inline__ __device__ std::pair<char const *, char const *> trim_whitespaces_quotes(
  char const *begin, char const *end, char quotechar = '\0')
{
  auto not_whitespace = [] __device__(auto c) { return !is_whitespace(c); };

  begin = thrust::find_if(thrust::seq, begin, end, not_whitespace);
  end   = thrust::find_if(thrust::seq,
                        thrust::make_reverse_iterator(end),
                        thrust::make_reverse_iterator(begin),
                        not_whitespace)
          .base();

  return {(*begin == quotechar) ? ++begin : begin, (*(end - 1) == quotechar) ? end - 1 : end};
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
 * @brief Contains information on a JSON file field.
 */
struct field_descriptor {
  cudf::size_type column;
  char const *value_begin;
  char const *value_end;
};

/**
 * @brief Parse the first field in the given range and return its descriptor.
 *
 * @param[in] begin Pointer to the first character in the parsing range
 * @param[in] end pointer to the first character after the parsing range
 * @param[in] opts The global parsing behavior options
 * @param[in] field_idx Index of the current field in the input row
 * @param[in] col_map Pointer to the (column name hash -> solumn index) map in device memory.
 * nullptr is passed when the input file does not consist of objects.
 * @return Descriptor of the parsed field
 */
__device__ field_descriptor next_field_descriptor(const char *begin,
                                                  const char *end,
                                                  ParseOptions const &opts,
                                                  cudf::size_type field_idx,
                                                  col_map_type *col_map)
{
  auto const desc_pre_trim =
    col_map == nullptr
      // No key - column and begin are trivial
      ? field_descriptor{field_idx, begin, cudf::io::gpu::seek_field_end(begin, end, opts, true)}
      : [&]() {
          auto const key_range = get_next_key(begin, end, opts.quotechar);
          auto const key_hash  = MurmurHash3_32<cudf::string_view>{}(
            cudf::string_view(key_range.first, key_range.second - key_range.first));
          auto const hash_col = col_map->find(key_hash);
          // Fall back to field index if not found (parsing error)
          auto const column = (hash_col != col_map->end()) ? (*hash_col).second : field_idx;

          // Skip the colon between the key and the value
          auto const value_begin = thrust::find(thrust::seq, key_range.second, end, ':') + 1;
          return field_descriptor{
            column, value_begin, cudf::io::gpu::seek_field_end(value_begin, end, opts, true)};
        }();

  // Modify start & end to ignore whitespace and quotechars
  auto const trimmed_value_range =
    trim_whitespaces_quotes(desc_pre_trim.value_begin, desc_pre_trim.value_end, opts.quotechar);
  return {desc_pre_trim.column, trimmed_value_range.first, trimmed_value_range.second};
}

/**
 * @brief Returns the range that contains the data in a given row.
 *
 * Excludes the top-level brackets.
 *
 * @param[in] data Pointer to the JSON data in device memory
 * @param[in] data_size Size of the data buffer, in bytes
 * @param[in] rec_starts The offset of each row in the input
 * @param[in] num_rows The number of lines/rows
 * @param[in] row Index of the row for which the range is returned
 *
 * @return The begin and end iterators of the row data.
 */
__device__ std::pair<char const *, char const *> get_row_data_range(
  device_span<char const> const data, device_span<uint64_t const> const row_offsets, size_type row)
{
  auto const row_begin = data.begin() + row_offsets[row];
  auto const row_end =
    data.begin() + ((row < row_offsets.size() - 1) ? row_offsets[row + 1] : data.size());
  return limit_range_to_brackets(row_begin, row_end);
}

/**
 * @brief CUDA kernel that parses and converts plain text data into cuDF column data.
 *
 * Data is processed one record at a time
 *
 * @param[in] opts A set of parsing options
 * @param[in] data The entire data to read
 * @param[in] row_offsets The offset of each row in the input
 * @param[in] column_types The data type of each column
 * @param[in] col_map Pointer to the (column name hash -> solumn index) map in device memory.
 * nullptr is passed when the input file does not consist of objects.
 * @param[out] output_columns The output column data
 * @param[out] valid_fields The bitmaps indicating whether column fields are valid
 * @param[out] num_valid_fields The numbers of valid fields in columns
 */
__global__ void convert_data_to_columns_kernel(ParseOptions opts,
                                               device_span<char const> const data,
                                               device_span<uint64_t const> const row_offsets,
                                               device_span<data_type const> const column_types,
                                               col_map_type *col_map,
                                               device_span<void *const> const output_columns,
                                               device_span<bitmask_type *const> const valid_fields,
                                               device_span<cudf::size_type> const num_valid_fields)
{
  const auto rec_id = threadIdx.x + (blockDim.x * blockIdx.x);
  if (rec_id >= row_offsets.size()) return;

  auto const row_data_range = get_row_data_range(data, row_offsets, rec_id);

  auto current = row_data_range.first;
  for (size_type input_field_index = 0;
       input_field_index < column_types.size() && current < row_data_range.second;
       input_field_index++) {
    auto const desc =
      next_field_descriptor(current, row_data_range.second, opts, input_field_index, col_map);
    auto const value_len = desc.value_end - desc.value_begin;

    current = desc.value_end + 1;

    // Empty fields are not legal values
    if (value_len > 0 && !serializedTrieContains(opts.naValuesTrie, desc.value_begin, value_len)) {
      // Type dispatcher does not handle strings
      if (column_types[desc.column].id() == type_id::STRING) {
        auto str_list           = static_cast<string_pair *>(output_columns[desc.column]);
        str_list[rec_id].first  = desc.value_begin;
        str_list[rec_id].second = value_len;

        // set the valid bitmap - all bits were set to 0 to start
        set_bit(valid_fields[desc.column], rec_id);
        atomicAdd(&num_valid_fields[desc.column], 1);
      } else {
        if (cudf::type_dispatcher(column_types[desc.column],
                                  ConvertFunctor{},
                                  desc.value_begin,
                                  desc.value_end,
                                  output_columns[desc.column],
                                  rec_id,
                                  opts)) {
          // set the valid bitmap - all bits were set to 0 to start
          set_bit(valid_fields[desc.column], rec_id);
          atomicAdd(&num_valid_fields[desc.column], 1);
        }
      }
    } else if (column_types[desc.column].id() == type_id::STRING) {
      auto str_list           = static_cast<string_pair *>(output_columns[desc.column]);
      str_list[rec_id].first  = nullptr;
      str_list[rec_id].second = 0;
    }
  }
}

/**
 * @brief CUDA kernel that processes a buffer of data and determines information about the
 * column types within.
 *
 * Data is processed in one row/record at a time, so the number of total
 * threads (tid) is equal to the number of rows.
 *
 * @param[in] opts A set of parsing options
 * @param[in] data Input data buffer
 * @param[in] rec_starts The offset of each row in the input
 * @param[in] col_map Pointer to the (column name hash -> column index) map in device memory.
 * nullptr is passed when the input file does not consist of objects.
 * @param[in] num_columns The number of columns of input data
 * @param[out] column_infos The count for each column data type
 */
__global__ void detect_data_types_kernel(ParseOptions const opts,
                                         device_span<char const> const data,
                                         device_span<uint64_t const> const row_offsets,
                                         col_map_type *col_map,
                                         int num_columns,
                                         device_span<column_info> const column_infos)
{
  auto const rec_id = threadIdx.x + (blockDim.x * blockIdx.x);
  if (rec_id >= row_offsets.size()) return;

  auto const are_rows_objects = col_map != nullptr;
  auto const row_data_range   = get_row_data_range(data, row_offsets, rec_id);

  size_type input_field_index = 0;
  for (auto current = row_data_range.first;
       input_field_index < num_columns && current < row_data_range.second;
       input_field_index++) {
    auto const desc =
      next_field_descriptor(current, row_data_range.second, opts, input_field_index, col_map);
    auto const value_len = desc.value_end - desc.value_begin;

    // Advance to the next field; +1 to skip the delimiter
    current = desc.value_end + 1;

    // Checking if the field is empty/valid
    if (value_len <= 0 || serializedTrieContains(opts.naValuesTrie, desc.value_begin, value_len)) {
      // Increase the null count for array rows, where the null count is initialized to zero.
      if (!are_rows_objects) { atomicAdd(&column_infos[desc.column].null_count, 1); }
      continue;
    } else if (are_rows_objects) {
      // For files with object rows, null count is initialized to row count. The value is decreased
      // here for every valid field.
      atomicAdd(&column_infos[desc.column].null_count, -1);
    }
    // Don't need counts to detect strings, any field in quotes is deduced to be a string
    if (*(desc.value_begin - 1) == opts.quotechar && *desc.value_end == opts.quotechar) {
      atomicAdd(&column_infos[desc.column].string_count, 1);
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
      ((value_len > 2 && *desc.value_begin == '0' && *(desc.value_begin + 1) == 'x') ||
       (value_len > 3 && *desc.value_begin == '-' && *(desc.value_begin + 1) == '0' &&
        *(desc.value_begin + 2) == 'x'));
    for (auto pos = desc.value_begin; pos < desc.value_end; ++pos) {
      if (is_digit(*pos, maybe_hex)) {
        digit_count++;
        continue;
      }
      // Looking for unique characters that will help identify column types
      switch (*pos) {
        case '.': decimal_count++; break;
        case '-': dash_count++; break;
        case '/': slash_count++; break;
        case ':': colon_count++; break;
        case 'e':
        case 'E':
          if (!maybe_hex && pos > desc.value_begin && pos < desc.value_end - 1) exponent_count++;
          break;
        default: other_count++; break;
      }
    }

    // Integers have to have the length of the string
    int int_req_number_cnt = value_len;
    // Off by one if they start with a minus sign
    if (*desc.value_begin == '-' && value_len > 1) { --int_req_number_cnt; }
    // Off by one if they are a hexadecimal number
    if (maybe_hex) { --int_req_number_cnt; }
    if (serializedTrieContains(opts.trueValuesTrie, desc.value_begin, value_len) ||
        serializedTrieContains(opts.falseValuesTrie, desc.value_begin, value_len)) {
      atomicAdd(&column_infos[desc.column].bool_count, 1);
    } else if (digit_count == int_req_number_cnt) {
      atomicAdd(&column_infos[desc.column].int_count, 1);
    } else if (is_like_float(value_len, digit_count, decimal_count, dash_count, exponent_count)) {
      atomicAdd(&column_infos[desc.column].float_count, 1);
    }
    // A date-time field cannot have more than 3 non-special characters
    // A number field cannot have more than one decimal point
    else if (other_count > 3 || decimal_count > 1) {
      atomicAdd(&column_infos[desc.column].string_count, 1);
    } else {
      // A date field can have either one or two '-' or '\'; A legal combination will only have one
      // of them To simplify the process of auto column detection, we are not covering all the
      // date-time formation permutations
      if ((dash_count > 0 && dash_count <= 2 && slash_count == 0) ||
          (dash_count == 0 && slash_count > 0 && slash_count <= 2)) {
        if (colon_count <= 2) {
          atomicAdd(&column_infos[desc.column].datetime_count, 1);
        } else {
          atomicAdd(&column_infos[desc.column].string_count, 1);
        }
      } else {
        // Default field type is string
        atomicAdd(&column_infos[desc.column].string_count, 1);
      }
    }
  }
  if (!are_rows_objects) {
    // For array rows, mark missing fields as null
    for (; input_field_index < num_columns; ++input_field_index)
      atomicAdd(&column_infos[input_field_index].null_count, 1);
  }
}

/**
 * @brief Input data range that contains a field in key:value format.
 */
struct key_value_range {
  char const *key_begin;
  char const *key_end;
  char const *value_begin;
  char const *value_end;
};

/**
 * @brief Parse the next field in key:value format and return ranges of its parts.
 */
__device__ key_value_range get_next_key_value_range(char const *begin,
                                                    char const *end,
                                                    ParseOptions const &opts)
{
  auto const key_range = get_next_key(begin, end, opts.quotechar);

  // Colon between the key and the value
  auto const colon = thrust::find(thrust::seq, key_range.second, end, ':');
  if (colon == end) return {end, end, end};

  // Field value (including delimiters)
  auto const value_end = cudf::io::gpu::seek_field_end(colon + 1, end, opts, true);
  return {key_range.first, key_range.second, colon + 1, value_end};
}

/**
 * @brief Cuda kernel that collects information about JSON object keys in the file.
 *
 * @param[in] options A set of parsing options
 * @param[in] data Input data buffer
 * @param[in] row_offsets The offset of each row in the input
 * @param[out] keys_cnt Number of keys found in the file
 * @param[out] keys_info optional, information (offset, length, hash) for each found key
 *
 */
__global__ void collect_keys_info_kernel(ParseOptions const options,
                                         device_span<char const> const data,
                                         device_span<uint64_t const> const row_offsets,
                                         unsigned long long int *keys_cnt,
                                         thrust::optional<mutable_table_device_view> keys_info)
{
  auto const rec_id = threadIdx.x + (blockDim.x * blockIdx.x);
  if (rec_id >= row_offsets.size()) return;

  auto const row_data_range = get_row_data_range(data, row_offsets, rec_id);

  auto advance = [&](const char *begin) {
    return get_next_key_value_range(begin, row_data_range.second, options);
  };
  for (auto field_range = advance(row_data_range.first);
       field_range.key_begin < row_data_range.second;
       field_range = advance(field_range.value_end)) {
    auto const idx = atomicAdd(keys_cnt, 1);
    if (keys_info.has_value()) {
      auto const len                              = field_range.key_end - field_range.key_begin;
      keys_info->column(0).element<uint64_t>(idx) = field_range.key_begin - data.begin();
      keys_info->column(1).element<uint16_t>(idx) = len;
      keys_info->column(2).element<uint32_t>(idx) =
        MurmurHash3_32<cudf::string_view>{}(cudf::string_view(field_range.key_begin, len));
    }
  }
}

}  // namespace

/**
 * @copydoc cudf::io::json::gpu::convert_json_to_columns
 */
void convert_json_to_columns(ParseOptions const &opts,
                             device_span<char const> const data,
                             device_span<uint64_t const> const row_offsets,
                             device_span<data_type const> const column_types,
                             col_map_type *col_map,
                             device_span<void *const> const output_columns,
                             device_span<bitmask_type *const> const valid_fields,
                             device_span<cudf::size_type> num_valid_fields,
                             cudaStream_t stream)
{
  int block_size;
  int min_grid_size;
  CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(
    &min_grid_size, &block_size, convert_data_to_columns_kernel));

  const int grid_size = (row_offsets.size() + block_size - 1) / block_size;

  convert_data_to_columns_kernel<<<grid_size, block_size, 0, stream>>>(
    opts, data, row_offsets, column_types, col_map, output_columns, valid_fields, num_valid_fields);

  CUDA_TRY(cudaGetLastError());
}

/**
 * @copydoc cudf::io::json::gpu::detect_data_types
 */

std::vector<cudf::io::json::column_info> detect_data_types(
  const ParseOptions &options,
  device_span<char const> const data,
  device_span<uint64_t const> const row_offsets,
  bool do_set_null_count,
  int num_columns,
  col_map_type *col_map,
  cudaStream_t stream)
{
  int block_size;
  int min_grid_size;
  CUDA_TRY(
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, detect_data_types_kernel));

  rmm::device_vector<cudf::io::json::column_info> d_column_infos(num_columns,
                                                                 cudf::io::json::column_info{});

  if (do_set_null_count) {
    // Set the null count to the row count (all fields assumes to be null).
    thrust::for_each(
      rmm::exec_policy(stream)->on(stream),
      d_column_infos.begin(),
      d_column_infos.end(),
      [num_records = row_offsets.size()] __device__(auto &info) { info.null_count = num_records; });
  }

  // Calculate actual block count to use based on records count
  const int grid_size = (row_offsets.size() + block_size - 1) / block_size;

  detect_data_types_kernel<<<grid_size, block_size, 0, stream>>>(
    options, data, row_offsets, col_map, num_columns, d_column_infos);

  CUDA_TRY(cudaGetLastError());

  auto h_column_infos = std::vector<cudf::io::json::column_info>(num_columns);

  thrust::copy(d_column_infos.begin(), d_column_infos.end(), h_column_infos.begin());

  return h_column_infos;
}

/**
 * @copydoc cudf::io::json::gpu::gpu_collect_keys_info
 */
void collect_keys_info(ParseOptions const &options,
                       device_span<char const> const data,
                       device_span<uint64_t const> const row_offsets,
                       unsigned long long int *keys_cnt,
                       thrust::optional<mutable_table_device_view> keys_info,
                       cudaStream_t stream)
{
  int block_size;
  int min_grid_size;
  CUDA_TRY(
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, collect_keys_info_kernel));

  // Calculate actual block count to use based on records count
  const int grid_size = (row_offsets.size() + block_size - 1) / block_size;

  collect_keys_info_kernel<<<grid_size, block_size, 0, stream>>>(
    options, data, row_offsets, keys_cnt, keys_info);

  CUDA_TRY(cudaGetLastError());
}

}  // namespace gpu
}  // namespace json
}  // namespace io
}  // namespace cudf
