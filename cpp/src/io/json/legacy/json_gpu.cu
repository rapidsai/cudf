/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include "io/utilities/column_type_histogram.hpp"
#include "io/utilities/parsing_utils.cuh"
#include "io/utilities/trie.cuh"
#include "json_gpu.hpp"

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/optional>
#include <thrust/advance.h>
#include <thrust/detail/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/generate.h>
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/mismatch.h>
#include <thrust/pair.h>

using cudf::device_span;
using cudf::detail::grid_1d;

namespace cudf::io::json::detail::legacy {

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
__device__ std::pair<char const*, char const*> limit_range_to_brackets(char const* begin,
                                                                       char const* end)
{
  auto const data_begin = thrust::next(thrust::find_if(
    thrust::seq, begin, end, [] __device__(auto c) { return c == '[' || c == '{'; }));
  auto const data_end   = thrust::next(thrust::find_if(thrust::seq,
                                                     thrust::make_reverse_iterator(end),
                                                     thrust::make_reverse_iterator(data_begin),
                                                     [](auto c) { return c == ']' || c == '}'; }))
                          .base();
  return {data_begin, data_end};
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
__device__ std::pair<char const*, char const*> get_next_key(char const* begin,
                                                            char const* end,
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
  char const* value_begin;
  char const* value_end;
  bool is_quoted;
};

/**
 * @brief Parse the first field in the given range and return its descriptor.
 *
 * @param[in] begin Pointer to the first character in the parsing range
 * @param[in] end pointer to the first character after the parsing range
 * @param[in] opts The global parsing behavior options
 * @param[in] field_idx Index of the current field in the input row
 * @param[in] col_map Pointer to the (column name hash -> column index) map in device memory.
 * nullptr is passed when the input file does not consist of objects.
 * @return Descriptor of the parsed field
 */
__device__ field_descriptor next_field_descriptor(char const* begin,
                                                  char const* end,
                                                  parse_options_view const& opts,
                                                  cudf::size_type field_idx,
                                                  col_map_type col_map)
{
  auto const desc_pre_trim =
    col_map.capacity() == 0
      // No key - column and begin are trivial
      ? field_descriptor{field_idx,
                         begin,
                         cudf::io::gpu::seek_field_end(begin, end, opts, true),
                         false}
      : [&]() {
          auto const key_range = get_next_key(begin, end, opts.quotechar);
          auto const key_hash  = cudf::hashing::detail::MurmurHash3_x86_32<cudf::string_view>{}(
            cudf::string_view(key_range.first, key_range.second - key_range.first));
          auto const hash_col = col_map.find(key_hash);
          // Fall back to field index if not found (parsing error)
          auto const column = (hash_col != col_map.end()) ? (*hash_col).second : field_idx;

          // Skip the colon between the key and the value
          auto const value_begin = thrust::find(thrust::seq, key_range.second, end, ':') + 1;
          return field_descriptor{column,
                                  value_begin,
                                  cudf::io::gpu::seek_field_end(value_begin, end, opts, true),
                                  false};
        }();

  // Modify start & end to ignore whitespace and quotechars
  auto const trimmed_value_range =
    trim_whitespaces(desc_pre_trim.value_begin, desc_pre_trim.value_end);
  bool const is_quoted =
    thrust::distance(trimmed_value_range.first, trimmed_value_range.second) >= 2 and
    *trimmed_value_range.first == opts.quotechar and
    *thrust::prev(trimmed_value_range.second) == opts.quotechar;
  return {desc_pre_trim.column,
          trimmed_value_range.first + static_cast<std::ptrdiff_t>(is_quoted),
          trimmed_value_range.second - static_cast<std::ptrdiff_t>(is_quoted),
          is_quoted};
}

/**
 * @brief Returns the range that contains the data in a given row.
 *
 * Excludes the top-level brackets.
 *
 * @param[in] data Device span pointing to the JSON data in device memory
 * @param[in] row_offsets The offset of each row in the input
 * @param[in] row Index of the row for which the range is returned
 *
 * @return The begin and end iterators of the row data.
 */
__device__ std::pair<char const*, char const*> get_row_data_range(
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
 * @param[in] col_map Pointer to the (column name hash -> column index) map in device memory.
 * nullptr is passed when the input file does not consist of objects.
 * @param[out] output_columns The output column data
 * @param[out] valid_fields The bitmaps indicating whether column fields are valid
 * @param[out] num_valid_fields The numbers of valid fields in columns
 */
CUDF_KERNEL void convert_data_to_columns_kernel(parse_options_view opts,
                                                device_span<char const> const data,
                                                device_span<uint64_t const> const row_offsets,
                                                device_span<data_type const> const column_types,
                                                col_map_type col_map,
                                                device_span<void* const> const output_columns,
                                                device_span<bitmask_type* const> const valid_fields,
                                                device_span<cudf::size_type> const num_valid_fields)
{
  auto const rec_id = grid_1d::global_thread_id();
  if (rec_id >= row_offsets.size()) return;

  auto const row_data_range = get_row_data_range(data, row_offsets, rec_id);

  auto current = row_data_range.first;
  for (size_type input_field_index = 0;
       input_field_index < column_types.size() && current < row_data_range.second;
       input_field_index++) {
    auto const desc =
      next_field_descriptor(current, row_data_range.second, opts, input_field_index, col_map);
    auto const value_len = static_cast<size_t>(std::max(desc.value_end - desc.value_begin, 0L));
    auto const is_quoted = static_cast<std::ptrdiff_t>(desc.is_quoted);

    current = desc.value_end + 1;

    using string_index_pair = thrust::pair<char const*, size_type>;

    if (!serialized_trie_contains(opts.trie_na,
                                  {desc.value_begin - is_quoted, value_len + is_quoted * 2})) {
      // Type dispatcher does not handle strings
      if (column_types[desc.column].id() == type_id::STRING) {
        auto str_list           = static_cast<string_index_pair*>(output_columns[desc.column]);
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
                                  column_types[desc.column],
                                  opts,
                                  false)) {
          // set the valid bitmap - all bits were set to 0 to start
          set_bit(valid_fields[desc.column], rec_id);
          atomicAdd(&num_valid_fields[desc.column], 1);
        }
      }
    } else if (column_types[desc.column].id() == type_id::STRING) {
      auto str_list           = static_cast<string_index_pair*>(output_columns[desc.column]);
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
CUDF_KERNEL void detect_data_types_kernel(
  parse_options_view const opts,
  device_span<char const> const data,
  device_span<uint64_t const> const row_offsets,
  col_map_type col_map,
  int num_columns,
  device_span<cudf::io::column_type_histogram> const column_infos)
{
  auto const rec_id = grid_1d::global_thread_id();
  if (rec_id >= row_offsets.size()) return;

  auto const are_rows_objects = col_map.capacity() != 0;
  auto const row_data_range   = get_row_data_range(data, row_offsets, rec_id);

  size_type input_field_index = 0;
  for (auto current = row_data_range.first;
       input_field_index < num_columns && current < row_data_range.second;
       input_field_index++) {
    auto const desc =
      next_field_descriptor(current, row_data_range.second, opts, input_field_index, col_map);
    auto const value_len = static_cast<size_t>(std::max(desc.value_end - desc.value_begin, 0L));

    // Advance to the next field; +1 to skip the delimiter
    current = desc.value_end + 1;

    // Checking if the field is empty/valid
    if (serialized_trie_contains(opts.trie_na, {desc.value_begin, value_len})) {
      // Increase the null count for array rows, where the null count is initialized to zero.
      if (!are_rows_objects) { atomicAdd(&column_infos[desc.column].null_count, 1); }
      continue;
    } else if (are_rows_objects) {
      // For files with object rows, null count is initialized to row count. The value is decreased
      // here for every valid field.
      atomicAdd(&column_infos[desc.column].null_count, -1);
    }
    // Don't need counts to detect strings, any field in quotes is deduced to be a string
    if (desc.is_quoted) {
      atomicAdd(&column_infos[desc.column].string_count, 1);
      continue;
    }

    int digit_count    = 0;
    int decimal_count  = 0;
    int slash_count    = 0;
    int dash_count     = 0;
    int plus_count     = 0;
    int colon_count    = 0;
    int exponent_count = 0;
    int other_count    = 0;

    bool const maybe_hex =
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
        case '+': plus_count++; break;
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
    if ((*desc.value_begin == '-' || *desc.value_begin == '+') && value_len > 1) {
      --int_req_number_cnt;
    }
    // Off by one if they are a hexadecimal number
    if (maybe_hex) { --int_req_number_cnt; }
    if (serialized_trie_contains(opts.trie_true, {desc.value_begin, value_len}) ||
        serialized_trie_contains(opts.trie_false, {desc.value_begin, value_len})) {
      atomicAdd(&column_infos[desc.column].bool_count, 1);
    } else if (digit_count == int_req_number_cnt) {
      bool is_negative       = (*desc.value_begin == '-');
      char const* data_begin = desc.value_begin + (is_negative || (*desc.value_begin == '+'));
      cudf::size_type* ptr   = cudf::io::gpu::infer_integral_field_counter(
        data_begin, data_begin + digit_count, is_negative, column_infos[desc.column]);
      atomicAdd(ptr, 1);
    } else if (is_like_float(
                 value_len, digit_count, decimal_count, dash_count + plus_count, exponent_count)) {
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
  char const* key_begin;
  char const* key_end;
  char const* value_begin;
  char const* value_end;
};

/**
 * @brief Parse the next field in key:value format and return ranges of its parts.
 */
__device__ key_value_range get_next_key_value_range(char const* begin,
                                                    char const* end,
                                                    parse_options_view const& opts)
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
 */
CUDF_KERNEL void collect_keys_info_kernel(parse_options_view const options,
                                          device_span<char const> const data,
                                          device_span<uint64_t const> const row_offsets,
                                          unsigned long long int* keys_cnt,
                                          cuda::std::optional<mutable_table_device_view> keys_info)
{
  auto const rec_id = grid_1d::global_thread_id();
  if (rec_id >= row_offsets.size()) return;

  auto const row_data_range = get_row_data_range(data, row_offsets, rec_id);

  auto advance = [&](char const* begin) {
    return get_next_key_value_range(begin, row_data_range.second, options);
  };
  for (auto field_range = advance(row_data_range.first);
       field_range.key_begin < row_data_range.second;
       field_range = advance(field_range.value_end)) {
    auto const idx = atomicAdd(keys_cnt, 1ULL);
    if (keys_info.has_value()) {
      auto const len                              = field_range.key_end - field_range.key_begin;
      keys_info->column(0).element<uint64_t>(idx) = field_range.key_begin - data.begin();
      keys_info->column(1).element<uint16_t>(idx) = len;
      keys_info->column(2).element<uint32_t>(idx) =
        cudf::hashing::detail::MurmurHash3_x86_32<cudf::string_view>{}(
          cudf::string_view(field_range.key_begin, len));
    }
  }
}

}  // namespace

/**
 * @copydoc cudf::io::json::detail::legacy::convert_json_to_columns
 */
void convert_json_to_columns(parse_options_view const& opts,
                             device_span<char const> const data,
                             device_span<uint64_t const> const row_offsets,
                             device_span<data_type const> const column_types,
                             col_map_type* col_map,
                             device_span<void* const> const output_columns,
                             device_span<bitmask_type* const> const valid_fields,
                             device_span<cudf::size_type> num_valid_fields,
                             rmm::cuda_stream_view stream)
{
  int block_size;
  int min_grid_size;
  CUDF_CUDA_TRY(cudaOccupancyMaxPotentialBlockSize(
    &min_grid_size, &block_size, convert_data_to_columns_kernel));

  int const grid_size = (row_offsets.size() + block_size - 1) / block_size;

  convert_data_to_columns_kernel<<<grid_size, block_size, 0, stream.value()>>>(opts,
                                                                               data,
                                                                               row_offsets,
                                                                               column_types,
                                                                               *col_map,
                                                                               output_columns,
                                                                               valid_fields,
                                                                               num_valid_fields);

  CUDF_CHECK_CUDA(stream.value());
}

/**
 * @copydoc cudf::io::json::detail::legacy::detect_data_types
 */

std::vector<cudf::io::column_type_histogram> detect_data_types(
  parse_options_view const& options,
  device_span<char const> const data,
  device_span<uint64_t const> const row_offsets,
  bool do_set_null_count,
  int num_columns,
  col_map_type* col_map,
  rmm::cuda_stream_view stream)
{
  int block_size;
  int min_grid_size;
  CUDF_CUDA_TRY(
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, detect_data_types_kernel));

  auto d_column_infos = [&]() {
    if (do_set_null_count) {
      rmm::device_uvector<cudf::io::column_type_histogram> d_column_infos(num_columns, stream);
      // Set the null count to the row count (all fields assumes to be null).
      thrust::generate(
        rmm::exec_policy(stream),
        d_column_infos.begin(),
        d_column_infos.end(),
        [num_records = static_cast<cudf::size_type>(row_offsets.size())] __device__() {
          return cudf::io::column_type_histogram{num_records};
        });
      return d_column_infos;
    } else {
      return cudf::detail::make_zeroed_device_uvector_async<cudf::io::column_type_histogram>(
        num_columns, stream, rmm::mr::get_current_device_resource());
    }
  }();

  // Calculate actual block count to use based on records count
  int const grid_size = (row_offsets.size() + block_size - 1) / block_size;

  detect_data_types_kernel<<<grid_size, block_size, 0, stream.value()>>>(
    options, data, row_offsets, *col_map, num_columns, d_column_infos);

  return cudf::detail::make_std_vector_sync(d_column_infos, stream);
}

/**
 * @copydoc cudf::io::json::detail::legacy::collect_keys_info
 */
void collect_keys_info(parse_options_view const& options,
                       device_span<char const> const data,
                       device_span<uint64_t const> const row_offsets,
                       unsigned long long int* keys_cnt,
                       cuda::std::optional<mutable_table_device_view> keys_info,
                       rmm::cuda_stream_view stream)
{
  int block_size;
  int min_grid_size;
  CUDF_CUDA_TRY(
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, collect_keys_info_kernel));

  // Calculate actual block count to use based on records count
  int const grid_size = (row_offsets.size() + block_size - 1) / block_size;

  collect_keys_info_kernel<<<grid_size, block_size, 0, stream.value()>>>(
    options, data, row_offsets, keys_cnt, keys_info);

  CUDF_CHECK_CUDA(stream.value());
}

}  // namespace cudf::io::json::detail::legacy
