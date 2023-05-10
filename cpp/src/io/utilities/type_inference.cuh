/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#pragma once

#include <io/utilities/column_type_histogram.hpp>
#include <io/utilities/parsing_utils.cuh>
#include <io/utilities/trie.cuh>

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>

#include <thrust/distance.h>
#include <thrust/tuple.h>

#include <cub/block/block_reduce.cuh>

#include <cstddef>

namespace cudf::io::detail {
/**
 * @brief Custom column_type_histogram sum reduction callable
 */
struct custom_sum {
  __device__ inline cudf::io::column_type_histogram operator()(
    cudf::io::column_type_histogram const& lhs, cudf::io::column_type_histogram const& rhs)
  {
    return {lhs.null_count + rhs.null_count,
            lhs.float_count + rhs.float_count,
            lhs.datetime_count + rhs.datetime_count,
            lhs.string_count + rhs.string_count,
            lhs.negative_small_int_count + rhs.negative_small_int_count,
            lhs.positive_small_int_count + rhs.positive_small_int_count,
            lhs.big_int_count + rhs.big_int_count,
            lhs.bool_count + rhs.bool_count};
  }
};

/**
 * @brief Returns true if the input character is a valid digit.
 * Supports both decimal and hexadecimal digits (uppercase and lowercase).
 *
 * @param c Character to check
 * @param is_hex Whether to check as a hexadecimal
 *
 * @return `true` if it is digit-like, `false` otherwise
 */
__device__ __inline__ bool is_digit(char const c, bool const is_hex = false)
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
__device__ __inline__ bool is_like_float(std::size_t len,
                                         uint32_t digit_cnt,
                                         uint32_t decimal_cnt,
                                         uint32_t dash_cnt,
                                         uint32_t exponent_cnt)
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
 * @brief Constructs column type histogram for a given column string input `data`.
 *
 * @tparam BlockSize Number of threads in each block
 * @tparam OptionsView Type of inference options view
 * @tparam ColumnStringIter Iterator type whose `value_type` is a
 * `thrust::tuple<offset_t, length_t>`, where `offset_t` and `length_t` are of integral type and
 * `offset_t` needs to be convertible to `std::size_t`.
 *
 * @param[in] options View of inference options
 * @param[in] data JSON string input
 * @param[in] column_strings_begin The beginning of an offset-length tuple sequence
 * @param[in] size Size of the string input
 * @param[out] column_info Histogram of column type counters
 */
template <int BlockSize, typename OptionsView, typename ColumnStringIter>
__global__ void infer_column_type_kernel(OptionsView options,
                                         device_span<char const> data,
                                         ColumnStringIter column_strings_begin,
                                         std::size_t size,
                                         cudf::io::column_type_histogram* column_info)
{
  auto thread_type_histogram = cudf::io::column_type_histogram{};

  for (auto idx = threadIdx.x + blockDim.x * blockIdx.x; idx < size;
       idx += gridDim.x * blockDim.x) {
    auto const field_offset = thrust::get<0>(*(column_strings_begin + idx));
    auto const field_len    = thrust::get<1>(*(column_strings_begin + idx));
    auto const field_begin  = data.begin() + field_offset;

    if (cudf::detail::serialized_trie_contains(
          options.trie_na, {field_begin, static_cast<std::size_t>(field_len)})) {
      ++thread_type_histogram.null_count;
      continue;
    }

    // Handling strings
    if (field_len >= 2 and *field_begin == options.quote_char and
        field_begin[field_len - 1] == options.quote_char) {
      ++thread_type_histogram.string_count;
      continue;
    }

    uint32_t digit_count    = 0;
    uint32_t decimal_count  = 0;
    uint32_t slash_count    = 0;
    uint32_t dash_count     = 0;
    uint32_t plus_count     = 0;
    uint32_t colon_count    = 0;
    uint32_t exponent_count = 0;
    uint32_t other_count    = 0;

    auto const maybe_hex =
      (field_len > 2 && field_begin[0] == '0' && field_begin[1] == 'x') ||
      (field_len > 3 && field_begin[0] == '-' && field_begin[1] == '0' && field_begin[2] == 'x');
    auto const field_end = field_begin + field_len;

    for (auto pos = field_begin; pos < field_end; ++pos) {
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
          if (!maybe_hex && pos > field_begin && pos < field_end - 1) exponent_count++;
          break;
        default: other_count++; break;
      }
    }

    // All characters must be digits in an integer, except for the starting sign and 'x' in the
    // hexadecimal prefix
    auto const int_req_number_cnt =
      static_cast<uint32_t>(field_len) -
      ((*field_begin == '-' || *field_begin == '+') && field_len > 1) - maybe_hex;
    if (cudf::detail::serialized_trie_contains(
          options.trie_true, {field_begin, static_cast<std::size_t>(field_len)}) ||
        cudf::detail::serialized_trie_contains(
          options.trie_false, {field_begin, static_cast<std::size_t>(field_len)})) {
      ++thread_type_histogram.bool_count;
    } else if (digit_count == int_req_number_cnt) {
      auto const is_negative = (*field_begin == '-');
      char const* data_begin = field_begin + (is_negative || (*field_begin == '+'));
      cudf::size_type* ptr   = cudf::io::gpu::infer_integral_field_counter(
        data_begin, data_begin + digit_count, is_negative, thread_type_histogram);
      ++*ptr;
    } else if (is_like_float(
                 field_len, digit_count, decimal_count, dash_count + plus_count, exponent_count)) {
      ++thread_type_histogram.float_count;
    }
    // All invalid JSON values are treated as string
    else {
      ++thread_type_histogram.string_count;
    }
  }  // grid-stride for loop

  using BlockReduce = cub::BlockReduce<cudf::io::column_type_histogram, BlockSize>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  auto const block_type_histogram =
    BlockReduce(temp_storage).Reduce(thread_type_histogram, custom_sum{});
  if (threadIdx.x == 0) {
    atomicAdd(&column_info->null_count, block_type_histogram.null_count);
    atomicAdd(&column_info->float_count, block_type_histogram.float_count);
    atomicAdd(&column_info->datetime_count, block_type_histogram.datetime_count);
    atomicAdd(&column_info->string_count, block_type_histogram.string_count);
    atomicAdd(&column_info->negative_small_int_count,
              block_type_histogram.negative_small_int_count);
    atomicAdd(&column_info->positive_small_int_count,
              block_type_histogram.positive_small_int_count);
    atomicAdd(&column_info->big_int_count, block_type_histogram.big_int_count);
    atomicAdd(&column_info->bool_count, block_type_histogram.bool_count);
  }
}

/**
 * @brief Constructs column type histogram for a given column string input `data`.
 *
 * @tparam OptionsView Type of inference options view
 * @tparam ColumnStringIter Iterator type whose `value_type` is a
 * `thrust::tuple<offset_t, length_t>`, where `offset_t` and `length_t` are of integral type and
 * `offset_t` needs to be convertible to `std::size_t`.
 *
 * @param options View of inference options
 * @param data JSON string input
 * @param column_strings_begin The beginning of an offset-length tuple sequence
 * @param size Size of the string input
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return A histogram containing column-specific type counters
 */
template <typename OptionsView, typename ColumnStringIter>
cudf::io::column_type_histogram infer_column_type(OptionsView const& options,
                                                  cudf::device_span<char const> data,
                                                  ColumnStringIter column_strings_begin,
                                                  std::size_t const size,
                                                  rmm::cuda_stream_view stream)
{
  constexpr int block_size = 128;

  auto const grid_size = (size + block_size - 1) / block_size;
  auto d_column_info   = rmm::device_scalar<cudf::io::column_type_histogram>(stream);
  CUDF_CUDA_TRY(cudaMemsetAsync(
    d_column_info.data(), 0, sizeof(cudf::io::column_type_histogram), stream.value()));

  infer_column_type_kernel<block_size><<<grid_size, block_size, 0, stream.value()>>>(
    options, data, column_strings_begin, size, d_column_info.data());

  return d_column_info.value(stream);
}

/**
 * @brief Infers data type for a given JSON string input `data`.
 *
 * @throw cudf::logic_error if input size is 0
 * @throw cudf::logic_error if date time is not inferred as string
 * @throw cudf::logic_error if data type inference failed
 *
 * @tparam OptionsView Type of inference options view
 * @tparam ColumnStringIter Iterator type whose `value_type` is convertible to
 * `thrust::tuple<device_span, string_view>`
 *
 * @param options View of inference options
 * @param data JSON string input
 * @param column_strings_begin The beginning of an offset-length tuple sequence
 * @param size Size of the string input
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return The inferred data type
 */
template <typename OptionsView, typename ColumnStringIter>
cudf::data_type infer_data_type(OptionsView const& options,
                                device_span<char const> data,
                                ColumnStringIter column_strings_begin,
                                std::size_t const size,
                                rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(size != 0, "No data available for data type inference.\n");

  auto const h_column_info = infer_column_type(options, data, column_strings_begin, size, stream);

  auto get_type_id = [&](auto const& cinfo) {
    auto int_count_total =
      cinfo.big_int_count + cinfo.negative_small_int_count + cinfo.positive_small_int_count;
    if (cinfo.null_count == static_cast<cudf::size_type>(size)) {
      // Entire column is NULL; allocate the smallest amount of memory
      return type_id::INT8;
    } else if (cinfo.string_count > 0) {
      return type_id::STRING;
    } else if (cinfo.datetime_count > 0) {
      CUDF_FAIL("Date time is inferred as string.\n");
    } else if (cinfo.float_count > 0) {
      return type_id::FLOAT64;
    } else if (cinfo.big_int_count == 0 && int_count_total != 0) {
      return type_id::INT64;
    } else if (cinfo.big_int_count != 0 && cinfo.negative_small_int_count != 0) {
      return type_id::STRING;
    } else if (cinfo.big_int_count != 0) {
      return type_id::UINT64;
    } else if (cinfo.bool_count > 0) {
      return type_id::BOOL8;
    }
    CUDF_FAIL("Data type inference failed.\n");
  };
  return cudf::data_type{get_type_id(h_column_info)};
}
}  // namespace cudf::io::detail
