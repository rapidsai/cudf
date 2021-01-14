/*
 * Copyright (c) 2021, Baidu CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <thrust/logical.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {
/**
 * Check whether the string is valid when convert string to signed integers,
 * like INT8/16/32/64. For example, if allow_decimal is true, then strings 
 * `['1.23', '123']` will return `[true, true]`.
 * If `allow_decimal` is false, then this function will return `[false, true]`.
 * 
 * Note that, in this method we accumulate the result in negative format, and convert it to
 * positive format at the end, if this string is not started with '-'. This is because min value
 * is bigger than max value in digits, e.g. Long.MAX_VALUE is '9223372036854775807' and
 * Long.MIN_VALUE is '-9223372036854775808'.
 *
 * This code is heavily based off of LazyLong.parseLong from Hive, but updated for C++.
 *
 * @param d_str String to check.
 * @param allow_decimal whether we allow the data is Decimal type or not.
 * @param min_value min_value that corresponds to the type that is checking.
 * @return true if string has valid integer characters or decimal characters.
 */
__device__ bool is_valid_element(string_view const& d_str, bool allow_decimal, long min_value)
{
  int offset = 0;
  size_type bytes = d_str.size_bytes();
  const char* data    = d_str.data();
  // strip leading white space
  while (offset < bytes && data[offset] == ' ') ++offset;
  if (offset == bytes)  return false;

  int end = bytes - 1;
  // strip trailing white space
  while (end > offset && data[end] == ' ') --end;

  char c_sign = data[offset];
  const bool negative = c_sign == '-';
  if (negative || c_sign == '+'){
    if (end - offset == 0)  return false;
    ++offset;
  }

  const char separator = '.';
  const int radix = 10;
  const long stop_value = min_value / radix;
  long result = 0;

  while (offset <= end) {
    const char c = data[offset];
    ++offset;
    // We allow decimals and will return a truncated integral in that case.
    // Therefore we won't throw an exception here (checking the fractional
    // part happens below).
    if (c == separator && allow_decimal)  break;

    int digit;
    if (c >= '0' && c <= '9'){
      digit = c - '0';
    } else {
      return false;
    }

    // We are going to process the new digit and accumulate the result. However, 
    // before doing this, if the result is already smaller than the stop_value which is
    // (std::numeric_limits<data_type>::min() / radix), then result * 10 will definitely 
    // be smaller than the min_value, and we can stop.
    if (result < stop_value)  return false;

    result = result * radix - digit;

    // Since the previous result is less than or equal to stopValue which is 
    // (std::numeric_limits<data_type>::min() / radix), we can just use `result > 0` 
    // to check overflow. If result overflows, we should stop.
    if (result > 0) return false;
  }
  // This is the case when we've encountered a decimal separator. The fractional
  // part will not change the number, but we will verify that the fractional part
  // is well formed.
  if (offset <= end && thrust::any_of(thrust::seq,
                                      data+offset,
                                      data+end,
                                      [] (char ch) {
                                        return (ch<'0' || ch>'9');
                                      }))
    return false;

  if (!negative) {
    result = -result;
    if (result < 0) return false;
  }

  return true;
}

} //namespace

/**
 * @brief The dispatch functions return the min value of the input data type
 * used for checking overflow.
 *
 * The output is the min value of specified type.
 */
struct min_value_of_type{
  template <typename T>
  long operator()()
  { 
    CUDF_FAIL("Unsupported current data type check."); 
  }
};

template <>
long min_value_of_type::operator()<int8_t>() { return std::numeric_limits<int8_t>::min(); }

template <>
long min_value_of_type::operator()<int16_t>() { return std::numeric_limits<int16_t>::min(); }

template <>
long min_value_of_type::operator()<int32_t>() { return std::numeric_limits<int32_t>::min(); }

template <>
long min_value_of_type::operator()<int64_t>() { return std::numeric_limits<int64_t>::min(); }

std::unique_ptr<column> is_valid_element(
  strings_column_view const& strings,
  bool allow_decimal,
  data_type input_type,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_column       = *strings_column;
  auto d_allow_decimal = allow_decimal;

  // ready a min_value corresponds to the input type in order to check overflow
  long d_min_value = cudf::type_dispatcher(input_type, min_value_of_type{}) ;

  // create output column
  auto results   = make_numeric_column(data_type{type_id::BOOL8},
                                     strings.size(),
                                     cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                                     strings.null_count(),
                                     stream,
                                     mr);
  auto d_results = results->mutable_view().data<bool>();
  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(strings.size()),
                    d_results,
                    [d_column,d_allow_decimal,d_min_value] __device__(size_type idx) {
                      if (d_column.is_null(idx)) return false;
                      return is_valid_element(d_column.element<string_view>(idx), d_allow_decimal, d_min_value);
                    });
  results->set_null_count(strings.null_count());
  return results;
}

}  // namespace detail

// external API

std::unique_ptr<column> is_valid_element(strings_column_view const& strings,
                                          bool allow_decimal,
                                          data_type input_type,
                                          rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::is_valid_element(strings, allow_decimal, input_type, rmm::cuda_stream_default, mr);
}

}  // namespace strings
}  // namespace cudf

