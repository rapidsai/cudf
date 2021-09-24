/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <thrust/reduce.h>

#include <cudf/wrappers/timestamps.hpp>
#include <io/utilities/parsing_utils.cuh>

namespace cudf {
namespace io {

/**
 * @brief Parses non-negative integral vales.
 *
 * This helper function is only intended to handle positive integers. The input
 * character string is expected to be well-formed.
 *
 * @param begin Pointer to the first element of the string
 * @param end Pointer to the first element after the string
 * @return The parsed and converted value
 */
template <typename T>
__inline__ __device__ T to_non_negative_integer(char const* begin, char const* end)
{
  T value = 0;

  for (; begin < end; ++begin) {
    if (*begin >= '0' && *begin <= '9') {
      value *= 10;
      value += *begin - '0';
    }
  }

  return value;
}

/**
 * @brief Extracts the Day, Month, and Year from a string.
 *
 * @param begin Pointer to the first element of the string
 * @param end Pointer to the first element after the string
 * @param dayfirst Flag indicating that first field is the day
 * @return Extracted year, month and day in `cuda::std::chrono::year_month_day` format
 */
__inline__ __device__ cuda::std::chrono::year_month_day extract_date(char const* begin,
                                                                     char const* end,
                                                                     bool dayfirst)
{
  char sep = '/';

  auto sep_pos = thrust::find(thrust::seq, begin, end, sep);

  if (sep_pos == end) {
    sep     = '-';
    sep_pos = thrust::find(thrust::seq, begin, end, sep);
  }

  int y;          // year is signed
  unsigned m, d;  // month and day are unsigned

  //--- is year the first filed?
  if ((sep_pos - begin) == 4) {
    y = to_non_negative_integer<int>(begin, sep_pos);

    // Month
    auto s2 = sep_pos + 1;
    sep_pos = thrust::find(thrust::seq, s2, end, sep);

    if (sep_pos == end) {
      //--- Data is just Year and Month - no day
      m = to_non_negative_integer<int>(s2, end);
      d = 1;

    } else {
      m = to_non_negative_integer<int>(s2, sep_pos);
      d = to_non_negative_integer<int>((sep_pos + 1), end);
    }

  } else {
    //--- if the dayfirst flag is set, then restricts the format options
    if (dayfirst) {
      d = to_non_negative_integer<int>(begin, sep_pos);

      auto s2 = sep_pos + 1;
      sep_pos = thrust::find(thrust::seq, s2, end, sep);

      m = to_non_negative_integer<int>(s2, sep_pos);
      y = to_non_negative_integer<int>((sep_pos + 1), end);

    } else {
      m = to_non_negative_integer<int>(begin, sep_pos);

      auto s2 = sep_pos + 1;
      sep_pos = thrust::find(thrust::seq, s2, end, sep);

      if (sep_pos == end) {
        //--- Data is just Year and Month - no day
        y = to_non_negative_integer<int>(s2, end);
        d = 1;

      } else {
        d = to_non_negative_integer<int>(s2, sep_pos);
        y = to_non_negative_integer<int>((sep_pos + 1), end);
      }
    }
  }

  using namespace cuda::std::chrono;
  return year_month_day{year{y}, month{m}, day{d}};
}

/**
 * @brief Parses a string to extract the hour, minute, second and millisecond time field
 * values.
 *
 * Incoming format is expected to be `HH:MM:SS.MS`, with the latter second and millisecond fields
 * optional. Each time field can be a single, double, or triple (in the case of milliseconds)
 * digits. 12-hr and 24-hr time format is detected via the absence or presence of AM/PM characters
 * at the end.
 *
 * @param begin Pointer to the first element of the string
 * @param end Pointer to the first element after the string
 * @return Duration in cudf milliseconds by summing up the extracted hours, minutes, seconds and
 * milliseconds
 */
__inline__ __device__ cudf::duration_ms extract_time(char const* begin, char const* end)
{
  constexpr char sep = ':';

  // Adjust for AM/PM and any whitespace before
  int32_t hour_adjust = 0;
  auto last           = end - 1;
  if (*last == 'M' || *last == 'm') {
    if (*(last - 1) == 'P' || *(last - 1) == 'p') { hour_adjust = 12; }
    last = last - 2;
    while (*last == ' ') {
      --last;
    }
  }
  end = last + 1;

  // Find hour-minute separator
  const auto hm_sep = thrust::find(thrust::seq, begin, end, sep);
  cudf::duration_h d_h{to_non_negative_integer<int>(begin, hm_sep) + hour_adjust};

  int m, s, ms;

  // Find minute-second separator (if present)
  const auto ms_sep = thrust::find(thrust::seq, hm_sep + 1, end, sep);
  if (ms_sep == end) {
    m  = to_non_negative_integer<int>(hm_sep + 1, end);
    s  = 0;
    ms = 0;
  } else {
    m = to_non_negative_integer<int>(hm_sep + 1, ms_sep);

    // Find second-millisecond separator (if present)
    const auto sms_sep = thrust::find(thrust::seq, ms_sep + 1, end, '.');
    if (sms_sep == end) {
      s  = to_non_negative_integer<int>(ms_sep + 1, end);
      ms = 0;
    } else {
      s  = to_non_negative_integer<int>(ms_sep + 1, sms_sep);
      ms = to_non_negative_integer<int>(sms_sep + 1, end);
    }
  }
  return d_h + duration_m{m} + duration_s{s} + duration_ms{ms};
}

/**
 * @brief Parses a datetime string and computes the corresponding time stamp.
 *
 * This function takes a string and produces a `timestamp_type` representation.
 * Acceptable formats are a combination of `MM/YYYY` and `MM/DD/YYYY`.
 *
 * @tparam timestamp_type Type of output time stamp
 * @param begin Pointer to the first element of the string
 * @param end Pointer to the first element after the string
 * @param dayfirst Flag to indicate day/month or month/day order
 * @return Time stamp converted to `timestamp_type`
 */
template <typename timestamp_type>
__inline__ __device__ timestamp_type to_date_time(char const* begin, char const* end, bool dayfirst)
{
  using duration_type = typename timestamp_type::duration;

  auto sep_pos = end;

  // Find end of the date portion
  int count        = 0;
  bool digits_only = true;
  for (auto i = begin; i < end; ++i) {
    digits_only &= (*i >= '0' and *i <= '9');
    if (*i == 'T') {
      sep_pos = i;
      break;
    } else if (count == 3 && *i == ' ') {
      sep_pos = i;
      break;
    } else if ((*i == '/' || *i == '-') || (count == 2 && *i != ' ')) {
      count++;
    }
  }

  // Exit if the input string is digit-only
  if (digits_only) {
    return timestamp_type{
      duration_type{to_non_negative_integer<typename timestamp_type::rep>(begin, end)}};
  }

  auto ymd = extract_date(begin, sep_pos, dayfirst);
  timestamp_type answer{cuda::std::chrono::sys_days{ymd}};

  // Extract time only if separator is present
  if (sep_pos != end) {
    auto t = extract_time(sep_pos + 1, end);
    answer += cuda::std::chrono::duration_cast<duration_type>(t);
  }

  return answer;
}

/**
 * @brief Parses the input string into an integral value of the given type.
 *
 * Moves the `begin` iterator past the parsed value.
 *
 * @param begin[in, out] Pointer to the first element of the string
 * @param end Pointer to the first element after the string
 * @return The parsed and converted value
 */
template <typename T>
__inline__ __device__ T parse_integer(char const** begin, char const* end)
{
  bool const is_negative = (**begin == '-');
  T value                = 0;

  auto cur = *begin + is_negative;
  while (cur < end) {
    if (*cur >= '0' && *cur <= '9') {
      value *= 10;
      value += *cur - '0';
    } else
      break;
    ++cur;
  }
  *begin = cur;

  return is_negative ? -value : value;
}

/**
 * @brief Parses the input string into an integral value of the given type if the delimiter is
 * present.
 *
 * Moves the `begin` iterator past the parsed value.
 *
 * @param begin[in, out] Pointer to the first element of the string
 * @param end Pointer to the first element after the string
 * @return The parsed and converted value, zero is delimiter is not present
 */
template <typename T>
__inline__ __device__ T parse_optional_integer(char const** begin, char const* end, char delimiter)
{
  if (**begin != delimiter) { return 0; }

  ++(*begin);
  return parse_integer<T>(begin, end);
}

/**
 * @brief Parses the input string into a duration of the given type.
 *
 * @param begin Pointer to the first element of the string
 * @param end Pointer to the first element after the string
 * @return The parsed duration
 */
template <typename T>
__inline__ __device__ int64_t to_time_delta(char const* begin, char const* end)
{
  using cuda::std::chrono::duration_cast;

  // %d days [+]%H:%M:%S.n => %d days, %d days [+]%H:%M:%S,  %H:%M:%S.n, %H:%M:%S, %value.
  constexpr char sep = ':';

  int32_t days{0};
  int8_t hour{0};
  // single pass to parse days, hour, minute, seconds, nanosecond
  auto cur         = begin;
  auto const value = parse_integer<int32_t>(&cur, end);
  cur              = skip_spaces(cur, end);
  if (std::is_same_v<T, cudf::duration_D> || cur >= end) {  // %value
    return value;
  }
  // " days [+]"
  auto const after_days_sep     = skip_if_starts_with(cur, end, "days");
  auto const has_days_seperator = (after_days_sep != cur);
  cur                           = skip_spaces(after_days_sep, end);
  cur += (*cur == '+');
  if (has_days_seperator) {
    days = value;
    hour = parse_integer<int8_t>(&cur, end);
  } else {
    hour = value;
  }

  auto const minute = parse_optional_integer<int8_t>(&cur, end, sep);
  auto const second = parse_optional_integer<int8_t>(&cur, end, sep);

  cudf::duration_D d{days};
  cudf::duration_h h{hour};
  cudf::duration_m m{minute};
  cudf::duration_s s{second};
  // Convert all durations to the given type
  auto res_duration = duration_cast<T>(d).count() + duration_cast<T>(h).count() +
                      duration_cast<T>(m).count() + duration_cast<T>(s).count();

  int nanosecond = 0;

  if (std::is_same_v<T, cudf::duration_s>) {
    return res_duration;
  } else if (*cur == '.') {  //.n
    auto const start_subsecond        = ++cur;
    nanosecond                        = parse_integer<int>(&cur, end);
    int8_t const num_digits           = min(9L, cur - start_subsecond);
    constexpr int64_t powers_of_ten[] = {
      1L, 10L, 100L, 1000L, 10000L, 100000L, 1000000L, 10000000L, 100000000L, 1000000000L};
    nanosecond *= powers_of_ten[9 - num_digits];
  }

  return res_duration + duration_cast<T>(cudf::duration_ns{nanosecond}).count();
}

}  // namespace io
}  // namespace cudf
