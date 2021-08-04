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

#include <cudf/wrappers/durations.hpp>
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

// User-defined literals to clarify numbers and units for time calculation
__inline__ __device__ constexpr uint32_t operator"" _days(unsigned long long int days)
{
  return days;
}
__inline__ __device__ constexpr uint32_t operator"" _erasInDays(unsigned long long int eras)
{
  return eras * 146097_days;  // multiply by days within an era (400 year span)
}
__inline__ __device__ constexpr uint32_t operator"" _years(unsigned long long int years)
{
  return years;
}
__inline__ __device__ constexpr uint32_t operator"" _erasInYears(unsigned long long int eras)
{
  return (eras * 1_erasInDays) / 365_days;
}

/**
 * @brief Computes the number of days since "March 1, 0000", given a date.
 *
 * This function takes year, month, and day and returns the number of days since the baseline which
 * is taken as 0000-03-01. This value is chosen as the origin for ease of calculation (now February
 * becomes the last month).
 *
 * @return days since March 1, 0000
 */
__inline__ __device__ constexpr int32_t days_since_baseline(int year, int month, int day)
{
  // More details of this formula are located in cuDF datetime_ops
  // In brief, the calculation is split over several components:
  //     era: a 400 year range, where the date cycle repeats exactly
  //     yoe: year within the 400 range of an era
  //     doy: day within the 364 range of a year
  //     doe: exact day within the whole era
  // The months are shifted so that March is the starting month and February
  // (possible leap day in it) is the last month for the linear calculation
  year -= (month <= 2) ? 1 : 0;

  const int32_t era = (year >= 0 ? year : year - 399_years) / 1_erasInYears;
  const int32_t yoe = year - era * 1_erasInYears;
  const int32_t doy = (153_days * (month + (month > 2 ? -3 : 9)) + 2) / 5 + day - 1;
  const int32_t doe = (yoe * 365_days) + (yoe / 4_years) - (yoe / 100_years) + doy;

  return (era * 1_erasInDays) + doe;
}

/**
 * @brief Computes the number of days since epoch, given a date.
 *
 * This function takes year, month, and day and returns the number of days since epoch (1970-01-01).
 *
 * @return days since epoch
 */
__inline__ __device__ constexpr int32_t days_since_epoch(int year, int month, int day)
{
  // Shift the start date to epoch to match unix time
  static_assert(static_cast<uint32_t>(days_since_baseline(1970, 1, 1)) == 719468_days,
                "Baseline to epoch returns incorrect number of days");

  return days_since_baseline(year, month, day) - days_since_baseline(1970, 1, 1);
}

/**
 * @brief Computes the number of seconds since epoch, given a date and time.
 *
 * This function takes year, month, day, hour, minute and second and returns
 * the number of seconds since epoch (1970-01-01),
 *
 * @return seconds since epoch
 */
__inline__ __device__ constexpr int64_t seconds_since_epoch(
  int year, int month, int day, int hour, int minute, int second)
{
  // Leverage the function to find the days since epoch
  const int64_t days = days_since_epoch(year, month, day);

  // Return sum total seconds from each time portion
  return (days * 24 * 60 * 60) + (hour * 60 * 60) + (minute * 60) + second;
}

/**
 * @brief Extracts the Day, Month, and Year from a string.
 *
 * @param[in] begin Pointer to the first element of the string
 * @param[in] end Pointer to the first element after the string
 * @param[in] dayfirst Flag indicating that first field is the day
 * @param[out] year
 * @param[out] month
 * @param[out] day
 * @return true if successful, false otherwise
 */
__inline__ __device__ bool extract_date(
  char const* begin, char const* end, bool dayfirst, int* year, int* month, int* day)
{
  char sep = '/';

  auto sep_pos = thrust::find(thrust::seq, begin, end, sep);

  if (sep_pos == end) {
    sep     = '-';
    sep_pos = thrust::find(thrust::seq, begin, end, sep);
  }

  if (sep_pos == end) return false;

  //--- is year the first filed?
  if ((sep_pos - begin) == 4) {
    *year = to_non_negative_integer<int>(begin, sep_pos);

    // Month
    auto s2 = sep_pos + 1;
    sep_pos = thrust::find(thrust::seq, s2, end, sep);

    if (sep_pos == end) {
      //--- Data is just Year and Month - no day
      *month = to_non_negative_integer<int>(s2, end);
      *day   = 1;

    } else {
      *month = to_non_negative_integer<int>(s2, sep_pos);
      *day   = to_non_negative_integer<int>((sep_pos + 1), end);
    }

  } else {
    //--- if the dayfirst flag is set, then restricts the format options
    if (dayfirst) {
      *day = to_non_negative_integer<int>(begin, sep_pos);

      auto s2 = sep_pos + 1;
      sep_pos = thrust::find(thrust::seq, s2, end, sep);

      *month = to_non_negative_integer<int>(s2, sep_pos);
      *year  = to_non_negative_integer<int>((sep_pos + 1), end);

    } else {
      *month = to_non_negative_integer<int>(begin, sep_pos);

      auto s2 = sep_pos + 1;
      sep_pos = thrust::find(thrust::seq, s2, end, sep);

      if (sep_pos == end) {
        //--- Data is just Year and Month - no day
        *year = to_non_negative_integer<int>(s2, end);
        *day  = 1;

      } else {
        *day  = to_non_negative_integer<int>(s2, sep_pos);
        *year = to_non_negative_integer<int>((sep_pos + 1), end);
      }
    }
  }

  return true;
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
 * @param[in] begin Pointer to the first element of the string
 * @param[in] end Pointer to the first element after the string
 * @param[out] hour The hour value
 * @param[out] minute The minute value
 * @param[out] second The second value (0 if not present)
 * @param[out] millisecond The millisecond (0 if not present)
 */
__inline__ __device__ void extract_time(
  char const* begin, char const* end, int* hour, int* minute, int* second, int* millisecond)
{
  constexpr char sep = ':';

  // Adjust for AM/PM and any whitespace before
  int hour_adjust = 0;
  auto last       = end - 1;
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
  *hour             = to_non_negative_integer<int>(begin, hm_sep) + hour_adjust;

  // Find minute-second separator (if present)
  const auto ms_sep = thrust::find(thrust::seq, hm_sep + 1, end, sep);
  if (ms_sep == end) {
    *minute      = to_non_negative_integer<int>(hm_sep + 1, end);
    *second      = 0;
    *millisecond = 0;
  } else {
    *minute = to_non_negative_integer<int>(hm_sep + 1, ms_sep);

    // Find second-millisecond separator (if present)
    const auto sms_sep = thrust::find(thrust::seq, ms_sep + 1, end, '.');
    if (sms_sep == end) {
      *second      = to_non_negative_integer<int>(ms_sep + 1, end);
      *millisecond = 0;
    } else {
      *second      = to_non_negative_integer<int>(ms_sep + 1, sms_sep);
      *millisecond = to_non_negative_integer<int>(sms_sep + 1, end);
    }
  }
}

/**
 * @brief Parses a date string into a `date32`, days since epoch.
 *
 * This function takes a string and produces a `date32` representation.
 * Acceptable formats are a combination of `MM/YYYY` and `MM/DD/YYYY`.
 *
 * @param[in] begin Pointer to the first element of the string
 * @param[in] end Pointer to the first element after the string
 * @param[in] dayfirst Flag to indicate that day is the first field - `DD/MM/YYYY`
 * @return Number of days since epoch
 */
__inline__ __device__ int32_t to_date(char const* begin, char const* end, bool dayfirst)
{
  int day, month, year;

  return extract_date(begin, end, dayfirst, &year, &month, &day)
           ? days_since_epoch(year, month, day)
           : -1;
}

/**
 * @brief Parses a datetime string and computes the number of milliseconds since epoch.
 *
 * This function takes a string and produces a `date32` representation.
 * Acceptable formats are a combination of `MM/YYYY` and `MM/DD/YYYY`.
 *
 * @param begin Pointer to the first element of the string
 * @param end Pointer to the first element after the string
 * @param dayfirst Flag to indicate day/month or month/day order
 * @return Milliseconds since epoch
 */
__inline__ __device__ int64_t to_date_time(char const* begin, char const* end, bool dayfirst)
{
  int day, month, year;
  int hour, minute, second, millisecond = 0;
  int64_t answer = -1;

  // Find end of the date portion
  // TODO: Refactor all the date/time parsing to remove multiple passes over each character because
  // of find() then convert(); that can also avoid the ugliness below.
  auto sep_pos = thrust::find(thrust::seq, begin, end, 'T');
  if (sep_pos == end) {
    // Attempt to locate the position between date and time, ignore premature space separators
    // around the day/month/year portions
    int count = 0;
    for (auto i = begin; i < end; ++i) {
      if (count == 3 && *i == ' ') {
        sep_pos = i;
        break;
      } else if ((*i == '/' || *i == '-') || (count == 2 && *i != ' ')) {
        count++;
      }
    }
  }

  // There is only date if there's no separator, otherwise it's malformed
  if (sep_pos != end) {
    if (extract_date(begin, sep_pos, dayfirst, &year, &month, &day)) {
      extract_time(sep_pos + 1, end, &hour, &minute, &second, &millisecond);
      answer = seconds_since_epoch(year, month, day, hour, minute, second) * 1000 + millisecond;
    }
  } else {
    if (extract_date(begin, end, dayfirst, &year, &month, &day)) {
      answer = seconds_since_epoch(year, month, day, 0, 0, 0) * 1000;
    }
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
  // %d days [+]%H:%M:%S.n => %d days, %d days [+]%H:%M:%S,  %H:%M:%S.n, %H:%M:%S, %value.
  constexpr char sep = ':';

  int32_t days{0};
  int8_t hour{0};
  // single pass to parse days, hour, minute, seconds, nanosecond
  auto cur         = begin;
  auto const value = parse_integer<int32_t>(&cur, end);
  cur              = skip_spaces(cur, end);
  if (std::is_same<T, cudf::duration_D>::value || cur >= end) {  // %value
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

  int nanosecond = 0;
  if (std::is_same<T, cudf::duration_s>::value) {
    return ((days * 24L + hour) * 60L + minute) * 60L + second;
  } else if (*cur == '.') {  //.n
    auto const start_subsecond        = ++cur;
    nanosecond                        = parse_integer<int>(&cur, end);
    int8_t const num_digits           = min(9L, cur - start_subsecond);
    constexpr int64_t powers_of_ten[] = {
      1L, 10L, 100L, 1000L, 10000L, 100000L, 1000000L, 10000000L, 100000000L, 1000000000L};
    nanosecond *= powers_of_ten[9 - num_digits];
  }

  return cuda::std::chrono::duration_cast<T>(
           cudf::duration_s{((days * 24L + hour) * 60L + minute) * 60L + second})
           .count() +
         cuda::std::chrono::duration_cast<T>(cudf::duration_ns{nanosecond}).count();
}

}  // namespace io
}  // namespace cudf
