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

#pragma once

#include <cudf/wrappers/durations.hpp>

/**
 * @brief Returns location to the first occurrence of a character in a string
 *
 * This helper function takes a string and a search range to return the location
 * of the first instance of the specified character.
 *
 * @param[in] begin Beginning of the character string
 * @param[in] end End of the character string
 * @param[in] c Character to find
 *
 * @return index into the string, or -1 if the character is not found
 */
#include "thrust/reduce.h"
__inline__ __device__ char const* findFirstOccurrence(char const* begin, char const* end, char c)
{
  while (begin < end and *begin != c) { begin++; }

  return begin;
}

/**
 * @brief Simplified parsing function for use by date and time parsing
 *
 * This helper function is only intended to handle positive integers. The input
 * character string is expected to be well-formed.
 *
 * @param[in] begin Beginning of the character string
 * @param[in] end End of the character string
 *
 * @return The parsed and converted value
 */
template <typename T>
__inline__ __device__ T convertStrToInteger(char const* begin, char const* end)
{
  T value = 0;

  for (; begin <= end; ++begin) {
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
 * @brief Compute number of days since "March 1, 0000", given a date
 *
 * This function takes year, month, and day and returns the number of days
 * since the baseline which is taken as 0000-03-01. This value is chosen as the
 * origin for ease of calculation (now February becomes the last month).
 *
 * @param[in] year
 * @param[in] month
 * @param[in] day
 *
 * @return days since March 1, 0000
 */
__inline__ __device__ constexpr int32_t daysSinceBaseline(int year, int month, int day)
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
 * @brief Compute number of days since epoch, given a date
 *
 * This function takes year, month, and day and returns the number of days
 * since epoch (1970-01-01).
 *
 * @param[in] year
 * @param[in] month
 * @param[in] day
 *
 * @return days since epoch
 */
__inline__ __device__ constexpr int32_t daysSinceEpoch(int year, int month, int day)
{
  // Shift the start date to epoch to match unix time
  static_assert(daysSinceBaseline(1970, 1, 1) == 719468_days,
                "Baseline to epoch returns incorrect number of days");

  return daysSinceBaseline(year, month, day) - daysSinceBaseline(1970, 1, 1);
}

/**
 * @brief Compute the number of seconds since epoch, given a date and time
 *
 * This function takes year, month, day, hour, minute and second and returns
 * the number of seconds since epoch (1970-01-01)
 *
 * @param[in] year
 * @param[in] month
 * @param[in] day
 * @param[in] hour
 * @param[in] minute
 * @param[in] second
 *
 * @return seconds since epoch
 */
__inline__ __device__ constexpr int64_t secondsSinceEpoch(
  int year, int month, int day, int hour, int minute, int second)
{
  // Leverage the function to find the days since epoch
  const int64_t days = daysSinceEpoch(year, month, day);

  // Return sum total seconds from each time portion
  return (days * 24 * 60 * 60) + (hour * 60 * 60) + (minute * 60) + second;
}

/**
 * @brief Extract the Day, Month, and Year from a string
 *
 * @param[in] begin Beginning of the character string
 * @param[in] end End of the character string
 * @param[in] dayfirst Flag indicating that first field is the day
 * @param[out] year
 * @param[out] month
 * @param[out] day
 *
 * @return true if successful, false otherwise
 */
__inline__ __device__ bool extractDate(
  char const* begin, char const* end, bool dayfirst, int* year, int* month, int* day)
{
  char sep = '/';

  auto sep_pos = findFirstOccurrence(begin, end, sep);

  if (sep_pos == end) {
    sep     = '-';
    sep_pos = findFirstOccurrence(begin, end, sep);
  }

  if (sep_pos == end) return false;

  //--- is year the first filed?
  if ((sep_pos - begin) == 4) {
    *year = convertStrToInteger<int>(begin, (sep_pos - 1));

    // Month
    auto s2 = sep_pos + 1;
    sep_pos = findFirstOccurrence(s2, end, sep);

    if (sep_pos == end) {
      //--- Data is just Year and Month - no day
      *month = convertStrToInteger<int>(s2, end);
      *day   = 1;

    } else {
      *month = convertStrToInteger<int>(s2, (sep_pos - 1));
      *day   = convertStrToInteger<int>((sep_pos + 1), end);
    }

  } else {
    //--- if the dayfirst flag is set, then restricts the format options
    if (dayfirst) {
      *day = convertStrToInteger<int>(begin, (sep_pos - 1));

      auto s2 = sep_pos + 1;
      sep_pos = findFirstOccurrence(s2, end, sep);

      *month = convertStrToInteger<int>(s2, (sep_pos - 1));
      *year  = convertStrToInteger<int>((sep_pos + 1), end);

    } else {
      *month = convertStrToInteger<int>(begin, (sep_pos - 1));

      auto s2 = sep_pos + 1;
      sep_pos = findFirstOccurrence(s2, end, sep);

      if (sep_pos == end) {
        //--- Data is just Year and Month - no day
        *year = convertStrToInteger<int>(s2, end);
        *day  = 1;

      } else {
        *day  = convertStrToInteger<int>(s2, (sep_pos - 1));
        *year = convertStrToInteger<int>((sep_pos + 1), end);
      }
    }
  }

  return true;
}

/**
 * @brief Parse a character stream to extract the hour, minute, second and
 * millisecond time field values.
 *
 * Incoming format is expected to be HH:MM:SS.MS, with the latter second and
 * millisecond fields optional. Each time field can be a single, double,
 * or triple (in the case of milliseconds) digits. 12-hr and 24-hr time format
 * is detected via the absence or presence of AM/PM characters at the end.
 *
 * @param[in] begin Beginning of the character string
 * @param[in] end End of the character string
 * @param[out] hour The hour value
 * @param[out] minute The minute value
 * @param[out] second The second value (0 if not present)
 * @param[out] millisecond The millisecond (0 if not present)
 */
__inline__ __device__ void extractTime(
  char const* begin, char const* end, int* hour, int* minute, int* second, int* millisecond)
{
  constexpr char sep = ':';

  // Adjust for AM/PM and any whitespace before
  int hour_adjust = 0;
  if (*end == 'M' || *end == 'm') {
    if (*(end - 1) == 'P' || *(end - 1) == 'p') { hour_adjust = 12; }
    end = end - 2;
    while (*end == ' ') { --end; }
  }

  // Find hour-minute separator
  const auto hm_sep = findFirstOccurrence(begin, end, sep);
  *hour             = convertStrToInteger<int>(begin, hm_sep - 1) + hour_adjust;

  // Find minute-second separator (if present)
  const auto ms_sep = findFirstOccurrence(hm_sep + 1, end, sep);
  if (ms_sep == end) {
    *minute      = convertStrToInteger<int>(hm_sep + 1, end);
    *second      = 0;
    *millisecond = 0;
  } else {
    *minute = convertStrToInteger<int>(hm_sep + 1, ms_sep - 1);

    // Find second-millisecond separator (if present)
    const auto sms_sep = findFirstOccurrence(ms_sep + 1, end, '.');
    if (sms_sep == end) {
      *second      = convertStrToInteger<int>(ms_sep + 1, end);
      *millisecond = 0;
    } else {
      *second      = convertStrToInteger<int>(ms_sep + 1, sms_sep - 1);
      *millisecond = convertStrToInteger<int>(sms_sep + 1, end);
    }
  }
}

/**
 * @brief Parse a Date string into a date32, days since epoch
 *
 * This function takes a string and produces a date32 representation
 * Acceptable formats are a combination of MM/YYYY and MM/DD/YYYY
 *
 * @param[in] begin Beginning of the character string
 * @param[in] end End of the character string
 * @param[in] dayfirst Flag to indicate that day is the first field - DD/MM/YYYY
 *
 * @return returns the number of days since epoch
 */
__inline__ __device__ int32_t parseDateFormat(char const* begin, char const* end, bool dayfirst)
{
  int day, month, year;
  int32_t e = -1;

  bool status = extractDate(begin, end, dayfirst, &year, &month, &day);

  if (status) e = daysSinceEpoch(year, month, day);

  return e;
}

/**
 * @brief Parses a datetime character stream and computes the number of
 * milliseconds since epoch.
 *
 * This function takes a string and produces a date32 representation
 * Acceptable formats are a combination of MM/YYYY and MM/DD/YYYY
 *
 * @param[in] begin Beginning of the character string
 * @param[in] end End of the character string
 * @param[in] dayfirst Flag to indicate day/month or month/day order
 *
 * @return Milliseconds since epoch
 */
__inline__ __device__ int64_t parseDateTimeFormat(char const* begin, char const* end, bool dayfirst)
{
  int day, month, year;
  int hour, minute, second, millisecond = 0;
  int64_t answer = -1;

  // Find end of the date portion
  // TODO: Refactor all the date/time parsing to remove multiple passes over
  // each character because of find() then convert(); that can also avoid the
  // ugliness below.
  auto sep_pos = findFirstOccurrence(begin, end, 'T');
  if (sep_pos == end) {
    // Attempt to locate the position between date and time, ignore premature
    // space separators around the day/month/year portions
    int count = 0;
    for (auto i = begin; i <= end; ++i) {
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
    if (extractDate(begin, sep_pos - 1, dayfirst, &year, &month, &day)) {
      extractTime(sep_pos + 1, end, &hour, &minute, &second, &millisecond);
      answer = secondsSinceEpoch(year, month, day, hour, minute, second) * 1000 + millisecond;
    }
  } else {
    if (extractDate(begin, end, dayfirst, &year, &month, &day)) {
      answer = secondsSinceEpoch(year, month, day, 0, 0, 0) * 1000;
    }
  }

  return answer;
}

// parse integer and update the start position
template <typename T>
__inline__ __device__ T parse_integer(const char* data, long& start, long end)
{
  bool is_negative = data[start] == '-';
  T value          = 0;

  long index = start + is_negative;
  while (index <= end) {
    if (data[index] >= '0' && data[index] <= '9') {
      value *= 10;
      value += data[index] - '0';
    } else
      break;
    ++index;
  }
  start = index;

  return is_negative ? -value : value;
}

__inline__ __device__ bool is_present(
  const char* data, long& start, long end, const char* needle, int len)
{
  if (start + len - 1 > end) return false;
  for (auto i = 0; i < len; i++) {
    if (data[start + i] != needle[i]) return false;
  }
  start += len;
  return true;
}

__inline__ __device__ int64_t parseDaysDeltaFormat(const char* data, long start, long end)
{
  return parse_integer<int>(data, start, end);
}

template <typename T>
__inline__ __device__ int64_t parseTimeDeltaFormat(const char* data, long start, long end)
{
  // %d days [+]%H:%M:%S.n => %d days, %d days [+]%H:%M:%S,  %H:%M:%S.n, %H:%M:%S, %value.
  int days{0};
  int8_t hour{0}, minute{0}, second{0};
  int nanosecond     = 0;
  constexpr char sep = ':';

  // single pass to parse days, hour, minute, seconds, nanosecond
  long moving_pos = start;
  int32_t value   = parse_integer<int>(data, moving_pos, end);
  if (std::is_same<T, cudf::duration_D>::value) return value;
  while (data[moving_pos] == ' ') moving_pos++;
  if (moving_pos >= end) return value;  // %value
  // " days [+]"
  const bool days_seperator = is_present(data, moving_pos, end, "days", 4);
  while (data[moving_pos] == ' ') moving_pos++;
  moving_pos += (data[moving_pos] == '+');
  if (days_seperator) {
    days = value;
    hour = parse_integer<int>(data, moving_pos, end);
  } else {
    hour = value;
  }

  //:%M:%S
  if (data[moving_pos] == sep) { minute = parse_integer<int>(data, ++moving_pos, end); }
  if (data[moving_pos] == sep) { second = parse_integer<int>(data, ++moving_pos, end); }
  if (std::is_same<T, cudf::duration_s>::value) {
    return ((days * 24L + hour) * 60L + minute) * 60L + second;
  }
  //.n
  if (data[moving_pos] == '.') {
    auto start_subsecond              = moving_pos + 1;
    nanosecond                        = parse_integer<int>(data, ++moving_pos, end);
    int8_t num_digits                 = min(9L, moving_pos - start_subsecond);
    constexpr int64_t powers_of_ten[] = {
      1L, 10L, 100L, 1000L, 10000L, 100000L, 1000000L, 10000000L, 100000000L, 1000000000L};
    nanosecond *= powers_of_ten[9 - num_digits];
  }

  return simt::std::chrono::duration_cast<T>(
           cudf::duration_s{((days * 24L + hour) * 60L + minute) * 60L + second})
           .count() +
         simt::std::chrono::duration_cast<T>(cudf::duration_ns{nanosecond}).count();
}
