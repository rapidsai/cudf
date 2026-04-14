/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "io/utilities/parsing_utils.cuh"
#include "io/utilities/time_utils.cuh"

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/strings/string_view.hpp>

#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/reduce.h>

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
 * @brief Converts a month name token to its numeric value 1–12, or 0 if not recognized.
 *
 * Supports abbreviated names (e.g. "Jan") and full names (e.g. "January"). Matching is
 * case-insensitive for ASCII characters.
 *
 * When @p locale_names is non-null it must point to a device array of 24 `cudf::string_view`
 * objects laid out as follows (matching the order returned by `nl_langinfo`):
 *   - indices  0–11 : abbreviated month names (ABMON_1 .. ABMON_12)
 *   - indices 12–23 : full month names        (MON_1   .. MON_12)
 *
 * This locale-aware path mirrors the behaviour of the `cudf::strings::to_timestamps` API
 * which accepts names derived from `nl_langinfo(ABMON_*)` / `nl_langinfo(MON_*)` on the
 * host and passes them into the GPU kernel — see `cudf/strings/convert/convert_datetime.hpp`.
 *
 * When @p locale_names is null the function falls back to the hard-coded English ASCII table
 * (same behaviour as before locale support was added).
 *
 * @param begin        Pointer to the first character of the month-name token
 * @param end          Pointer one past the last character of the token
 * @param locale_names Device pointer to 24 locale `string_view`s, or nullptr
 * @return Month number 1–12, or 0 if the token is not recognised
 */
__inline__ __device__ int month_from_name(char const* begin,
                                          char const* end,
                                          cudf::string_view const* locale_names = nullptr)
{
  auto const len = static_cast<cudf::size_type>(end - begin);

  if (locale_names != nullptr) {
    // Locale-aware path: compare the token case-insensitively against every abbreviated
    // (indices 0–11) and full (indices 12–23) locale month name.
    for (int i = 0; i < 12; ++i) {
      // Check abbreviated name first (shorter, so try first to avoid a prefix match
      // of, e.g., "Mar" matching the full name "March").
      for (int pass = 0; pass < 2; ++pass) {
        auto const& sv = locale_names[(pass == 0) ? i : (12 + i)];
        if (sv.size_bytes() == 0 || sv.size_bytes() != len) { continue; }
        bool match = true;
        for (cudf::size_type j = 0; j < len && match; ++j) {
          // Case-insensitive ASCII comparison via | 0x20 (works for ASCII A-Z only;
          // locale names that require multi-byte case-folding are compared as-is).
          match = ((begin[j] | 0x20) == (sv.data()[j] | 0x20));
        }
        if (match) { return i + 1; }
      }
    }
    return 0;
  }

  // English ASCII fallback: match on the first 3 characters only, so both abbreviated
  // ("Jan") and full ("January") names are recognised with a single switch.
  // The | 0x20 trick folds ASCII upper-case to lower-case; it is acceptable here
  // because English month names are always ASCII.
  if (len < 3) { return 0; }
  char const c0 = (*begin | 0x20);
  char const c1 = (*(begin + 1) | 0x20);
  char const c2 = (*(begin + 2) | 0x20);
  switch (c0) {
    case 'j':
      if (c1 == 'a' && c2 == 'n') return 1;  // Jan / January
      if (c1 == 'u' && c2 == 'n') return 6;  // Jun / June
      if (c1 == 'u' && c2 == 'l') return 7;  // Jul / July
      break;
    case 'f':
      if (c1 == 'e' && c2 == 'b') return 2;  // Feb / February
      break;
    case 'm':
      if (c1 == 'a' && c2 == 'r') return 3;  // Mar / March
      if (c1 == 'a' && c2 == 'y') return 5;  // May
      break;
    case 'a':
      if (c1 == 'p' && c2 == 'r') return 4;  // Apr / April
      if (c1 == 'u' && c2 == 'g') return 8;  // Aug / August
      break;
    case 's':
      if (c1 == 'e' && c2 == 'p') return 9;  // Sep / September
      break;
    case 'o':
      if (c1 == 'c' && c2 == 't') return 10;  // Oct / October
      break;
    case 'n':
      if (c1 == 'o' && c2 == 'v') return 11;  // Nov / November
      break;
    case 'd':
      if (c1 == 'e' && c2 == 'c') return 12;  // Dec / December
      break;
  }
  return 0;
}

/**
 * @brief Extracts the Day, Month, and Year from a string.
 *
 * This function takes a string and produces a `year_month_day` representation.
 * Acceptable formats are a combination of `YYYY`, `M`, `MM`, `D` and `DD` with
 * `/` or `-` as separators, as well as space-separated formats with month names.
 * Specifically, the following formats are supported:
 *   - Numeric: `YYYY/MM/DD`, `DD/MM/YYYY`, `MM/DD/YYYY` (same with `-`)
 *   - Year+month only: `YYYY-MM`, `MM/YYYY` (day defaults to 1)
 *   - Named month with `-`: `DD-Mon-YYYY`, `DD-MonthName-YYYY`
 *   - Named month with space: `D MonthName YYYY`, `MonthName YYYY`, `D Mon`
 *
 * @param begin Pointer to the first element of the string
 * @param end Pointer to the first element after the string
 * @param dayfirst Flag indicating that first field is the day
 * @return Extracted year, month and day in `cuda::std::chrono::year_month_day` format
 */
__inline__ __device__ cuda::std::chrono::year_month_day extract_date(
  char const* begin,
  char const* end,
  bool dayfirst,
  cudf::string_view const* locale_names = nullptr)
{
  using namespace cuda::std::chrono;

  char sep = '/';

  auto sep_pos = thrust::find(thrust::seq, begin, end, sep);

  if (sep_pos == end) {
    sep     = '-';
    sep_pos = thrust::find(thrust::seq, begin, end, sep);
  }

  year y;
  month m;
  day d;

  if (sep_pos == end) {
    // No '/' or '-' found: try space-separated format with named months.
    // Supported: "D MonthName [YYYY]", "MonthName YYYY"
    auto const space_pos = thrust::find(thrust::seq, begin, end, ' ');
    if (space_pos == end) {
      // Single token with no separator — can't parse as a meaningful date
      return year_month_day{year{1}, month{1}, day{1}};
    }

    int const first_month = month_from_name(begin, space_pos, locale_names);
    if (first_month > 0) {
      // "MonthName YYYY" — no day, default to 1
      m = month{static_cast<uint32_t>(first_month)};
      d = day{1};
      // Bound the year to the next space so that trailing content (e.g. a time
      // component like "10:30:00") is not absorbed by to_non_negative_integer.
      auto const year_end = thrust::find(thrust::seq, space_pos + 1, end, ' ');
      y                   = year{to_non_negative_integer<int32_t>(space_pos + 1, year_end)};
    } else {
      // "D MonthName [YYYY]"
      d                     = day{to_non_negative_integer<uint32_t>(begin, space_pos)};
      auto const s2         = space_pos + 1;
      auto const space_pos2 = thrust::find(thrust::seq, s2, end, ' ');
      int const named_m     = month_from_name(s2, space_pos2, locale_names);
      m                     = month{(named_m > 0) ? static_cast<uint32_t>(named_m) : 0u};
      if (space_pos2 == end) {
        // "D Mon" with no year — default to year 1 (matches pandas behaviour)
        y = year{1};
      } else {
        // Bound the year token to the next space so that trailing content (e.g. a time
        // component like "10:30:00") is not absorbed by to_non_negative_integer.
        auto const space_pos3 = thrust::find(thrust::seq, space_pos2 + 1, end, ' ');
        y                     = year{to_non_negative_integer<int32_t>(space_pos2 + 1, space_pos3)};
      }
    }
    return year_month_day{y, m, d};
  }

  //--- is year the first field?
  if ((sep_pos - begin) == 4) {
    y = year{to_non_negative_integer<int32_t>(begin, sep_pos)};  //  year is signed

    // Month
    auto s2 = sep_pos + 1;
    sep_pos = thrust::find(thrust::seq, s2, end, sep);

    if (sep_pos == end) {
      //--- Data is just Year and Month - no day
      m = month{to_non_negative_integer<uint32_t>(s2, end)};  // month and day are unsigned
      d = day{1};

    } else {
      m = month{to_non_negative_integer<uint32_t>(s2, sep_pos)};
      d = day{to_non_negative_integer<uint32_t>((sep_pos + 1), end)};
    }

  } else {
    //--- if the dayfirst flag is set, then restricts the format options
    if (dayfirst) {
      d = day{to_non_negative_integer<uint32_t>(begin, sep_pos)};

      auto s2        = sep_pos + 1;
      sep_pos        = thrust::find(thrust::seq, s2, end, sep);
      auto month_end = (sep_pos == end) ? end : sep_pos;

      // Support named months: "DD-Mon-YYYY" (e.g. "15-May-2009")
      int const named_m = month_from_name(s2, month_end, locale_names);
      if (named_m > 0) {
        m = month{static_cast<uint32_t>(named_m)};
      } else {
        m = month{to_non_negative_integer<uint32_t>(s2, month_end)};
      }
      y = year{to_non_negative_integer<int32_t>((sep_pos + 1), end)};

    } else {
      // Support named months at the front: "Mon-DD-YYYY"
      int const named_m = month_from_name(begin, sep_pos, locale_names);
      if (named_m > 0) {
        m = month{static_cast<uint32_t>(named_m)};
      } else {
        m = month{to_non_negative_integer<uint32_t>(begin, sep_pos)};
      }

      auto s2 = sep_pos + 1;
      sep_pos = thrust::find(thrust::seq, s2, end, sep);

      if (sep_pos == end) {
        //--- Data is just Year and Month - no day
        y = year{to_non_negative_integer<int32_t>(s2, end)};
        d = day{1};

      } else {
        d = day{to_non_negative_integer<uint32_t>(s2, sep_pos)};
        y = year{to_non_negative_integer<int32_t>((sep_pos + 1), end)};
      }
    }
  }

  return year_month_day{y, m, d};
}

/**
 * @brief Parses a string to extract the hour, minute, second and sub-second time field
 * values of a day.
 *
 * Incoming format is expected to be `HH:MM:SS.frac`, with the second and fractional fields
 * optional. Each time field can be one or two digits; the fractional part may have 1–9 digits
 * (capped at 6 for microsecond precision). 12-hr and 24-hr time formats are detected via the
 * absence or presence of AM/PM characters at the end.
 *
 * @param begin Pointer to the first element of the string
 * @param end Pointer to the first element after the string
 * @return Extracted hours, minutes, seconds and sub-seconds as `chrono::hh_mm_ss` with
 * microsecond precision
 */
__inline__ __device__ cuda::std::chrono::hh_mm_ss<duration_us> extract_time_of_day(
  char const* begin, char const* end)
{
  constexpr char sep = ':';

  // Adjust for AM/PM and any whitespace before
  duration_h d_h{0};
  auto last = end - 1;
  if (*last == 'M' || *last == 'm') {
    if (*(last - 1) == 'P' || *(last - 1) == 'p') { d_h = duration_h{12}; }
    last = last - 2;
    while (*last == ' ') {
      --last;
    }
  }
  end = last + 1;

  // Find hour-minute separator
  auto const hm_sep = thrust::find(thrust::seq, begin, end, sep);
  // Extract hours
  d_h += cudf::duration_h{to_non_negative_integer<int>(begin, hm_sep)};

  duration_m d_m{0};
  duration_s d_s{0};
  duration_us d_us{0};

  // Find minute-second separator (if present)
  auto const ms_sep = thrust::find(thrust::seq, hm_sep + 1, end, sep);
  if (ms_sep == end) {
    d_m = duration_m{to_non_negative_integer<int32_t>(hm_sep + 1, end)};
  } else {
    d_m = duration_m{to_non_negative_integer<int32_t>(hm_sep + 1, ms_sep)};

    // Find second-fractional separator (if present)
    auto const sms_sep = thrust::find(thrust::seq, ms_sep + 1, end, '.');
    if (sms_sep == end) {
      d_s = duration_s{to_non_negative_integer<int64_t>(ms_sep + 1, end)};
    } else {
      d_s = duration_s{to_non_negative_integer<int64_t>(ms_sep + 1, sms_sep)};

      // Scale fractional digits to microseconds (cap at 6 digits).
      // Count only actual digit characters to avoid including timezone suffixes
      // (e.g. 'Z') in the digit count, which would cause incorrect scaling.
      auto const frac_begin = sms_sep + 1;
      auto frac_end         = frac_begin;
      while (frac_end < end && *frac_end >= '0' && *frac_end <= '9') {
        frac_end++;
      }
      auto const frac_digits = min(static_cast<ptrdiff_t>(6), frac_end - frac_begin);
      auto const raw_frac = to_non_negative_integer<int64_t>(frac_begin, frac_begin + frac_digits);
      // raw_frac has `frac_digits` significant digits; scale up to 6 digits (microseconds)
      int64_t scale = 1;
      for (int i = 0; i < 6 - static_cast<int>(frac_digits); ++i) {
        scale *= 10;
      }
      d_us = duration_us{raw_frac * scale};
    }
  }
  return cuda::std::chrono::hh_mm_ss<duration_us>{d_h + d_m + d_s + d_us};
}

/**
 * @brief Checks whether `c` is decimal digit
 */
__device__ constexpr bool is_digit(char c) { return c >= '0' and c <= '9'; }

/**
 * @brief Parses a datetime string and computes the corresponding timestamp.
 *
 * Acceptable date formats are a combination of `YYYY`, `M`, `MM`, `D` and `DD` with `/` or `-` as
 * separators. Input with only year and month (no day) is also valid. Character `T` or blank space
 * is expected to be the separator between date and time of day. Optional time of day information
 * like hours, minutes, seconds and milliseconds are expected to be `HH:MM:SS.MS`. Each time field
 * can be a single, double, or triple (in the case of milliseconds) digits. 12-hr and 24-hr time
 * format is detected via the absence or presence of AM/PM characters at the end.
 *
 * @tparam timestamp_type Type of output timestamp
 * @param begin Pointer to the first element of the string
 * @param end Pointer to the first element after the string
 * @param dayfirst Flag to indicate day/month or month/day order
 * @return Timestamp converted to `timestamp_type`
 */
template <typename timestamp_type>
__inline__ __device__ timestamp_type to_timestamp(char const* begin,
                                                  char const* end,
                                                  bool dayfirst,
                                                  cudf::string_view const* locale_names = nullptr)
{
  using duration_type = typename timestamp_type::duration;

  auto sep_pos = end;

  // Find end of the date portion
  int count        = 0;
  bool digits_only = true;
  for (auto i = begin; i < end; ++i) {
    digits_only = digits_only and is_digit(*i);
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

  auto ymd = extract_date(begin, sep_pos, dayfirst, locale_names);
  timestamp_type answer{cuda::std::chrono::sys_days{ymd}};

  // Extract time only if separator is present
  if (sep_pos != end) {
    auto t = extract_time_of_day(sep_pos + 1, end);
    answer += cuda::std::chrono::duration_cast<duration_type>(t.to_duration());
  }

  return answer;
}

/**
 * @brief Parses the input string into an integral value of the given type.
 *
 * Moves the `begin` iterator past the parsed value.
 *
 * @param[in, out] begin Pointer to the first element of the string
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
 * @param[in, out] begin Pointer to the first element of the string
 * @param end Pointer to the first element after the string
 * @param delimiter delimiter character
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
 * @brief Finds the first element after the leading space characters.
 *
 * @param begin Pointer to the first element of the string
 * @param end Pointer to the first element after the string
 * @return Pointer to the first character excluding any leading spaces
 */
__inline__ __device__ auto skip_spaces(char const* begin, char const* end)
{
  return thrust::find_if(thrust::seq, begin, end, [](auto elem) { return elem != ' '; });
}

/**
 * @brief Excludes the prefix from the input range if the string starts with the prefix.
 *
 * @tparam N length of the prefix, plus one
 * @param begin Pointer to the first element of the string
 * @param end Pointer to the first element after the string
 * @param prefix String we're searching for at the start of the input range
 * @return Pointer to the start of the string excluding the prefix
 */
template <int N>
__inline__ __device__ auto skip_if_starts_with(char const* begin,
                                               char const* end,
                                               char const (&prefix)[N])
{
  static constexpr size_t prefix_len = N - 1;
  if (end - begin < prefix_len) return begin;
  return thrust::equal(thrust::seq, begin, begin + prefix_len, prefix) ? begin + prefix_len : begin;
}

/**
 * @brief Parses the input string into a duration of `duration_type`.
 *
 * The expected format can be one of the following: `DD days`, `DD days +HH:MM:SS.NS`, `DD days
 * HH:MM::SS.NS`, `HH:MM::SS.NS` and digits-only string. Note `DD` and optional `NS` field can
 * contain arbitrary number of digits while `HH`, `MM` and `SS` can be single or double digits.
 *
 * @tparam duration_type Type of the parsed duration
 * @param begin Pointer to the first element of the string
 * @param end Pointer to the first element after the string
 * @return The parsed duration in `duration_type`
 */
template <typename duration_type>
__inline__ __device__ duration_type to_duration(char const* begin, char const* end)
{
  using cuda::std::chrono::duration_cast;

  // %d days [+]%H:%M:%S.n => %d days, %d days [+]%H:%M:%S,  %H:%M:%S.n, %H:%M:%S, %value.
  constexpr char sep = ':';

  // single pass to parse days, hour, minute, seconds, nanosecond
  auto cur         = begin;
  auto const value = parse_integer<int32_t>(&cur, end);
  cur              = skip_spaces(cur, end);
  if (std::is_same_v<duration_type, cudf::duration_D> || cur >= end) {
    return duration_type{static_cast<typename duration_type::rep>(value)};
  }

  // " days [+]"
  auto const after_days_sep     = skip_if_starts_with(cur, end, "days");
  auto const has_days_seperator = (after_days_sep != cur);
  cur                           = skip_spaces(after_days_sep, end);
  cur += (*cur == '+');

  duration_D d_d{0};
  duration_h d_h{0};
  if (has_days_seperator) {
    d_d = duration_D{value};
    d_h = duration_h{parse_integer<int32_t>(&cur, end)};
  } else {
    d_h = duration_h{value};
  }

  duration_m d_m{parse_optional_integer<int32_t>(&cur, end, sep)};
  duration_s d_s{parse_optional_integer<int64_t>(&cur, end, sep)};

  // Convert all durations to the given type
  auto output_d = duration_cast<duration_type>(d_d + d_h + d_m + d_s);

  if constexpr (std::is_same_v<duration_type, cudf::duration_s>) { return output_d; }

  auto const d_ns = (*cur != '.') ? duration_ns{0} : [&]() {
    auto const start_subsecond     = ++cur;
    auto const unscaled_subseconds = parse_integer<int64_t>(&cur, end);
    auto const scale               = min(9L, cur - start_subsecond) - 9;
    auto const rescaled = numeric::decimal64{unscaled_subseconds, numeric::scale_type{scale}};
    return duration_ns{rescaled.value()};
  }();

  return output_d + duration_cast<duration_type>(d_ns);
}

}  // namespace io
}  // namespace cudf
