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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/convert/convert_datetime.hpp>
#include <cudf/strings/detail/converters.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/wrappers/timestamps.hpp>
#include <strings/utilities.cuh>

#include <rmm/thrust_rmm_allocator.h>
#include <map>
#include <vector>

namespace cudf {
namespace strings {
namespace detail {
namespace {
/**
 * @brief  Units for timestamp conversion.
 * These are defined since there are more than what cudf supports.
 */
enum class timestamp_units {
  years,    ///< precision is years
  months,   ///< precision is months
  days,     ///< precision is days
  hours,    ///< precision is hours
  minutes,  ///< precision is minutes
  seconds,  ///< precision is seconds
  ms,       ///< precision is milliseconds
  us,       ///< precision is microseconds
  ns        ///< precision is nanoseconds
};

// used to index values in a timeparts array
enum timestamp_parse_component {
  TP_YEAR        = 0,
  TP_MONTH       = 1,
  TP_DAY         = 2,
  TP_DAY_OF_YEAR = 3,
  TP_HOUR        = 4,
  TP_MINUTE      = 5,
  TP_SECOND      = 6,
  TP_SUBSECOND   = 7,
  TP_TZ_MINUTES  = 8,
  TP_ARRAYSIZE   = 9
};

enum class format_char_type : int8_t {
  literal,   // literal char type passed through
  specifier  // timestamp format specifier
};

/**
 * @brief Represents a format specifier or literal from a timestamp format string.
 *
 * Created by the format_compiler when parsing a format string.
 */
struct alignas(4) format_item {
  format_char_type item_type;  // specifier or literal indicator
  char value;                  // specifier or literal value
  int8_t length;               // item length in bytes

  static format_item new_specifier(char format_char, int8_t length)
  {
    return format_item{format_char_type::specifier, format_char, length};
  }
  static format_item new_delimiter(char literal)
  {
    return format_item{format_char_type::literal, literal, 1};
  }
};

/**
 * @brief The format_compiler parses a timestamp format string into a vector of
 * format_items.
 *
 * The vector of format_items are used when parsing a string into timestamp
 * components and when formatting a string from timestamp components.
 */
struct format_compiler {
  std::string format;
  std::string template_string;
  timestamp_units units;
  rmm::device_vector<format_item> d_items;

  std::map<char, int8_t> specifier_lengths = {{'Y', 4},
                                              {'y', 2},
                                              {'m', 2},
                                              {'d', 2},
                                              {'H', 2},
                                              {'I', 2},
                                              {'M', 2},
                                              {'S', 2},
                                              {'f', 6},
                                              {'z', 5},
                                              {'Z', 3},
                                              {'p', 2},
                                              {'j', 3}};

  format_compiler(const char* format, timestamp_units units) : format(format), units(units) {}

  format_item const* compile_to_device()
  {
    std::vector<format_item> items;
    const char* str = format.c_str();
    auto length     = format.length();
    while (length > 0) {
      char ch = *str++;
      length--;
      if (ch != '%') {
        items.push_back(format_item::new_delimiter(ch));
        template_string.append(1, ch);
        continue;
      }
      CUDF_EXPECTS(length > 0, "Unfinished specifier in timestamp format");

      ch = *str++;
      length--;
      if (ch == '%')  // escaped % char
      {
        items.push_back(format_item::new_delimiter(ch));
        template_string.append(1, ch);
        continue;
      }
      if (ch >= '0' && ch <= '9') {
        CUDF_EXPECTS(*str == 'f', "precision not supported for specifier: " + std::string(1, *str));
        specifier_lengths[*str] = static_cast<int8_t>(ch - '0');
        ch                      = *str++;
        length--;
      }
      CUDF_EXPECTS(specifier_lengths.find(ch) != specifier_lengths.end(),
                   "invalid format specifier: " + std::string(1, ch));

      int8_t spec_length = specifier_lengths[ch];
      items.push_back(format_item::new_specifier(ch, spec_length));
      template_string.append((size_t)spec_length, ch);
    }
    // create program in device memory
    d_items.resize(items.size());
    CUDA_TRY(cudaMemcpyAsync(
      d_items.data().get(), items.data(), items.size() * sizeof(items[0]), cudaMemcpyHostToDevice));
    return d_items.data().get();
  }

  // these calls are only valid after compile_to_device is called
  size_type template_bytes() const { return static_cast<size_type>(template_string.size()); }
  size_type items_count() const { return static_cast<size_type>(d_items.size()); }
  int8_t subsecond_precision() const { return specifier_lengths.at('f'); }
};

// this parses date/time characters into a timestamp integer
template <typename T>  // timestamp type
struct parse_datetime {
  column_device_view const d_strings;
  format_item const* d_format_items;
  size_type items_count;
  timestamp_units units;
  int8_t subsecond_precision;

  //
  __device__ int32_t str2int(const char* str, size_type bytes)
  {
    const char* ptr = str;
    int32_t value   = 0;
    for (size_type idx = 0; idx < bytes; ++idx) {
      char chr = *ptr++;
      if (chr < '0' || chr > '9') break;
      value = (value * 10) + static_cast<int32_t>(chr - '0');
    }
    return value;
  }

  // Walk the format_items to read the datetime string.
  // Returns 0 if all ok.
  __device__ int parse_into_parts(string_view const& d_string, int32_t* timeparts)
  {
    auto ptr    = d_string.data();
    auto length = d_string.size_bytes();
    for (size_t idx = 0; idx < items_count; ++idx) {
      auto item = d_format_items[idx];
      if (length < item.length) return 1;
      if (item.item_type == format_char_type::literal) {  // static character we'll just skip;
        // consume item.length bytes from string
        ptr += item.length;
        length -= item.length;
        continue;
      }

      // special logic for each specifier
      switch (item.value) {
        case 'Y': timeparts[TP_YEAR] = str2int(ptr, item.length); break;
        case 'y': timeparts[TP_YEAR] = str2int(ptr, item.length) + 1900; break;
        case 'm': timeparts[TP_MONTH] = str2int(ptr, item.length); break;
        case 'd': timeparts[TP_DAY] = str2int(ptr, item.length); break;
        case 'j': timeparts[TP_DAY_OF_YEAR] = str2int(ptr, item.length); break;
        case 'H':
        case 'I': timeparts[TP_HOUR] = str2int(ptr, item.length); break;
        case 'M': timeparts[TP_MINUTE] = str2int(ptr, item.length); break;
        case 'S': timeparts[TP_SECOND] = str2int(ptr, item.length); break;
        case 'f': timeparts[TP_SUBSECOND] = str2int(ptr, item.length); break;
        case 'p': {
          string_view am_pm(ptr, 2);
          auto hour = timeparts[TP_HOUR];
          if ((am_pm.compare("AM", 2) == 0) || (am_pm.compare("am", 2) == 0)) {
            if (hour == 12) hour = 0;
          } else if (hour < 12)
            hour += 12;
          timeparts[TP_HOUR] = hour;
          break;
        }
        case 'z': {
          int sign = *ptr == '-' ? 1 : -1;  // revert timezone back to UTC
          int hh   = str2int(ptr + 1, 2);
          int mm   = str2int(ptr + 3, 2);
          // ignoring the rest for now
          // item.length has how many chars we should read
          timeparts[TP_TZ_MINUTES] = sign * ((hh * 60) + mm);
          break;
        }
        case 'Z': break;  // skip
        default: return 3;
      }
      ptr += item.length;
      length -= item.length;
    }
    return 0;
  }

  __device__ int64_t timestamp_from_parts(int32_t const* timeparts, timestamp_units units)
  {
    auto year = timeparts[TP_YEAR];
    if (units == timestamp_units::years) return year - 1970;
    auto month = timeparts[TP_MONTH];
    if (units == timestamp_units::months)
      return ((year - 1970) * 12) + (month - 1);  // months are 1-12, need to 0-base it here
    auto day = timeparts[TP_DAY];
    // The months are shifted so that March is the starting month and February
    // (possible leap day in it) is the last month for the linear calculation
    year -= (month <= 2) ? 1 : 0;
    // date cycle repeats every 400 years (era)
    constexpr int32_t erasInDays  = 146097;
    constexpr int32_t erasInYears = (erasInDays / 365);
    auto era                      = (year >= 0 ? year : year - 399) / erasInYears;
    auto yoe                      = year - era * erasInYears;
    auto doy = month == 0 ? day : ((153 * (month + (month > 2 ? -3 : 9)) + 2) / 5 + day - 1);
    auto doe = (yoe * 365) + (yoe / 4) - (yoe / 100) + doy;
    int32_t days =
      (era * erasInDays) + doe - 719468;  // 719468 = days from 0000-00-00 to 1970-03-01
    if (units == timestamp_units::days) return days;

    auto tzadjust = timeparts[TP_TZ_MINUTES];  // in minutes
    auto hour     = timeparts[TP_HOUR];
    if (units == timestamp_units::hours) return (days * 24L) + hour + (tzadjust / 60);

    auto minute = timeparts[TP_MINUTE];
    if (units == timestamp_units::minutes)
      return static_cast<int64_t>(days * 24L * 60L) + (hour * 60L) + minute + tzadjust;

    auto second = timeparts[TP_SECOND];
    int64_t timestamp =
      (days * 24L * 3600L) + (hour * 3600L) + (minute * 60L) + second + (tzadjust * 60);
    if (units == timestamp_units::seconds) return timestamp;

    int64_t powers_of_ten[] = {
      1L, 10L, 100L, 1000L, 10000L, 100000L, 1000000L, 10000000L, 100000000L, 1000000000L};
    int64_t subsecond =
      timeparts[TP_SUBSECOND] * powers_of_ten[9 - subsecond_precision];  // normalize to nanoseconds
    if (units == timestamp_units::ms) {
      timestamp *= 1000L;
      subsecond = subsecond / 1000000L;
    } else if (units == timestamp_units::us) {
      timestamp *= 1000000L;
      subsecond = subsecond / 1000L;
    } else if (units == timestamp_units::ns)
      timestamp *= 1000000000L;
    timestamp += subsecond;
    return timestamp;
  }

  __device__ T operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) return 0;
    string_view d_str = d_strings.element<string_view>(idx);
    if (d_str.empty()) return 0;
    //
    int32_t timeparts[TP_ARRAYSIZE] = {0, 1, 1};       // month and day are 1-based
    if (parse_into_parts(d_str, timeparts)) return 0;  // unexpected parse case
    //
    return static_cast<T>(timestamp_from_parts(timeparts, units));
  }
};

// convert cudf type to timestamp units
struct dispatch_timestamp_to_units_fn {
  template <typename T>
  timestamp_units operator()()
  {
    CUDF_FAIL("Invalid type for timestamp conversion.");
  }
};

template <>
timestamp_units dispatch_timestamp_to_units_fn::operator()<cudf::timestamp_D>()
{
  return timestamp_units::days;
}
template <>
timestamp_units dispatch_timestamp_to_units_fn::operator()<cudf::timestamp_s>()
{
  return timestamp_units::seconds;
}
template <>
timestamp_units dispatch_timestamp_to_units_fn::operator()<cudf::timestamp_ms>()
{
  return timestamp_units::ms;
}
template <>
timestamp_units dispatch_timestamp_to_units_fn::operator()<cudf::timestamp_us>()
{
  return timestamp_units::us;
}
template <>
timestamp_units dispatch_timestamp_to_units_fn::operator()<cudf::timestamp_ns>()
{
  return timestamp_units::ns;
}

// dispatch operator to map timestamp to native fixed-width-type
struct dispatch_to_timestamps_fn {
  template <typename T, std::enable_if_t<cudf::is_timestamp<T>()>* = nullptr>
  void operator()(column_device_view const& d_strings,
                  std::string const& format,
                  timestamp_units units,
                  mutable_column_view& results_view,
                  cudaStream_t stream) const
  {
    format_compiler compiler(format.c_str(), units);
    auto d_items   = compiler.compile_to_device();
    auto d_results = results_view.data<T>();
    parse_datetime<T> pfn{
      d_strings, d_items, compiler.items_count(), units, compiler.subsecond_precision()};
    thrust::transform(rmm::exec_policy(stream)->on(stream),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(results_view.size()),
                      d_results,
                      pfn);
  }
  template <typename T, std::enable_if_t<not cudf::is_timestamp<T>()>* = nullptr>
  void operator()(column_device_view const&,
                  std::string const&,
                  timestamp_units,
                  mutable_column_view&,
                  cudaStream_t) const
  {
    CUDF_FAIL("Only timestamps type are expected");
  }
};

}  // namespace

//
std::unique_ptr<cudf::column> to_timestamps(strings_column_view const& strings,
                                            data_type timestamp_type,
                                            std::string const& format,
                                            cudaStream_t stream,
                                            rmm::mr::device_memory_resource* mr)
{
  size_type strings_count = strings.size();
  if (strings_count == 0) return make_timestamp_column(timestamp_type, 0);

  CUDF_EXPECTS(!format.empty(), "Format parameter must not be empty.");
  timestamp_units units = cudf::type_dispatcher(timestamp_type, dispatch_timestamp_to_units_fn());

  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_column       = *strings_column;

  auto results      = make_timestamp_column(timestamp_type,
                                       strings_count,
                                       copy_bitmask(strings.parent(), stream, mr),
                                       strings.null_count(),
                                       stream,
                                       mr);
  auto results_view = results->mutable_view();
  cudf::type_dispatcher(
    timestamp_type, dispatch_to_timestamps_fn(), d_column, format, units, results_view, stream);
  results->set_null_count(strings.null_count());
  return results;
}

}  // namespace detail

// external API

std::unique_ptr<cudf::column> to_timestamps(strings_column_view const& strings,
                                            data_type timestamp_type,
                                            std::string const& format,
                                            rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::to_timestamps(strings, timestamp_type, format, cudaStream_t{}, mr);
}

namespace detail {
namespace {
// converts a timestamp into date-time string
template <typename T>
struct datetime_formatter {
  const column_device_view d_timestamps;
  const format_item* d_format_items;
  size_type items_count;
  timestamp_units units;
  const int32_t* d_offsets;
  char* d_chars;

  // divide timestamp integer into time components (year, month, day, etc)
  // TODO call the simt::std::chrono methods here instead when the are ready
  __device__ void dissect_timestamp(int64_t timestamp, int32_t* timeparts)
  {
    if (units == timestamp_units::years) {
      timeparts[TP_YEAR]  = static_cast<int32_t>(timestamp) + 1970;
      timeparts[TP_MONTH] = 1;
      timeparts[TP_DAY]   = 1;
      return;
    }

    // Specialized modulo expression that handles negative values.
    // Examples:
    //     modulo(1,60)    1
    //     modulo(-1,60)  59
    auto modulo_time = [](int64_t time, int64_t base) {
      return static_cast<int32_t>(((time % base) + base) % base);
    };

    // This function handles converting units by dividing and adjusting for negative values.
    // Examples:
    //     scale(-61,60)  -2
    //     scale(-60,60)  -1
    //     scale(-59,60)  -1
    //     scale( 59,60)   0
    //     scale( 60,60)   1
    //     scale( 61,60)   1
    auto scale_time = [](int64_t time, int64_t base) {
      return static_cast<int32_t>((time - ((time < 0) * (base - 1L))) / base);
    };

    if (units == timestamp_units::months) {
      int32_t month       = modulo_time(timestamp, 12);
      int32_t year        = scale_time(timestamp, 12) + 1970;
      timeparts[TP_YEAR]  = year;
      timeparts[TP_MONTH] = month + 1;  // months start at 1 and not 0
      timeparts[TP_DAY]   = 1;
      return;
    }

    // first, convert to days so we can handle months, leap years, etc.
    int32_t days = static_cast<int32_t>(timestamp);  // default to days
    if (units == timestamp_units::hours)
      days = scale_time(timestamp, 24L);
    else if (units == timestamp_units::minutes)
      days = scale_time(timestamp, 1440L);  // 24*60
    else if (units == timestamp_units::seconds)
      days = scale_time(timestamp, 86400L);  // 24*60*60
    else if (units == timestamp_units::ms)
      days = scale_time(timestamp, 86400000L);
    else if (units == timestamp_units::us)
      days = scale_time(timestamp, 86400000000L);
    else if (units == timestamp_units::ns)
      days = scale_time(timestamp, 86400000000000L);
    days = days + 719468;  // 719468 is days between 0000-00-00 and 1970-01-01

    int32_t const daysInEra     = 146097;  // (400*365)+97
    int32_t const daysInCentury = 36524;   // (100*365) + 24;
    int32_t const daysIn4Years  = 1461;    // (4*365) + 1;
    int32_t const daysInYear    = 365;

    // code logic handles leap years in chunks: 400y,100y,4y,1y
    int32_t year  = 400 * (days / daysInEra);
    days          = days % daysInEra;
    int32_t leapy = days / daysInCentury;
    days          = days % daysInCentury;
    if (leapy == 4) {  // landed exactly on a leap century
      days += daysInCentury;
      --leapy;
    }
    year += 100 * leapy;
    year += 4 * (days / daysIn4Years);
    days  = days % daysIn4Years;
    leapy = days / daysInYear;
    days  = days % daysInYear;
    if (leapy == 4) {  // landed exactly on a leap year
      days += daysInYear;
      --leapy;
    }
    year += leapy;

    // The months are shifted so that March is the starting month and February
    // (with possible leap day in it) is the last month for the linear calculation.
    // Day offsets for each month:   Mar Apr May June July Aug Sep  Oct  Nov  Dec  Jan  Feb
    const int32_t monthDayOffset[] = {0, 31, 61, 92, 122, 153, 184, 214, 245, 275, 306, 337, 366};
    // find month from days
    int32_t month = [days, monthDayOffset] {
      // find first offset that is bigger than days
      auto itr = thrust::find_if(
        thrust::seq, monthDayOffset, (monthDayOffset + 13), [days](auto d) { return days < d; });
      return itr != (monthDayOffset + 13) ? thrust::distance(monthDayOffset, itr - 1) : 12;
    }();

    // compute day of the year and account for calculating with March being the first month
    // for month >= 10, leap-day has been already been included
    timeparts[TP_DAY_OF_YEAR] = (month >= 10)
                                  ? days - monthDayOffset[10] + 1
                                  : days + /*Jan=*/31 + /*Feb=*/28 + 1 +  // 2-month shift
                                      ((year % 4 == 0) && ((year % 100 != 0) || (year % 400 == 0)));

    int32_t const day = days - monthDayOffset[month] + 1;  // compute day of month
    if (month >= 10) ++year;
    month = ((month + 2) % 12) + 1;  // adjust Jan-Mar offset

    timeparts[TP_YEAR]  = year;
    timeparts[TP_MONTH] = month;
    timeparts[TP_DAY]   = day;
    if (units == timestamp_units::days) return;

    // done with date, now work on time

    if (units == timestamp_units::hours) {
      timeparts[TP_HOUR] = modulo_time(timestamp, 24);
      return;
    }
    if (units == timestamp_units::minutes) {
      timeparts[TP_HOUR]   = modulo_time(scale_time(timestamp, 60), 24);
      timeparts[TP_MINUTE] = modulo_time(timestamp, 60);
      return;
    }
    if (units == timestamp_units::seconds) {
      timeparts[TP_HOUR]   = modulo_time(scale_time(timestamp, 3600), 24);
      timeparts[TP_MINUTE] = modulo_time(scale_time(timestamp, 60), 60);
      timeparts[TP_SECOND] = modulo_time(timestamp, 60);
      return;
    }

    // common utility for setting time components from a subsecond unit value
    auto subsecond_fn = [&](int64_t subsecond_base) {
      timeparts[TP_SUBSECOND] = modulo_time(timestamp, subsecond_base);
      timestamp               = timestamp / subsecond_base;
      timeparts[TP_HOUR]      = modulo_time(scale_time(timestamp, 3600), 24);
      timeparts[TP_MINUTE]    = modulo_time(scale_time(timestamp, 60), 60);
      timeparts[TP_SECOND]    = modulo_time(timestamp, 60);
    };

    if (units == timestamp_units::ms)
      subsecond_fn(1000);
    else if (units == timestamp_units::us)
      subsecond_fn(1000000);
    else
      subsecond_fn(1000000000);
  }

  // utility to create 0-padded integers (up to 9 chars)
  __device__ char* int2str(char* str, int bytes, int val)
  {
    char tmpl[9] = {'0', '0', '0', '0', '0', '0', '0', '0', '0'};
    char* ptr    = tmpl;
    while (val > 0) {
      int digit = val % 10;
      *ptr++    = '0' + digit;
      val       = val / 10;
    }
    ptr = tmpl + bytes - 1;
    while (bytes-- > 0) *str++ = *ptr--;
    return str;
  }

  __device__ char* format_from_parts(int32_t const* timeparts, char* ptr)
  {
    for (size_t idx = 0; idx < items_count; ++idx) {
      auto item = d_format_items[idx];
      if (item.item_type == format_char_type::literal) {
        *ptr++ = item.value;
        continue;
      }
      // special logic for each specifier
      switch (item.value) {
        case 'Y':  // 4-digit year
          ptr = int2str(ptr, item.length, timeparts[TP_YEAR]);
          break;
        case 'y':  // 2-digit year
        {
          auto year = timeparts[TP_YEAR];
          // remove hundredths digits and above
          ptr = int2str(ptr, item.length, year - ((year / 100) * 100));
          break;
        }
        case 'm':  // month
          ptr = int2str(ptr, item.length, timeparts[TP_MONTH]);
          break;
        case 'd':  // day of month
          ptr = int2str(ptr, item.length, timeparts[TP_DAY]);
          break;
        case 'j':  // day of year
          ptr = int2str(ptr, item.length, timeparts[TP_DAY_OF_YEAR]);
          break;
        case 'H':  // 24-hour
          ptr = int2str(ptr, item.length, timeparts[TP_HOUR]);
          break;
        case 'I':  // 12-hour
        {
          // 0 = 12am; 12 = 12pm; 6 = 06am; 18 = 06pm
          auto hour = timeparts[TP_HOUR];
          if (hour == 0) hour = 12;
          if (hour > 12) hour -= 12;
          ptr = int2str(ptr, item.length, hour);
          break;
        }
        case 'M':  // minute
          ptr = int2str(ptr, item.length, timeparts[TP_MINUTE]);
          break;
        case 'S':  // second
          ptr = int2str(ptr, item.length, timeparts[TP_SECOND]);
          break;
        case 'f':  // sub-second
        {
          char subsecond_digits[] = "000000000";  // 9 max digits
          const int digits        = [units = units] {
            if (units == timestamp_units::ms) return 3;
            if (units == timestamp_units::us) return 6;
            if (units == timestamp_units::ns) return 9;
            return 0;
          }();
          int2str(subsecond_digits, digits, timeparts[TP_SUBSECOND]);
          ptr = copy_and_increment(ptr, subsecond_digits, item.length);
          break;
        }
        case 'p':  // am or pm
          // 0 = 12am, 12 = 12pm
          if (timeparts[TP_HOUR] < 12)
            memcpy(ptr, "AM", 2);
          else
            memcpy(ptr, "PM", 2);
          ptr += 2;
          break;
        case 'z':                   // timezone
          memcpy(ptr, "+0000", 5);  // always UTC
          ptr += 5;
          break;
        case 'Z':
          memcpy(ptr, "UTC", 3);
          ptr += 3;
          break;
        default:  // ignore everything else
          break;
      }
    }
    return ptr;
  }

  __device__ void operator()(size_type idx)
  {
    if (d_timestamps.is_null(idx)) return;
    auto timestamp                  = d_timestamps.element<T>(idx);
    int32_t timeparts[TP_ARRAYSIZE] = {0};
    dissect_timestamp(timestamp.time_since_epoch().count(), timeparts);
    // convert to characters
    format_from_parts(timeparts, d_chars + d_offsets[idx]);
  }
};

//
struct dispatch_from_timestamps_fn {
  template <typename T, std::enable_if_t<cudf::is_timestamp<T>()>* = nullptr>
  void operator()(column_device_view const& d_timestamps,
                  format_item const* d_format_items,
                  size_type items_count,
                  timestamp_units units,
                  const int32_t* d_offsets,
                  char* d_chars,
                  cudaStream_t stream) const
  {
    datetime_formatter<T> pfn{d_timestamps, d_format_items, items_count, units, d_offsets, d_chars};
    thrust::for_each_n(rmm::exec_policy(stream)->on(stream),
                       thrust::make_counting_iterator<cudf::size_type>(0),
                       d_timestamps.size(),
                       pfn);
  }
  template <typename T, std::enable_if_t<not cudf::is_timestamp<T>()>* = nullptr>
  void operator()(column_device_view const&,
                  format_item const*,
                  size_type,
                  timestamp_units,
                  const int32_t*,
                  char* d_chars,
                  cudaStream_t stream) const
  {
    CUDF_FAIL("Only timestamps type are expected");
  }
};

}  // namespace

//
std::unique_ptr<column> from_timestamps(column_view const& timestamps,
                                        std::string const& format,
                                        cudaStream_t stream,
                                        rmm::mr::device_memory_resource* mr)
{
  size_type strings_count = timestamps.size();
  if (strings_count == 0) return make_empty_strings_column(mr, stream);

  CUDF_EXPECTS(!format.empty(), "Format parameter must not be empty.");
  timestamp_units units =
    cudf::type_dispatcher(timestamps.type(), dispatch_timestamp_to_units_fn());

  format_compiler compiler(format.c_str(), units);
  auto d_format_items = compiler.compile_to_device();

  auto column   = column_device_view::create(timestamps, stream);
  auto d_column = *column;

  // copy null mask
  rmm::device_buffer null_mask = copy_bitmask(timestamps, stream, mr);
  // Each string will be the same number of bytes which can be determined
  // directly from the format string.
  auto d_str_bytes = compiler.template_bytes();  // size in bytes of each string
  // build offsets column
  auto offsets_transformer_itr =
    thrust::make_transform_iterator(thrust::make_counting_iterator<size_type>(0),
                                    [d_column, d_str_bytes] __device__(size_type idx) {
                                      return (d_column.is_null(idx) ? 0 : d_str_bytes);
                                    });
  auto offsets_column = make_offsets_child_column(
    offsets_transformer_itr, offsets_transformer_itr + strings_count, mr, stream);
  auto offsets_view  = offsets_column->view();
  auto d_new_offsets = offsets_view.template data<int32_t>();

  // build chars column
  size_type bytes = thrust::device_pointer_cast(d_new_offsets)[strings_count];
  auto chars_column =
    create_chars_child_column(strings_count, timestamps.null_count(), bytes, mr, stream);
  auto chars_view = chars_column->mutable_view();
  auto d_chars    = chars_view.template data<char>();
  // fill in chars column with timestamps
  // dispatcher is called to handle the different timestamp types
  cudf::type_dispatcher(timestamps.type(),
                        dispatch_from_timestamps_fn(),
                        d_column,
                        d_format_items,
                        compiler.items_count(),
                        units,
                        d_new_offsets,
                        d_chars,
                        stream);
  //
  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             std::move(chars_column),
                             timestamps.null_count(),
                             std::move(null_mask),
                             stream,
                             mr);
}

}  // namespace detail

// external API

std::unique_ptr<column> from_timestamps(column_view const& timestamps,
                                        std::string const& format,
                                        rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::from_timestamps(timestamps, format, cudaStream_t{}, mr);
}

}  // namespace strings
}  // namespace cudf
