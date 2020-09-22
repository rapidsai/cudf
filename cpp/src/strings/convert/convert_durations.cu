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
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/types.hpp>
#include <strings/convert/utilities.cuh>
#include <strings/utilities.cuh>

#include <thrust/transform_reduce.h>
#include <map>
#include <rmm/device_uvector.hpp>
#include <vector>

namespace cudf {
namespace strings {
namespace detail {

namespace {

// duration components timeparts structure
struct alignas(4) duration_component {
  int32_t day;        //-2,147,483,648 to 2,147,483,647
  int32_t subsecond;  // 000000000 to 999999999
  int8_t hour;        // 00 to 23
  int8_t minute;      // 00 to 59
  int8_t second;      // 00 to 59
  bool is_negative;   // true/false
};

enum class format_char_type : int8_t {
  literal,   // literal char type passed through
  specifier  // duration format specifier
};

/**
 * @brief Represents a format specifier or literal from a duration format string.
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
 * @brief The format_compiler parses a duration format string into a vector of
 * format_items.
 *
 * The vector of format_items are used when parsing a string into duration
 * components and when formatting a string from duration components.
 */
struct format_compiler {
  std::string format;
  rmm::device_uvector<format_item> d_items;
  format_compiler(const char* format_, cudaStream_t stream) : format(format_), d_items(0, stream)
  {
    static std::map<char, int8_t> const specifier_lengths = {
      {'-', -1},  // '-' if negative
      {'D', -1},  // 1 to 11 (not in std::format)
      {'H', 2},   // HH
      {'I', 2},   // HH
      {'M', 2},   // MM
      {'S', -1},  // 2 to 13 SS[.mmm][uuu][nnn] (uuu,nnn are not in std::format)
      {'p', 2},   // AM/PM
      {'R', 5},   // 5 HH:MM
      {'T', 8},   // 8 HH:MM:SS"
      {'r', 11}   // HH:MM:SS AM/PM
    };
    std::vector<format_item> items;
    const char* str = format.c_str();
    auto length     = format.length();
    bool negative_sign{true};
    while (length > 0) {
      char ch = *str++;
      length--;
      if (ch != '%') {
        items.push_back(format_item::new_delimiter(ch));
        continue;
      }
      CUDF_EXPECTS(length > 0, "Unfinished specifier in duration format");

      ch = *str++;
      length--;
      if (ch == '%')  // escaped % char
      {
        items.push_back(format_item::new_delimiter(ch));
        continue;
      } else if (ch == 'n') {
        items.push_back(format_item::new_delimiter('\n'));
        continue;
      } else if (ch == 't') {
        items.push_back(format_item::new_delimiter('\t'));
        continue;
      }
      if (ch == 'O') {
        CUDF_EXPECTS(*str == 'H' || *str == 'I' || *str == 'M' || *str == 'S',
                     "locale's alternative representation not supported for specifier: " +
                       std::string(1, *str));
        ch = *str++;
        length--;
        items.push_back(format_item::new_specifier(ch, 2));  // without sign
        continue;
      }
      CUDF_EXPECTS(specifier_lengths.find(ch) != specifier_lengths.end(),
                   "invalid format specifier: " + std::string(1, ch));

      // negative sign should be present only once.
      if (negative_sign) {
        if (std::string("DHIMSRT").find_first_of(ch) != std::string::npos) {
          items.push_back(format_item::new_specifier('-', specifier_lengths.at('-')));
          negative_sign = false;
        }
      }

      int8_t spec_length = specifier_lengths.at(ch);
      items.push_back(format_item::new_specifier(ch, spec_length));
    }

    // create program in device memory
    d_items.resize(items.size(), stream);
    CUDA_TRY(cudaMemcpyAsync(d_items.data(),
                             items.data(),
                             items.size() * sizeof(items[0]),
                             cudaMemcpyHostToDevice,
                             stream));
  }

  format_item const* compiled_format_items() { return d_items.data(); }

  size_type items_count() const { return static_cast<size_type>(d_items.size()); }
};

template <typename T>
__device__ void dissect_duration(T duration, duration_component* timeparts)
{
  timeparts->is_negative = (duration < T{0});
  timeparts->day         = simt::std::chrono::duration_cast<duration_D>(duration).count();

  if (simt::std::is_same<T, duration_D>::value) return;

  duration_s seconds = simt::std::chrono::duration_cast<duration_s>(duration);
  timeparts->hour =
    (simt::std::chrono::duration_cast<simt::std::chrono::hours>(seconds) % duration_D(1)).count();
  timeparts->minute = (simt::std::chrono::duration_cast<simt::std::chrono::minutes>(seconds) %
                       simt::std::chrono::hours(1))
                        .count();
  timeparts->second = (seconds % simt::std::chrono::minutes(1)).count();
  if (not simt::std::is_same<T, duration_s>::value) {
    timeparts->subsecond = (duration % duration_s(1)).count();
  }
}

template <typename T>
struct duration_to_string_size_fn {
  const column_device_view d_durations;
  const format_item* d_format_items;
  size_type items_count;

  __device__ int8_t format_length(char format_char, duration_component const* const timeparts) const
  {
    switch (format_char) {
      case '-': return timeparts->is_negative; break;
      case 'D': return count_digits(timeparts->day) - (timeparts->day < 0); break;
      case 'S':
        return 2 + (timeparts->subsecond == 0 ? 0 : [] {
                 if (simt::std::is_same<T, duration_ms>::value) return 3 + 1;  // +1 is for dot
                 if (simt::std::is_same<T, duration_us>::value) return 6 + 1;  // +1 is for dot
                 if (simt::std::is_same<T, duration_ns>::value) return 9 + 1;  // +1 is for dot
                 return 0;
               }());
        break;
      default: return 2;
    }
  }

  __device__ size_type operator()(size_type idx)
  {
    if (d_durations.is_null(idx)) return 0;
    auto duration                = d_durations.element<T>(idx);
    duration_component timeparts = {0};  // days, hours, minutes, seconds, subseconds(9)
    dissect_duration(duration, &timeparts);
    return thrust::transform_reduce(
      thrust::seq,
      d_format_items,
      d_format_items + items_count,
      [this, &timeparts] __device__(format_item item) -> size_type {
        if (item.item_type == format_char_type::literal)
          return 1;
        else if (item.length != -1)
          return item.length;
        else
          return format_length(item.value, &timeparts);
      },
      size_type{0},
      thrust::plus<size_type>());
  }
};

template <typename T>
struct duration_to_string_fn : public duration_to_string_size_fn<T> {
  const int32_t* d_offsets;
  char* d_chars;
  using duration_to_string_size_fn<T>::d_durations;
  using duration_to_string_size_fn<T>::d_format_items;
  using duration_to_string_size_fn<T>::items_count;

  duration_to_string_fn(const column_device_view d_durations,
                        const format_item* d_format_items,
                        size_type items_count,
                        const int32_t* d_offsets,
                        char* d_chars)
    : duration_to_string_size_fn<T>{d_durations, d_format_items, items_count},
      d_offsets(d_offsets),
      d_chars(d_chars)
  {
  }

  // utility to create (optionally) 0-padded integers (up to 10 chars) without negative sign.
  // min_digits==-1 indicates no 0-padding.
  __device__ char* int2str(char* str, int min_digits, int32_t value)
  {
    constexpr int MAX_DIGITS = 10;  // largest 32-bit integer is 10 digits
    assert(min_digits <= MAX_DIGITS);
    if (value == 0) {
      do {
        *str++ = '0';
      } while (--min_digits > 0);
      return str;
    }

    char digits[MAX_DIGITS] = {'0', '0', '0', '0', '0', '0', '0', '0', '0', '0'};
    int digits_idx          = 0;
    while (value != 0) {
      assert(digits_idx < MAX_DIGITS);
      digits[digits_idx++] = '0' + std::abs(value % 10);
      // next digit
      value = value / 10;
    }
    digits_idx = std::max(digits_idx, min_digits);
    // digits are backwards, reverse the string into the output
    while (digits_idx-- > 0) *str++ = digits[digits_idx];
    return str;
  }

  __device__ char* int_to_2digitstr(char* str, int min_digits, int8_t value)
  {
    assert(value >= -99 && value <= 99);
    value  = std::abs(value);
    str[0] = '0' + value / 10;
    str[1] = '0' + value % 10;
    return str + 2;
  }

  inline __device__ char* day(char* ptr, duration_component const* timeparts)
  {
    return int2str(ptr, -1, timeparts->day);
  }

  inline __device__ char* hour_12(char* ptr, duration_component const* timeparts)
  {
    return int_to_2digitstr(ptr, 2, timeparts->hour % 12);
  }
  inline __device__ char* hour_24(char* ptr, duration_component const* timeparts)
  {
    return int_to_2digitstr(ptr, 2, timeparts->hour);
  }
  inline __device__ char* am_or_pm(char* ptr, duration_component const* timeparts)
  {
    *ptr++ = (timeparts->hour / 12 == 0 ? 'A' : 'P');
    *ptr++ = 'M';
    return ptr;
  }
  inline __device__ char* minute(char* ptr, duration_component const* timeparts)
  {
    return int_to_2digitstr(ptr, 2, timeparts->minute);
  }
  inline __device__ char* second(char* ptr, duration_component const* timeparts)
  {
    return int_to_2digitstr(ptr, 2, timeparts->second);
  }

  inline __device__ char* subsecond(char* ptr, duration_component const* timeparts)
  {
    if (timeparts->subsecond == 0) return ptr;
    const int digits = duration_to_string_size_fn<T>::format_length('S', timeparts) - 3;
    *ptr             = '.';
    auto value       = timeparts->subsecond;
    for (int idx = digits; idx > 0; idx--) {
      *(ptr + idx) = '0' + std::abs(value % 10);
      value /= 10;
    }
    return ptr + digits + 1;
  }

  __device__ char* format_from_parts(duration_component const* timeparts, char* ptr)
  {
    for (size_t idx = 0; idx < items_count; ++idx) {
      auto item = d_format_items[idx];
      if (item.item_type == format_char_type::literal) {
        *ptr++ = item.value;
        continue;
      }
      // special logic for each specifier
      switch (item.value) {
        case 'D':  // days
          ptr = day(ptr, timeparts);
          break;
        case '-':  // - if value is negative
          if (timeparts->is_negative) *ptr++ = '-';
          break;
        case 'H':  // 24-hour
          ptr = hour_24(ptr, timeparts);
          break;
        case 'I':  // 12-hour
          ptr = hour_12(ptr, timeparts);
          break;
        case 'M':  // minute
          ptr = minute(ptr, timeparts);
          break;
        case 'S':  // second
          ptr = second(ptr, timeparts);
          if (item.length == 2) break;
        case 'f':  // sub-second
          ptr = subsecond(ptr, timeparts);
          break;
        case 'p': ptr = am_or_pm(ptr, timeparts); break;
        case 'R':  // HH:MM 24-hour
          ptr    = hour_24(ptr, timeparts);
          *ptr++ = ':';
          ptr    = minute(ptr, timeparts);
          break;
        case 'T':  // HH:MM:SS 24-hour
          ptr    = hour_24(ptr, timeparts);
          *ptr++ = ':';
          ptr    = minute(ptr, timeparts);
          *ptr++ = ':';
          ptr    = second(ptr, timeparts);
          break;
        case 'r':  // HH:MM:SS AM/PM 12-hour
          ptr    = hour_12(ptr, timeparts);
          *ptr++ = ':';
          ptr    = minute(ptr, timeparts);
          *ptr++ = ':';
          ptr    = second(ptr, timeparts);
          *ptr++ = ' ';
          ptr    = am_or_pm(ptr, timeparts);
          break;
        default:  // ignore everything else
          break;
      }
    }
    return ptr;
  }

  __device__ void operator()(size_type idx)
  {
    if (d_durations.is_null(idx)) return;
    auto duration                = d_durations.template element<T>(idx);
    duration_component timeparts = {0};  // days, hours, minutes, seconds, subseconds(9)
    dissect_duration(duration, &timeparts);
    // convert to characters
    format_from_parts(&timeparts, d_chars + d_offsets[idx]);
  }
};

/**
 * @brief This dispatch method is for converting durations into strings.
 *
 * The template function declaration ensures only duration types are used.
 */
struct dispatch_from_durations_fn {
  template <typename T, std::enable_if_t<cudf::is_duration<T>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& durations,
                                     std::string const& format,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream) const
  {
    CUDF_EXPECTS(!format.empty(), "Format parameter must not be empty.");

    format_compiler compiler(format.c_str(), stream);
    auto d_format_items = compiler.compiled_format_items();

    size_type strings_count = durations.size();
    auto column             = column_device_view::create(durations, stream);
    auto d_column           = *column;

    // copy null mask
    rmm::device_buffer null_mask = copy_bitmask(durations, stream, mr);
    // build offsets column
    auto offsets_transformer_itr = thrust::make_transform_iterator(
      thrust::make_counting_iterator<int32_t>(0),
      duration_to_string_size_fn<T>{d_column, d_format_items, compiler.items_count()});
    auto offsets_column = detail::make_offsets_child_column(
      offsets_transformer_itr, offsets_transformer_itr + strings_count, mr, stream);
    auto offsets_view  = offsets_column->view();
    auto d_new_offsets = offsets_view.template data<int32_t>();

    // build chars column
    auto const chars_bytes =
      cudf::detail::get_value<int32_t>(offsets_column->view(), strings_count, stream);
    auto chars_column = detail::create_chars_child_column(
      strings_count, durations.null_count(), chars_bytes, mr, stream);
    auto chars_view = chars_column->mutable_view();
    auto d_chars    = chars_view.template data<char>();

    thrust::for_each_n(rmm::exec_policy(stream)->on(stream),
                       thrust::make_counting_iterator<size_type>(0),
                       strings_count,
                       duration_to_string_fn<T>{
                         d_column, d_format_items, compiler.items_count(), d_new_offsets, d_chars});

    //
    return make_strings_column(strings_count,
                               std::move(offsets_column),
                               std::move(chars_column),
                               durations.null_count(),
                               std::move(null_mask),
                               stream,
                               mr);
  }

  // non-duration types throw an exception
  template <typename T, std::enable_if_t<not cudf::is_duration<T>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const&,
                                     std::string const& format,
                                     rmm::mr::device_memory_resource*,
                                     cudaStream_t) const
  {
    CUDF_FAIL("Values for from_durations function must be a duration type.");
  }
};

static const __device__ __constant__ int32_t powers_of_ten[10] = {
  1L, 10L, 100L, 1000L, 10000L, 100000L, 1000000L, 10000000L, 100000000L, 1000000000L};

// this parses duration string into a duration integer
template <typename T>  // duration type
struct parse_duration {
  column_device_view const d_strings;
  format_item const* d_format_items;
  size_type items_count;

  // function to parse string (maximum 10 digits) to integer.
  __device__ int32_t str2int(const char* str, int8_t max_bytes, int8_t& actual_length)
  {
    const char* ptr = (*str == '-' || *str == '+') ? str + 1 : str;
    int32_t value   = 0;
    for (int8_t idx = 0; idx < max_bytes; ++idx) {
      char chr = *ptr++;
      if (chr < '0' || chr > '9') {
        ptr--;  // roll back
        break;
      }
      value = (value * 10) + static_cast<int32_t>(chr - '0');
    }
    actual_length += (ptr - str);
    return (*str == '-') ? -value : value;
  }

  // function to parse fraction of decimal value with trailing zeros removed.
  __device__ int32_t str2int_fixed(const char* str,
                                   int8_t fixed_width,
                                   size_type string_length,
                                   int8_t& actual_length)
  {
    const char* ptr = (*str == '.') ? str + 1 : str;
    int32_t value   = 0;
    // parse till fixed_width or end of string.
    for (int8_t idx = 0; idx < fixed_width && idx < string_length; ++idx) {
      char chr = *ptr++;
      if (chr < '0' || chr > '9') {
        ptr--;  // roll back
        break;
      }
      value = (value * 10) + static_cast<int32_t>(chr - '0');
    }
    auto parsed_length = ptr - str;
    // compensate for missing trailing zeros
    if (parsed_length < fixed_width) value *= powers_of_ten[fixed_width - parsed_length];
    actual_length += parsed_length;
    return value;
  }

  // parse 2 digit string to integer
  __device__ int8_t parse_2digit_int(const char* str, int8_t& actual_length)
  {
    const char* ptr = (*str == '-' || *str == '+') ? str + 1 : str;
    int8_t value    = 0;
    if (*ptr >= '0' && *ptr <= '9') value = (value * 10) + static_cast<int32_t>(*ptr++ - '0');
    if (*ptr >= '0' && *ptr <= '9') value = (value * 10) + static_cast<int32_t>(*ptr++ - '0');
    actual_length += (ptr - str);
    return (*str == '-') ? -value : value;
  }
  inline __device__ int8_t parse_hour(const char* str, int8_t& actual_length)
  {
    return parse_2digit_int(str, actual_length);
  }
  inline __device__ int8_t parse_minute(const char* str, int8_t& actual_length)
  {
    return parse_2digit_int(str, actual_length);
  }
  inline __device__ int8_t parse_second(const char* str, int8_t& actual_length)
  {
    return parse_2digit_int(str, actual_length);
  }

  // Walk the format_items to read the datetime string.
  // Returns 0 if all ok.
  __device__ int parse_into_parts(string_view const& d_string, duration_component* timeparts)
  {
    auto ptr    = d_string.data();
    auto length = d_string.size_bytes();
    int8_t hour_shift{0};
    for (size_t idx = 0; idx < items_count; ++idx) {
      auto item = d_format_items[idx];
      if (length < item.length) return 1;
      if (item.item_type == format_char_type::literal) {  // static character we'll just skip;
        // consume item.length bytes from string
        ptr += item.length;
        length -= item.length;
        continue;
      }
      timeparts->is_negative |= (*ptr == '-');

      // special logic for each specifier
      int8_t item_length{0};
      switch (item.value) {
        case 'D':  // day
          timeparts->day = str2int(ptr, 11, item_length);
          break;
        case '-':  // skip
          item_length = (*ptr == '-');
          break;
        case 'H':  // 24-hour
          timeparts->hour = parse_hour(ptr, item_length);
          hour_shift      = 0;
          break;
        case 'I':  // 12-hour
          timeparts->hour = parse_hour(ptr, item_length);
          break;
        case 'M':  // minute
          timeparts->minute = parse_minute(ptr, item_length);
          break;
        case 'S':  // [-]SS[.mmm][uuu][nnn]
          timeparts->second = parse_second(ptr, item_length);
          if (*(ptr + item_length) == '.') {
            item_length++;
            int64_t nanoseconds = str2int_fixed(
              ptr + item_length, 9, length - item_length, item_length);  // normalize to nanoseconds
            timeparts->subsecond = nanoseconds;
          }
          break;
        case 'p':  // AM/PM
          if (*ptr == 'P' && *(ptr + 1) == 'M')
            hour_shift = 12;
          else
            hour_shift = 0;
          item_length = 2;
          break;
        case 'R':  // [-]HH:SS
          timeparts->hour = parse_hour(ptr, item_length);
          hour_shift      = 0;
          item_length++;  // :
          timeparts->minute = parse_minute(ptr + item_length, item_length);
          break;
        case 'T':  // [-]HH:MM:SS
          timeparts->hour = parse_hour(ptr, item_length);
          hour_shift      = 0;
          item_length++;  // :
          timeparts->minute = parse_minute(ptr + item_length, item_length);
          item_length++;  // :
          timeparts->second = parse_second(ptr + item_length, item_length);
          break;
        case 'r':  // hh:MM:SS AM/PM
          timeparts->hour = parse_hour(ptr, item_length);
          item_length++;  // :
          timeparts->minute = parse_minute(ptr + item_length, item_length);
          item_length++;  // :
          timeparts->second = parse_second(ptr + item_length, item_length);
          item_length++;  // space
          if (*(ptr + item_length) == 'P' && *(ptr + item_length + 1) == 'M')
            hour_shift = 12;
          else
            hour_shift = 0;
          item_length += 2;
          break;
        default: return 3;
      }
      ptr += item_length;
      length -= item_length;
    }
    // negate all if duration has negative sign
    if (timeparts->is_negative) {
      auto negate          = [](auto i) { return (i < 0 ? i : -i); };
      timeparts->day       = negate(timeparts->day);
      timeparts->hour      = negate(timeparts->hour);
      timeparts->minute    = negate(timeparts->minute);
      timeparts->second    = negate(timeparts->second);
      timeparts->subsecond = negate(timeparts->subsecond);
      hour_shift           = -hour_shift;
    }
    timeparts->hour += hour_shift;
    return 0;
  }

  inline __device__ int64_t duration_from_parts(duration_component const* timeparts)
  {
    int32_t days  = timeparts->day;
    auto hour     = timeparts->hour;
    auto minute   = timeparts->minute;
    auto second   = timeparts->second;
    auto duration = duration_D(days) + simt::std::chrono::hours(hour) +
                    simt::std::chrono::minutes(minute) + duration_s(second);
    if (simt::std::is_same<T, duration_D>::value)
      return simt::std::chrono::duration_cast<duration_D>(duration).count();
    else if (simt::std::is_same<T, duration_s>::value)
      return simt::std::chrono::duration_cast<duration_s>(duration).count();

    duration_ns subsecond(timeparts->subsecond);  // ns
    if (simt::std::is_same<T, duration_ms>::value) {
      return simt::std::chrono::duration_cast<duration_ms>(duration + subsecond).count();
    } else if (simt::std::is_same<T, duration_us>::value) {
      return simt::std::chrono::duration_cast<duration_us>(duration + subsecond).count();
    } else if (simt::std::is_same<T, duration_ns>::value)
      return simt::std::chrono::duration_cast<duration_ns>(duration + subsecond).count();
    return simt::std::chrono::duration_cast<duration_ns>(duration + subsecond).count();
  }

  __device__ T operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) return T{0};
    string_view d_str = d_strings.element<string_view>(idx);
    if (d_str.empty()) return T{0};
    //
    duration_component timeparts = {0};
    if (parse_into_parts(d_str, &timeparts)) return T{0};  // unexpected parse case
    //
    return static_cast<T>(duration_from_parts(&timeparts));
  }
};

/**
 * @brief This dispatch method is for converting strings to durations.
 *
 * The template function declaration ensures only duration types are used.
 */
struct dispatch_to_durations_fn {
  template <typename T, std::enable_if_t<cudf::is_duration<T>()>* = nullptr>
  void operator()(column_device_view const& d_strings,
                  std::string const& format,
                  mutable_column_view& results_view,
                  cudaStream_t stream) const
  {
    format_compiler compiler(format.c_str(), stream);
    auto d_items   = compiler.compiled_format_items();
    auto d_results = results_view.data<T>();
    parse_duration<T> pfn{d_strings, d_items, compiler.items_count()};
    thrust::transform(rmm::exec_policy(stream)->on(stream),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(results_view.size()),
                      d_results,
                      pfn);
  }
  template <typename T, std::enable_if_t<not cudf::is_duration<T>()>* = nullptr>
  void operator()(column_device_view const&,
                  std::string const&,
                  mutable_column_view&,
                  cudaStream_t) const
  {
    CUDF_FAIL("Only durations type are expected for to_durations function");
  }
};

}  // namespace

std::unique_ptr<column> from_durations(column_view const& durations,
                                       std::string const& format,
                                       cudaStream_t stream,
                                       rmm::mr::device_memory_resource* mr)
{
  size_type strings_count = durations.size();
  if (strings_count == 0) return make_empty_strings_column(mr, stream);

  return type_dispatcher(
    durations.type(), dispatch_from_durations_fn{}, durations, format, mr, stream);
}

std::unique_ptr<column> to_durations(strings_column_view const& strings,
                                     data_type duration_type,
                                     std::string const& format,
                                     cudaStream_t stream,
                                     rmm::mr::device_memory_resource* mr)
{
  size_type strings_count = strings.size();
  if (strings_count == 0) return make_duration_column(duration_type, 0);

  CUDF_EXPECTS(!format.empty(), "Format parameter must not be empty.");

  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_column       = *strings_column;

  auto results      = make_duration_column(duration_type,
                                      strings_count,
                                      copy_bitmask(strings.parent(), stream, mr),
                                      strings.null_count(),
                                      stream,
                                      mr);
  auto results_view = results->mutable_view();
  cudf::type_dispatcher(
    duration_type, dispatch_to_durations_fn(), d_column, format, results_view, stream);
  results->set_null_count(strings.null_count());
  return results;
}

}  // namespace detail

std::unique_ptr<column> from_durations(column_view const& durations,
                                       std::string const& format,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::from_durations(durations, format, cudaStream_t{}, mr);
}

std::unique_ptr<column> to_durations(strings_column_view const& strings,
                                     data_type duration_type,
                                     std::string const& format,
                                     rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::to_durations(strings, duration_type, format, cudaStream_t{}, mr);
}

}  // namespace strings
}  // namespace cudf
