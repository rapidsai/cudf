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

// used to index values in a timeparts array
enum duration_parse_component {
  DU_DAY       = 0,
  DU_HOUR      = 1,
  DU_MINUTE    = 2,
  DU_SECOND    = 3,
  DU_SUBSECOND = 4,
  DU_NEGATIVE  = 5,
  DU_ARRAYSIZE = 6
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

  std::map<char, int8_t> specifier_lengths = {
    {'-', -1},  // '-' if negative
    {'D', -1},  // 1 to 11 (not in std::format) // TODO
    {'H', 2},   // HH
    {'I', 2},   // HH
    {'M', 2},   // MM
    {'S', -1},  // 2 to 13 SS[.mmm][uuu][nnn] (uuu,nnn are not in std::format)
    {'p', 2},   // AM/PM
    {'R', 5},   // 5 HH:MM
    {'T', 8},   // 8 HH:MM:SS"
    {'r', 11}   // HH:MM:SS AM/PM
  };
  format_compiler(const char* format_, cudaStream_t stream) : format(format_), d_items(0, stream)
  {
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
      }
      if (ch == 'O') {
        CUDF_EXPECTS(*str == 'H' || *str == 'M' || *str == 'S',
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
          items.push_back(format_item::new_specifier('-', specifier_lengths['-']));
          negative_sign = false;
        }
      }

      int8_t spec_length = specifier_lengths[ch];
      items.push_back(format_item::new_specifier(ch, spec_length));
    }
    for (auto item : items) {
      std::cout << int(item.item_type) << ":" << item.value << ":" << int(item.length) << "\n";
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

__device__ void dissect_duration(int64_t duration, int32_t* timeparts, type_id units)
{
  timeparts[DU_NEGATIVE] = (duration < 0);
  if (units == type_id::DURATION_DAYS) {
    timeparts[DU_DAY] = static_cast<int32_t>(duration);
    return;
  }

  duration_s seconds{0};
  if (units == type_id::DURATION_SECONDS) {
    seconds = duration_s(duration);
  } else if (units == type_id::DURATION_MILLISECONDS) {
    seconds                 = simt::std::chrono::duration_cast<duration_s>(duration_ms(duration));
    timeparts[DU_SUBSECOND] = (duration_ms(duration) % duration_s(1)).count();
  } else if (units == type_id::DURATION_MICROSECONDS) {
    seconds                 = simt::std::chrono::duration_cast<duration_s>(duration_us(duration));
    timeparts[DU_SUBSECOND] = (duration_us(duration) % duration_s(1)).count();
  } else if (units == type_id::DURATION_NANOSECONDS) {
    seconds                 = simt::std::chrono::duration_cast<duration_s>(duration_ns(duration));
    timeparts[DU_SUBSECOND] = (duration_ns(duration) % duration_s(1)).count();
  }
  timeparts[DU_DAY] = simt::std::chrono::duration_cast<duration_D>(seconds).count();
  timeparts[DU_HOUR] =
    (simt::std::chrono::duration_cast<simt::std::chrono::hours>(seconds) % duration_D(1)).count();
  timeparts[DU_MINUTE] = (simt::std::chrono::duration_cast<simt::std::chrono::minutes>(seconds) %
                          simt::std::chrono::hours(1))
                           .count();
  timeparts[DU_SECOND] = (seconds % simt::std::chrono::minutes(1)).count();
}

template <typename T>
struct duration_to_string_size_fn {
  const column_device_view d_durations;
  const format_item* d_format_items;
  size_type items_count;
  type_id type;

  __device__ int8_t format_length(char format_char, int32_t const* const timeparts) const
  {
    switch (format_char) {
      case '-': return timeparts[DU_NEGATIVE]; break;
      case 'D': return count_digits(timeparts[DU_DAY]) - (timeparts[DU_DAY] < 0); break;
      case 'S':
        return 2 + (timeparts[DU_SUBSECOND] == 0 ? 0 : [] {
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
    auto duration                   = d_durations.element<T>(idx);
    int32_t timeparts[DU_ARRAYSIZE] = {0};  // days, hours, minutes, seconds, subseconds(9)
    dissect_duration(duration.count(), timeparts, type);
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
          return format_length(item.value, timeparts);
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
  using duration_to_string_size_fn<T>::type;

  duration_to_string_fn(const column_device_view d_durations,
                        const format_item* d_format_items,
                        size_type items_count,
                        type_id type,
                        const int32_t* d_offsets,
                        char* d_chars)
    : duration_to_string_size_fn<T>{d_durations, d_format_items, items_count, type},
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

  inline __device__ char* day(char* ptr, int32_t const* timeparts)
  {
    return int2str(ptr, -1, timeparts[DU_DAY]);
  }

  inline __device__ char* hour_12(char* ptr, int32_t const* timeparts)
  {
    return int2str(ptr, 2, timeparts[DU_HOUR] % 12);
  }
  inline __device__ char* hour_24(char* ptr, int32_t const* timeparts)
  {
    return int2str(ptr, 2, timeparts[DU_HOUR]);
  }
  inline __device__ char* am_or_pm(char* ptr, int32_t const* timeparts)
  {
    *ptr++ = (timeparts[DU_HOUR] / 12 == 0 ? 'A' : 'P');
    *ptr++ = 'M';
    return ptr;
  }
  inline __device__ char* minute(char* ptr, int32_t const* timeparts)
  {
    return int2str(ptr, 2, timeparts[DU_MINUTE]);
  }
  inline __device__ char* second(char* ptr, int32_t const* timeparts)
  {
    return int2str(ptr, 2, timeparts[DU_SECOND]);
  }

  inline __device__ char* subsecond(char* ptr, int32_t const* timeparts)
  {
    if (timeparts[DU_SUBSECOND] == 0) return ptr;
    char subsecond_digits[] = ".000000000";  // 9 max digits
    const int digits        = duration_to_string_size_fn<T>::format_length('S', timeparts) - 3;
    int2str(subsecond_digits + 1, digits, timeparts[DU_SUBSECOND]);  // +1 is for dot
    ptr = copy_and_increment(ptr, subsecond_digits, digits + 1);
    return ptr;
  }

  // TODO argument order change
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
        case 'D':  // days
          ptr = day(ptr, timeparts);
          break;
        case '-':  // - if value is negative
          if (timeparts[DU_NEGATIVE]) *ptr++ = '-';
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
    auto duration                   = d_durations.template element<T>(idx);
    int32_t timeparts[DU_ARRAYSIZE] = {0};  // days, hours, minutes, seconds, subseconds(9)
    dissect_duration(duration.count(), timeparts, type);
    // convert to characters
    format_from_parts(timeparts, d_chars + d_offsets[idx]);
  }
};

/**
 * @brief This dispatch method is for converting integers into strings.
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
      duration_to_string_size_fn<T>{
        d_column, d_format_items, compiler.items_count(), durations.type().id()});
    auto offsets_column = detail::make_offsets_child_column(
      offsets_transformer_itr, offsets_transformer_itr + strings_count, mr, stream);
    auto offsets_view  = offsets_column->view();
    auto d_new_offsets = offsets_view.template data<int32_t>();
    rmm::device_vector<int32_t> ofst(offsets_transformer_itr,
                                     offsets_transformer_itr + strings_count);
    thrust::copy(ofst.begin(), ofst.end(), std::ostream_iterator<int32_t>(std::cout, " "));
    std::cout << "\n";

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
                       duration_to_string_fn<T>{d_column,
                                                d_format_items,
                                                compiler.items_count(),
                                                durations.type().id(),
                                                d_new_offsets,
                                                d_chars});

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

// this parses duration characters into a duration integer
template <typename T>  // duration type
struct parse_duration {
  column_device_view const d_strings;
  format_item const* d_format_items;
  size_type items_count;
  type_id type;

  //
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
    // add end of string condition.
    for (int8_t idx = 0; idx < fixed_width && idx < string_length; ++idx) {
      char chr = *ptr++;
      if (chr < '0' || chr > '9') {
        ptr--;  // roll back
        break;
      }
      value = (value * 10) + static_cast<int32_t>(chr - '0');
    }
    auto parsed_length = ptr - str;
    // trailing zeros
    constexpr int64_t powers_of_ten[] = {
      1L, 10L, 100L, 1000L, 10000L, 100000L, 1000000L, 10000000L, 100000000L, 1000000000L};
    if (parsed_length < fixed_width) value *= powers_of_ten[fixed_width - parsed_length];
    actual_length += parsed_length;
    return value;
  }

  // Walk the format_items to read the datetime string.
  // Returns 0 if all ok.
  __device__ int parse_into_parts(string_view const& d_string, int32_t* timeparts)
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
      timeparts[DU_NEGATIVE] |= (*ptr == '-');

      // special logic for each specifier
      // TODO parse_day, parse_hour, parse_minute, parse_second, parse_am_or_pm
      int8_t item_length{0};
      switch (item.value) {
        case 'D': timeparts[DU_DAY] = str2int(ptr, 11, item_length); break;
        case '-': break;  // skip
        case 'H': timeparts[DU_HOUR] = str2int(ptr, 2, item_length); hour_shift=0; break;
        case 'I': timeparts[DU_MINUTE] = str2int(ptr, 2, item_length); hour_shift=12; break;
        case 'M': timeparts[DU_MINUTE] = str2int(ptr, 2, item_length); break;
        case 'S':
          timeparts[DU_SECOND] = str2int(ptr, 2, item_length);
          if (*(ptr+item_length) == '.') {
            item_length++;
            int64_t nanoseconds      = str2int_fixed(
              ptr + item_length, 9, length - item_length, item_length);  // normalize to nanoseconds
            timeparts[DU_SUBSECOND] = nanoseconds;
          }
          break;
        case 'p':
          if (*ptr == 'P' && *(ptr + 1) == 'M') hour_shift = 12;
          item_length = 2;
          break;
        case 'R':
          timeparts[DU_HOUR] = str2int(ptr, 2, item_length); hour_shift=0;
          item_length++;
          timeparts[DU_MINUTE] = str2int(ptr+item_length, 2, item_length);
          break;
        case 'T':
          timeparts[DU_HOUR] = str2int(ptr, 2, item_length); hour_shift=0;
          item_length++;
          timeparts[DU_MINUTE] = str2int(ptr+item_length, 2, item_length);
          item_length++;
          timeparts[DU_SECOND] = str2int(ptr, 2, item_length);
          break;
        default: return 3;
      }
      ptr += item_length;
      length -= item_length;
    }
    timeparts[DU_NEGATIVE] =
      (timeparts[DU_NEGATIVE] || timeparts[DU_DAY] < 0 || timeparts[DU_HOUR] < 0 ||
       timeparts[DU_MINUTE] < 0 || timeparts[DU_SECOND] < 0 || timeparts[DU_SUBSECOND] < 0);
    auto negate = [](auto i, bool b) { return (i < 0 ? i : (b ? -i : i)); };
    if (timeparts[DU_NEGATIVE]) {
      timeparts[DU_DAY]       = negate(timeparts[DU_DAY], timeparts[DU_NEGATIVE]);
      timeparts[DU_HOUR]      = negate(timeparts[DU_HOUR], timeparts[DU_NEGATIVE]);
      timeparts[DU_MINUTE]    = negate(timeparts[DU_MINUTE], timeparts[DU_NEGATIVE]);
      timeparts[DU_SECOND]    = negate(timeparts[DU_SECOND], timeparts[DU_NEGATIVE]);
      timeparts[DU_SUBSECOND] = negate(timeparts[DU_SUBSECOND], timeparts[DU_NEGATIVE]);
      hour_shift = -hour_shift;
    }
    timeparts[DU_HOUR] += hour_shift;
    return 0;
  }

  __device__ int64_t duration_from_parts(int32_t const* timeparts, type_id units)
  {
    int32_t days = timeparts[DU_DAY];
    if (units == type_id::DURATION_DAYS) return days;

    auto hour     = timeparts[DU_HOUR];
    auto minute   = timeparts[DU_MINUTE];
    auto second   = timeparts[DU_SECOND];
    auto duration = duration_D(days) + simt::std::chrono::hours(hour) +
                    simt::std::chrono::minutes(minute) + duration_s(second);
    if (units == type_id::DURATION_SECONDS)
      return simt::std::chrono::duration_cast<duration_s>(duration).count();

    duration_ns subsecond(timeparts[DU_SUBSECOND]);  // ns
    if (units == type_id::DURATION_MILLISECONDS) {
      return simt::std::chrono::duration_cast<duration_ms>(duration + subsecond).count();
    } else if (units == type_id::DURATION_MICROSECONDS) {
      return simt::std::chrono::duration_cast<duration_us>(duration + subsecond).count();
    } else if (units == type_id::DURATION_NANOSECONDS)
      return simt::std::chrono::duration_cast<duration_ns>(duration + subsecond).count();
    return simt::std::chrono::duration_cast<duration_ns>(duration + subsecond).count();
  }

  __device__ T operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) return T{0};
    string_view d_str = d_strings.element<string_view>(idx);
    if (d_str.empty()) return T{0};
    //
    int32_t timeparts[DU_ARRAYSIZE] = {0};
    if (parse_into_parts(d_str, timeparts)) return T{0};  // unexpected parse case
    //
    return static_cast<T>(duration_from_parts(timeparts, type));
  }
};

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
    parse_duration<T> pfn{d_strings, d_items, compiler.items_count(), results_view.type().id()};
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
    CUDF_FAIL("Only durations type are expected");
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
