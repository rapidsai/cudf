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
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/types.hpp>
#include <strings/utilities.cuh>

#include <map>
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
  DU_ARRAYSIZE = 5
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
  type_id units;  // is this required?
  rmm::device_vector<format_item> d_items;

  std::map<char, int8_t> specifier_lengths = {{'d', -1},
                                              {'+', -1},  // only for negative days
                                              {'H', 2},
                                              {'M', 2},
                                              {'S', 2},
                                              {'u', -1},   // 0 or 6+1(dot)
                                              {'f', -1}};  // 0 or <=9+1(dot) without trialing zeros

  format_compiler(const char* format, type_id units) : format(format), units(units) {}

  format_item const* compile_to_device()
  {
    std::vector<format_item> items;
    const char* str         = format.c_str();
    auto length             = format.length();
    const bool is_isoformat = str[0] == 'P';
    if (is_isoformat) {
      specifier_lengths['H'] = -1;
      specifier_lengths['M'] = -1;
      specifier_lengths['S'] = -1;
    }
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
      if (ch >= '0' && ch <= '9') {
        CUDF_EXPECTS(*str == 'f', "precision not supported for specifier: " + std::string(1, *str));
        specifier_lengths[*str] = static_cast<int8_t>(ch - '0') + 1;
        ch                      = *str++;
        length--;
      }
      CUDF_EXPECTS(specifier_lengths.find(ch) != specifier_lengths.end(),
                   "invalid format specifier: " + std::string(1, ch));

      int8_t spec_length = specifier_lengths[ch];
      items.push_back(format_item::new_specifier(ch, spec_length));
    }
    // create program in device memory
    d_items.resize(items.size());
    CUDA_TRY(cudaMemcpyAsync(
      d_items.data().get(), items.data(), items.size() * sizeof(items[0]), cudaMemcpyHostToDevice));
    return d_items.data().get();
  }

  // these calls are only valid after compile_to_device is called
  size_type items_count() const { return static_cast<size_type>(d_items.size()); }
};

__device__ void dissect_duration(int64_t duration, int32_t* timeparts, type_id units)
{
  if (units == type_id::DURATION_DAYS) {
    timeparts[DU_DAY] = static_cast<int32_t>(duration);
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

  int64_t seconds{0};
  if (units == type_id::DURATION_SECONDS) {
    seconds = duration;
  } else if (units == type_id::DURATION_MILLISECONDS) {
    seconds = simt::std::chrono::floor<cudf::duration_s>(cudf::duration_ms(duration)).count();
    timeparts[DU_SUBSECOND] = modulo_time(duration, 1000L);
  } else if (units == type_id::DURATION_MICROSECONDS) {
    seconds = simt::std::chrono::floor<cudf::duration_s>(cudf::duration_us(duration)).count();
    timeparts[DU_SUBSECOND] = modulo_time(duration, 1000L * 1000);
  } else if (units == type_id::DURATION_NANOSECONDS) {
    seconds = simt::std::chrono::floor<cudf::duration_s>(cudf::duration_ns(duration)).count();
    timeparts[DU_SUBSECOND] = modulo_time(duration, 1000L * 1000 * 1000);
  }
  timeparts[DU_DAY]    = scale_time(seconds, 24 * 60 * 60);
  timeparts[DU_HOUR]   = modulo_time(scale_time(seconds, 60 * 60), 24);
  timeparts[DU_MINUTE] = modulo_time(scale_time(seconds, 60), 60);
  timeparts[DU_SECOND] = modulo_time(seconds, 60);
}

template <typename T>
struct duration_to_string_size_fn {
  const column_device_view d_durations;
  const format_item* d_format_items;
  size_type items_count;
  type_id type;

  __device__ int8_t count_trailing_zeros(int n) const
  {
    int8_t zeros = 0;
    if ((n % 100000000) == 0) {
      zeros += 8;
      n /= 100000000;
    }
    if ((n % 10000) == 0) {
      zeros += 4;
      n /= 10000;
    }
    if ((n % 100) == 0) {
      zeros += 2;
      n /= 100;
    }
    if ((n % 10) == 0) { zeros++; }
    return zeros;
  }

  __device__ int8_t format_length(char format_char, int32_t const* const timeparts) const
  {
    switch (format_char) {
      case 'd': return count_digits(timeparts[DU_DAY]); break;
      case '+': return timeparts[DU_DAY] < 0 ? 1 : 0; break;
      case 'H': return count_digits(timeparts[DU_HOUR]); break;
      case 'M': return count_digits(timeparts[DU_MINUTE]); break;
      case 'S': return count_digits(timeparts[DU_SECOND]); break;
      // 0 or 6 digits for pandas, include dot only if non-zero.
      case 'u': return (timeparts[DU_SUBSECOND] == 0) ? 0 : 6 + 1; break;
      // 0 or ns without trailing zeros
      case 'f':
        return (timeparts[DU_SUBSECOND] == 0) ? 0 : [units = type] {
          if (units == type_id::DURATION_MILLISECONDS) return 3 + 1;
          if (units == type_id::DURATION_MICROSECONDS) return 6 + 1;
          if (units == type_id::DURATION_NANOSECONDS) return 9 + 1;
          return 0;
        }() - count_trailing_zeros(timeparts[DU_SUBSECOND]);  // 3/6/9-trailing_zeros.
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
    size_type string_length{0};
    for (auto i = 0; i < items_count; i++) {
      format_item item = d_format_items[i];
      if (item.item_type == format_char_type::literal)
        string_length++;
      else if (item.length != -1)
        string_length += item.length;
      else
        string_length += format_length(item.value, timeparts);
    }
    return string_length;
    // convert to characters
    // format_from_parts(timeparts, d_chars + d_offsets[idx]);
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

  // utility to create (optionally) 0-padded integers (up to 10 chars) with negative sign.
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
    bool is_negative = (value < 0);

    char digits[MAX_DIGITS] = {'0', '0', '0', '0', '0', '0', '0', '0', '0', '0'};
    int digits_idx          = 0;
    while (value != 0) {
      assert(digits_idx < MAX_DIGITS);
      digits[digits_idx++] = '0' + std::abs(value % 10);
      // next digit
      value = value / 10;
    }
    if (is_negative) {
      *str++ = '-';
      min_digits--;
    }
    digits_idx = std::max(digits_idx, min_digits);
    // digits are backwards, reverse the string into the output
    while (digits_idx-- > 0) *str++ = digits[digits_idx];
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
        case 'd':  // days
          ptr = int2str(ptr, item.length, timeparts[DU_DAY]);
          break;
        case '+':  // + if day is negative
          if (timeparts[DU_DAY] < 0) *ptr++ = '+';
          break;
        case 'H':  // 24-hour
          ptr = int2str(ptr, item.length, timeparts[DU_HOUR]);
          break;
        case 'M':  // minute
          ptr = int2str(ptr, item.length, timeparts[DU_MINUTE]);
          break;
        case 'S':  // second
          ptr = int2str(ptr, item.length, timeparts[DU_SECOND]);
          break;
        case 'u':
        case 'f':  // sub-second
        {
          char subsecond_digits[] = ".000000000";  // 9 max digits
          const int digits        = [units = type] {
            if (units == type_id::DURATION_MILLISECONDS) return 3;
            if (units == type_id::DURATION_MICROSECONDS) return 6;
            if (units == type_id::DURATION_NANOSECONDS) return 9;
            return 0;
          }();
          int2str(subsecond_digits + 1, digits, timeparts[DU_SUBSECOND]);
          ptr = copy_and_increment(
            ptr,
            subsecond_digits,
            item.length > 0 ? item.length
                            : duration_to_string_size_fn<T>::format_length(item.value, timeparts));
          break;
        }
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
 * The template function declaration ensures only integer types are used.
 */
struct dispatch_from_durations_fn {
  template <typename T, std::enable_if_t<cudf::is_duration<T>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& durations,
                                     std::string const& format,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream) const
  {
    CUDF_EXPECTS(!format.empty(), "Format parameter must not be empty.");

    format_compiler compiler(format.c_str(), durations.type().id());
    auto d_format_items = compiler.compile_to_device();

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

    // build chars column
    size_type bytes = thrust::device_pointer_cast(d_new_offsets)[strings_count];
    auto chars_column =
      detail::create_chars_child_column(strings_count, durations.null_count(), bytes, mr, stream);
    auto chars_view = chars_column->mutable_view();
    auto d_chars    = chars_view.template data<char>();
    thrust::fill_n(rmm::exec_policy(stream)->on(stream), d_chars, bytes, '0');

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
    CUDF_FAIL("Values for from_duration function must be a duration type.");
  }
};


// this parses duration characters into a duration integer
template <typename T>  // duration type
struct parse_duration {
  column_device_view const d_strings;
  format_item const* d_format_items;
  size_type items_count;
  type_id units;
  //int8_t subsecond_precision;

  //
  __device__ int32_t str2int(const char* str, int8_t max_bytes, int8_t& actual_length)
  {
    const char* ptr = (*str=='-' || *str=='+')? str+1 : str;
    int32_t value   = 0;
    for (int8_t idx = 0; idx < max_bytes; ++idx) {
      char chr = *ptr++;
      if (chr < '0' || chr > '9') {
        ptr--; //roll back
        break;
      }
      value = (value * 10) + static_cast<int32_t>(chr - '0');
    }
    actual_length = ptr-str;
    return (*str=='-')? -value : value;
  }

 // function to parse fraction of decimal value with trailing zeros removed.
  __device__ int32_t str2int_fixed(const char* str, int8_t fixed_width, size_type string_length, int8_t& actual_length)
  {
    const char* ptr = (*str=='.')? str+1 : str;
    int32_t value   = 0;
    // add end of string condition.
    for (int8_t idx = 0; idx < fixed_width && idx<string_length; ++idx) {
      char chr = *ptr++;
      if (chr < '0' || chr > '9') {
        ptr--; //roll back
        break;
      }
      value = (value * 10) + static_cast<int32_t>(chr - '0');
    }
    actual_length = ptr-str;
    // trailing zeros
    constexpr int64_t powers_of_ten[] = {
      1L, 10L, 100L, 1000L, 10000L, 100000L, 1000000L, 10000000L, 100000000L, 1000000000L};
    if(actual_length<fixed_width)
      value *= powers_of_ten[fixed_width-actual_length];
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
      int8_t item_length{0};
      switch (item.value) {
        case 'd': timeparts[DU_DAY] = str2int(ptr, 11, item_length); break;
        case '+': if(*ptr == '+') item_length=1; break;  // skip
        case 'H': timeparts[DU_HOUR] = str2int(ptr, 2, item_length); break;
        case 'M': timeparts[DU_MINUTE] = str2int(ptr, 2, item_length); break;
        case 'S': timeparts[DU_SECOND] = str2int(ptr, 2, item_length); break;
        case 'u':
        case 'f':
            if(*ptr == '.') {
                auto subsecond_precision = (item.length==-1)? 9: item.length-1;
                auto subsecond = str2int_fixed(ptr+1, subsecond_precision, length-1, item_length);
                constexpr int64_t powers_of_ten[] = {
                  1L, 10L, 100L, 1000L, 10000L, 100000L, 1000000L, 10000000L, 100000000L, 1000000000L};
                int64_t nanoseconds =  subsecond * powers_of_ten[9 - subsecond_precision];  // normalize to nanoseconds
                timeparts[DU_SUBSECOND] = nanoseconds;
                item_length++;
            }
        break;
        default: return 3;
      }
      ptr += item_length;
      length -= item_length;
    }
    return 0;
  }

  __device__ int64_t timestamp_from_parts(int32_t const* timeparts)
  {
    int32_t days = timeparts[DU_DAY];
    if (units == type_id::DURATION_DAYS) return days;

    auto hour   = timeparts[DU_HOUR];
    auto minute = timeparts[DU_MINUTE];
    auto second = timeparts[DU_SECOND];
    int64_t timestamp = (days * 24L * 3600L) + (hour * 3600L) + (minute * 60L) + second;
    if (units == type_id::DURATION_SECONDS) return timestamp;

    auto subsecond = timeparts[DU_SUBSECOND];
    if (units == type_id::DURATION_MILLISECONDS) {
      timestamp *= 1000L;
      subsecond = subsecond / 1000000L;
    } else if (units == type_id::DURATION_MICROSECONDS) {
      timestamp *= 1000000L;
      subsecond = subsecond / 1000L;
    } else if (units == type_id::DURATION_NANOSECONDS)
      timestamp *= 1000000000L;
    timestamp += subsecond;
    return timestamp;
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
    return static_cast<T>(timestamp_from_parts(timeparts));
  }
};


// dispatch operator to map timestamp to native fixed-width-type
struct dispatch_to_durations_fn {
  template <typename T, std::enable_if_t<cudf::is_duration<T>()>* = nullptr>
  void operator()(column_device_view const& d_strings,
                  std::string const& format,
                  type_id units,
                  mutable_column_view& results_view,
                  cudaStream_t stream) const
  {
    format_compiler compiler(format.c_str(), d_strings.type().id());
    auto d_items   = compiler.compile_to_device();
    auto d_results = results_view.data<T>();
    parse_duration<T> pfn{
      d_strings, d_items, compiler.items_count(), units};
    thrust::transform(rmm::exec_policy(stream)->on(stream),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(results_view.size()),
                      d_results,
                      pfn);
  }
  template <typename T, std::enable_if_t<not cudf::is_duration<T>()>* = nullptr>
  void operator()(column_device_view const&,
                  std::string const&,
                  type_id,
                  mutable_column_view&,
                  cudaStream_t) const
  {
    CUDF_FAIL("Only durations type are expected");
  }
};

}  // namespace

// TODO skip days if non-zero days for non-ISO format.
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
    duration_type, dispatch_to_durations_fn(), d_column, format, duration_type.id(), results_view, stream);
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
