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
#include <cudf/types.hpp>
#include <strings/utilities.cuh>

#include <map>
#include <vector>
#include "cudf/utilities/traits.hpp"

namespace cudf {
namespace strings {
namespace detail {

// algo
// split internal rep to D,H,M,S,ns
// ask struct {int64_t, int8_t, int8_t, int8_t, int64_t}, then type specific function to convert.
// reuse conver_integers.cu code for size, string_conversion for days.
// move functions int->str, str->int to .cuh file, & reuse
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
                                              {'+', -1}, //only for negative days
                                              {'H', 2},
                                              {'M', 2},
                                              {'S', 2},
                                              {'u', -1}, //0 or 6.
                                              {'f', -1}};  //0 or <=9 without trialing zeros

  format_compiler(const char* format, type_id units) : format(format), units(units) {}

  format_item const* compile_to_device()
  {
    std::vector<format_item> items;
    const char* str = format.c_str();
    auto length     = format.length();
    const bool is_isoformat = str[0]=='P';
    if(is_isoformat) {
      specifier_lengths['H']=-1;
      specifier_lengths['M']=-1;
      specifier_lengths['S']=-1;
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
        specifier_lengths[*str] = static_cast<int8_t>(ch - '0');
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
  //int8_t subsecond_precision() const { return specifier_lengths.at('f'); }
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
    timeparts[DU_SUBSECOND] = modulo_time(duration, 1000);  // * 1000 * 1000;
  } else if (units == type_id::DURATION_MICROSECONDS) {
    seconds = simt::std::chrono::floor<cudf::duration_s>(cudf::duration_us(duration)).count();
    timeparts[DU_SUBSECOND] = modulo_time(duration, 1000 * 1000);  // * 1000;
  } else if (units == type_id::DURATION_NANOSECONDS) {
    seconds = simt::std::chrono::floor<cudf::duration_s>(cudf::duration_ns(duration)).count();
    timeparts[DU_SUBSECOND] = modulo_time(duration, 1000 * 1000 * 1000);
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
  // TODO: add boolean for isoformat

  __device__ int8_t countTrailingZeros(int n) const
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

  __device__ int8_t format_length(char format_char, int32_t const * const timeparts) const
  {
    switch (format_char) {
      case 'd': return count_digits(timeparts[DU_DAY]); break;
      case '+': return timeparts[DU_DAY] < 0 ? 1 : 0; break;
      case 'H': return count_digits(timeparts[DU_HOUR]); break;
      case 'M': return count_digits(timeparts[DU_MINUTE]); break;
      case 'S': return count_digits(timeparts[DU_SECOND]); break;
      // TODO: include dot only if non-zero.
      //0 or 6 digits for pandas.
      case 'u': return (timeparts[DU_SUBSECOND]==0) ? 0 : 6; break;
      //0 or ns without trailing zeros
       //TODO count digits without trailing zeros!
      case 'f':
        return (timeparts[DU_SUBSECOND] == 0) ? 0 :
        [units = type] {
          if (units == type_id::DURATION_MILLISECONDS) return 3;
          if (units == type_id::DURATION_MICROSECONDS) return 6;
          if (units == type_id::DURATION_NANOSECONDS) return 9;
          return 0;
        }() - countTrailingZeros(timeparts[DU_SUBSECOND]);//3/6/9-trailing_zeros.
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

    char digits[MAX_DIGITS]  = {'0', '0', '0', '0', '0', '0', '0', '0', '0', '0'};
    int digits_idx           = 0;
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
    digits_idx = std::max(digits_idx, min_digits); //TODO FIXME fixed_digits size, not min_digits
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
          if(timeparts[DU_DAY]<0) *ptr++='+';
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
          char subsecond_digits[] = "000000000";  // 9 max digits
          const int digits        = [units = type] {
            if (units == type_id::DURATION_MILLISECONDS) return 3;
            if (units == type_id::DURATION_MICROSECONDS) return 6;
            if (units == type_id::DURATION_NANOSECONDS) return 9;
            return 0;
          }();
          int2str(subsecond_digits, digits, timeparts[DU_SUBSECOND]);
          ptr = copy_and_increment(
            ptr,
            subsecond_digits,
            item.length > 0 ? item.length : duration_to_string_size_fn<T>::format_length(item.value, timeparts));
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
    auto column   = column_device_view::create(durations, stream);
    auto d_column = *column;

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

}  // namespace

std::unique_ptr<column> from_durations(
  column_view const& durations,
  std::string const& format,
  // "%d days %H:%M:%S.%6f",
  //"P%YY%MM%DDT%HH%MM%SS" is_iso_format() for no padding zeros for HMS and no trailing zeros for subseconds.
  // TODO
  // common for all: check if non-zero days,
  // per item: non-zero subseconds present. 1(.)+(width 3/6/9 based on non-zero ms/us/ns)
  // P%YY%MM%DDT%HH%MM%S%fS
  cudaStream_t stream,
  rmm::mr::device_memory_resource* mr)
{
  size_type strings_count = durations.size();
  if (strings_count == 0) return make_empty_strings_column(mr, stream);

  return type_dispatcher(durations.type(), dispatch_from_durations_fn{}, durations, format, mr, stream);
}

}  // namespace detail

std::unique_ptr<column> from_durations(column_view const& durations,
                                       std::string const& format,
                                       rmm::mr::device_memory_resource* mr)
{
  return detail::from_durations(durations, format, cudaStream_t{}, mr);
}
}  // namespace strings
}  // namespace cudf
