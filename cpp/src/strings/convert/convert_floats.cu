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
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/convert/convert_floats.hpp>
#include <cudf/strings/detail/converters.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <strings/utilities.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <cmath>
#include <limits>

namespace cudf {
namespace strings {
namespace detail {
namespace {
/**
 * @brief This function converts the given string into a
 * floating point double value.
 *
 * This will also map strings containing "NaN", "Inf" and "-Inf"
 * to the appropriate float values.
 *
 * This function will also handle scientific notation format.
 */
__device__ inline double stod(string_view const& d_str)
{
  const char* in_ptr = d_str.data();
  const char* end    = in_ptr + d_str.size_bytes();
  if (end == in_ptr) return 0.0;
  // special strings
  if (d_str.compare("NaN", 3) == 0) return std::numeric_limits<double>::quiet_NaN();
  if (d_str.compare("Inf", 3) == 0) return std::numeric_limits<double>::infinity();
  if (d_str.compare("-Inf", 4) == 0) return -std::numeric_limits<double>::infinity();
  double sign = 1.0;
  if (*in_ptr == '-' || *in_ptr == '+') {
    sign = (*in_ptr == '-' ? -1 : 1);
    ++in_ptr;
  }
  unsigned long max_mantissa = 0x0FFFFFFFFFFFFF;
  unsigned long digits       = 0;
  int exp_off                = 0;
  bool decimal               = false;
  while (in_ptr < end) {
    char ch = *in_ptr;
    if (ch == '.') {
      decimal = true;
      ++in_ptr;
      continue;
    }
    if (ch < '0' || ch > '9') break;
    if (digits > max_mantissa)
      exp_off += (int)!decimal;
    else {
      digits = (digits * 10L) + (unsigned long)(ch - '0');
      if (digits > max_mantissa) {
        digits = digits / 10L;
        exp_off += (int)!decimal;
      } else
        exp_off -= (int)decimal;
    }
    ++in_ptr;
  }
  // check for exponent char
  int exp_ten  = 0;
  int exp_sign = 1;
  if (in_ptr < end) {
    char ch = *in_ptr++;
    if (ch == 'e' || ch == 'E') {
      if (in_ptr < end) {
        ch = *in_ptr;
        if (ch == '-' || ch == '+') {
          exp_sign = (ch == '-' ? -1 : 1);
          ++in_ptr;
        }
        while (in_ptr < end) {
          ch = *in_ptr++;
          if (ch < '0' || ch > '9') break;
          exp_ten = (exp_ten * 10) + (int)(ch - '0');
        }
      }
    }
  }
  exp_ten *= exp_sign;
  exp_ten += exp_off;
  if (exp_ten > 308)
    return sign > 0 ? std::numeric_limits<double>::infinity()
                    : -std::numeric_limits<double>::infinity();
  else if (exp_ten < -308)
    return 0.0;
  // using exp10() since the pow(10.0,exp_ten) function is
  // very inaccurate in 10.2: http://nvbugs/2971187
  double value = static_cast<double>(digits) * exp10(static_cast<double>(exp_ten));
  return (value * sign);
}

/**
 * @brief Converts strings column entries into floats.
 *
 * Used by the dispatch method to convert to different float types.
 */
template <typename FloatType>
struct string_to_float_fn {
  const column_device_view strings_column;  // strings to convert

  __device__ FloatType operator()(size_type idx)
  {
    if (strings_column.is_null(idx)) return static_cast<FloatType>(0);
    // the cast to FloatType will create predictable results
    // for floats that are larger than the FloatType can hold
    return static_cast<FloatType>(stod(strings_column.element<string_view>(idx)));
  }
};

/**
 * @brief The dispatch functions for converting strings to floats.
 *
 * The output_column is expected to be one of the float types only.
 */
struct dispatch_to_floats_fn {
  template <typename FloatType,
            std::enable_if_t<std::is_floating_point<FloatType>::value>* = nullptr>
  void operator()(column_device_view const& strings_column,
                  mutable_column_view& output_column,
                  rmm::cuda_stream_view stream) const
  {
    auto d_results = output_column.data<FloatType>();
    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(strings_column.size()),
                      d_results,
                      string_to_float_fn<FloatType>{strings_column});
  }
  // non-integral types throw an exception
  template <typename T, std::enable_if_t<not std::is_floating_point<T>::value>* = nullptr>
  void operator()(column_device_view const&, mutable_column_view&, rmm::cuda_stream_view) const
  {
    CUDF_FAIL("Output for to_floats must be a float type.");
  }
};

}  // namespace

// This will convert a strings column into any float column type.
std::unique_ptr<column> to_floats(strings_column_view const& strings,
                                  data_type output_type,
                                  rmm::cuda_stream_view stream,
                                  rmm::mr::device_memory_resource* mr)
{
  size_type strings_count = strings.size();
  if (strings_count == 0) return make_numeric_column(output_type, 0);
  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;
  // create float output column copying the strings null-mask
  auto results      = make_numeric_column(output_type,
                                     strings_count,
                                     cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                                     strings.null_count(),
                                     stream,
                                     mr);
  auto results_view = results->mutable_view();
  // fill output column with floats
  type_dispatcher(output_type, dispatch_to_floats_fn{}, d_strings, results_view, stream);
  results->set_null_count(strings.null_count());
  return results;
}

}  // namespace detail

// external API

std::unique_ptr<column> to_floats(strings_column_view const& strings,
                                  data_type output_type,
                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::to_floats(strings, output_type, rmm::cuda_stream_default, mr);
}

namespace detail {
namespace {
/**
 * @brief Code logic for converting float value into a string.
 *
 * The floating point components are dissected and used to fill an
 * existing output char array.
 */
struct ftos_converter {
  // significant digits is independent of scientific notation range
  // digits more than this may require using long values instead of ints
  static constexpr unsigned int significant_digits = 10;
  // maximum power-of-10 that will fit in 32-bits
  static constexpr unsigned int nine_digits = 1000000000;  // 1x10^9
  // Range of numbers here is for normalizing the value.
  // If the value is above or below the following limits, the output is converted to
  // scientific notation in order to show (at most) the number of significant digits.
  static constexpr double upper_limit = 1000000000;  // max is 1x10^9
  static constexpr double lower_limit = 0.0001;      // printf uses scientific notation below this
  // Tables for doing normalization: converting to exponent form
  // IEEE double float has maximum exponent of 305 so these should cover everthing
  const double upper10[9]  = {10, 100, 10000, 1e8, 1e16, 1e32, 1e64, 1e128, 1e256};
  const double lower10[9]  = {.1, .01, .0001, 1e-8, 1e-16, 1e-32, 1e-64, 1e-128, 1e-256};
  const double blower10[9] = {1.0, .1, .001, 1e-7, 1e-15, 1e-31, 1e-63, 1e-127, 1e-255};

  // utility for quickly converting known integer range to character array
  __device__ char* int2str(int value, char* output)
  {
    if (value == 0) {
      *output++ = '0';
      return output;
    }
    char buffer[significant_digits];  // should be big-enough for significant digits
    char* ptr = buffer;
    while (value > 0) {
      *ptr++ = (char)('0' + (value % 10));
      value /= 10;
    }
    while (ptr != buffer) *output++ = *--ptr;  // 54321 -> 12345
    return output;
  }

  /**
   * @brief Dissect a float value into integer, decimal, and exponent components.
   *
   * @return The number of decimal places.
   */
  __device__ int dissect_value(double value,
                               unsigned int& integer,
                               unsigned int& decimal,
                               int& exp10)
  {
    int decimal_places = significant_digits - 1;
    // normalize step puts value between lower-limit and upper-limit
    // by adjusting the exponent up or down
    exp10 = 0;
    if (value > upper_limit) {
      int fx = 256;
      for (int idx = 8; idx >= 0; --idx) {
        if (value >= upper10[idx]) {
          value *= lower10[idx];
          exp10 += fx;
        }
        fx = fx >> 1;
      }
    } else if ((value > 0.0) && (value < lower_limit)) {
      int fx = 256;
      for (int idx = 8; idx >= 0; --idx) {
        if (value < blower10[idx]) {
          value *= upper10[idx];
          exp10 -= fx;
        }
        fx = fx >> 1;
      }
    }
    //
    unsigned int max_digits = nine_digits;
    integer                 = (unsigned int)value;
    for (unsigned int i = integer; i >= 10; i /= 10) {
      --decimal_places;
      max_digits /= 10;
    }
    double remainder = (value - (double)integer) * (double)max_digits;
    decimal          = (unsigned int)remainder;
    remainder -= (double)decimal;
    decimal += (unsigned int)(2.0 * remainder);
    if (decimal >= max_digits) {
      decimal = 0;
      ++integer;
      if (exp10 && (integer >= 10)) {
        ++exp10;
        integer = 1;
      }
    }
    //
    while ((decimal % 10) == 0 && (decimal_places > 0)) {
      decimal /= 10;
      --decimal_places;
    }
    return decimal_places;
  }

  /**
   * @brief Main kernel method for converting float value to char output array.
   *
   * Output need not be more than (significant_digits + 7) bytes:
   * 7 = 1 sign, 1 decimal point, 1 exponent ('e'), 1 exponent-sign, 3 digits for exponent
   *
   * @param value Float value to convert.
   * @param output Memory to write output characters.
   * @return Number of bytes written.
   */
  __device__ int float_to_string(double value, char* output)
  {
    // check for valid value
    if (std::isnan(value)) {
      memcpy(output, "NaN", 3);
      return 3;
    }
    bool bneg = false;
    if (signbit(value)) {  // handles -0.0 too
      value = -value;
      bneg  = true;
    }
    if (std::isinf(value)) {
      if (bneg)
        memcpy(output, "-Inf", 4);
      else
        memcpy(output, "Inf", 3);
      return bneg ? 4 : 3;
    }

    // dissect value into components
    unsigned int integer = 0, decimal = 0;
    int exp10          = 0;
    int decimal_places = dissect_value(value, integer, decimal, exp10);
    //
    // now build the string from the
    // components: sign, integer, decimal, exp10, decimal_places
    //
    // sign
    char* ptr = output;
    if (bneg) *ptr++ = '-';
    // integer
    ptr = int2str(integer, ptr);
    // decimal
    *ptr++ = '.';
    if (decimal_places) {
      char buffer[10];
      char* pb = buffer;
      while (decimal_places--) {
        *pb++ = (char)('0' + (decimal % 10));
        decimal /= 10;
      }
      while (pb != buffer)  // reverses the digits
        *ptr++ = *--pb;     // e.g. 54321 -> 12345
    } else
      *ptr++ = '0';  // always include at least .0
    // exponent
    if (exp10) {
      *ptr++ = 'e';
      if (exp10 < 0) {
        *ptr++ = '-';
        exp10  = -exp10;
      } else
        *ptr++ = '+';
      if (exp10 < 10) *ptr++ = '0';  // extra zero-pad
      ptr = int2str(exp10, ptr);
    }
    // done
    return (int)(ptr - output);  // number of bytes written
  }

  /**
   * @brief Compute how man bytes are needed to hold the output string.
   *
   * @param value Float value to convert.
   * @return Number of bytes required.
   */
  __device__ int compute_ftos_size(double value)
  {
    if (std::isnan(value)) return 3;  // NaN
    bool bneg = false;
    if (signbit(value)) {  // handles -0.0 too
      value = -value;
      bneg  = true;
    }
    if (std::isinf(value)) return 3 + (int)bneg;  // Inf

    // dissect float into parts
    unsigned int integer = 0, decimal = 0;
    int exp10          = 0;
    int decimal_places = dissect_value(value, integer, decimal, exp10);
    // now count up the components
    // sign
    int count = (int)bneg;
    // integer
    count += (int)(integer == 0);
    while (integer > 0) {
      integer /= 10;
      ++count;
    }  // log10(integer)
    // decimal
    ++count;  // decimal point
    if (decimal_places)
      count += decimal_places;
    else
      ++count;  // always include .0
    // exponent
    if (exp10) {
      count += 2;  // 'e±'
      if (exp10 < 0) exp10 = -exp10;
      count += (int)(exp10 < 10);  // padding
      while (exp10 > 0) {
        exp10 /= 10;
        ++count;
      }  // log10(exp10)
    }
    return count;
  }
};

template <typename FloatType>
struct float_to_string_size_fn {
  column_device_view d_column;

  __device__ size_type operator()(size_type idx)
  {
    if (d_column.is_null(idx)) return 0;
    FloatType value = d_column.element<FloatType>(idx);
    ftos_converter fts;
    return static_cast<size_type>(fts.compute_ftos_size(static_cast<double>(value)));
  }
};

template <typename FloatType>
struct float_to_string_fn {
  const column_device_view d_column;
  const int32_t* d_offsets;
  char* d_chars;

  __device__ void operator()(size_type idx)
  {
    if (d_column.is_null(idx)) return;
    FloatType value = d_column.element<FloatType>(idx);
    ftos_converter fts;
    fts.float_to_string(static_cast<double>(value), d_chars + d_offsets[idx]);
  }
};

/**
 * @brief This dispatch method is for converting floats into strings.
 *
 * The template function declaration ensures only float types are allowed.
 */
struct dispatch_from_floats_fn {
  template <typename FloatType,
            std::enable_if_t<std::is_floating_point<FloatType>::value>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& floats,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    size_type strings_count = floats.size();
    auto column             = column_device_view::create(floats, stream);
    auto d_column           = *column;

    // copy the null mask
    rmm::device_buffer null_mask = cudf::detail::copy_bitmask(floats, stream, mr);
    // build offsets column
    auto offsets_transformer_itr = thrust::make_transform_iterator(
      thrust::make_counting_iterator<int32_t>(0), float_to_string_size_fn<FloatType>{d_column});
    auto offsets_column = detail::make_offsets_child_column(
      offsets_transformer_itr, offsets_transformer_itr + strings_count, stream, mr);
    auto offsets_view = offsets_column->view();
    auto d_offsets    = offsets_view.template data<int32_t>();

    // build chars column
    size_type bytes = thrust::device_pointer_cast(d_offsets)[strings_count];
    auto chars_column =
      detail::create_chars_child_column(strings_count, floats.null_count(), bytes, stream, mr);
    auto chars_view = chars_column->mutable_view();
    auto d_chars    = chars_view.template data<char>();
    thrust::for_each_n(rmm::exec_policy(stream),
                       thrust::make_counting_iterator<size_type>(0),
                       strings_count,
                       float_to_string_fn<FloatType>{d_column, d_offsets, d_chars});
    //
    return make_strings_column(strings_count,
                               std::move(offsets_column),
                               std::move(chars_column),
                               floats.null_count(),
                               std::move(null_mask),
                               stream,
                               mr);
  }

  // non-float types throw an exception
  template <typename T, std::enable_if_t<not std::is_floating_point<T>::value>* = nullptr>
  std::unique_ptr<column> operator()(column_view const&,
                                     rmm::cuda_stream_view,
                                     rmm::mr::device_memory_resource*) const
  {
    CUDF_FAIL("Values for from_floats function must be a float type.");
  }
};

}  // namespace

// This will convert all float column types into a strings column.
std::unique_ptr<column> from_floats(column_view const& floats,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  size_type strings_count = floats.size();
  if (strings_count == 0) return detail::make_empty_strings_column(stream, mr);

  return type_dispatcher(floats.type(), dispatch_from_floats_fn{}, floats, stream, mr);
}

}  // namespace detail

// external API

std::unique_ptr<column> from_floats(column_view const& floats, rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::from_floats(floats, rmm::cuda_stream_default, mr);
}

}  // namespace strings
}  // namespace cudf
