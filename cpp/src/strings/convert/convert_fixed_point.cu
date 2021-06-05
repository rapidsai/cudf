/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/convert/convert_fixed_point.hpp>
#include <cudf/strings/detail/converters.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <strings/convert/utilities.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/optional.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {

struct string_to_decimal_base {
  /**
   * @brief Return the integer component of a decimal string.
   *
   * This is reads everything up to the exponent 'e' notation.
   * The return includes the integer digits and any exponent offset.
   *
   * @param[in,out] iter Start of characters to parse
   * @param[in] end End of characters to parse
   * @return Integer component and exponent offset.
   */
  __device__ thrust::pair<uint64_t, int32_t> parse_integer(char const*& iter,
                                                           char const* iter_end) const
  {
    // highest value where another decimal digit cannot be appended without an overflow;
    // this preserves the most digits when scaling the final result
    constexpr uint64_t decimal_max = (std::numeric_limits<uint64_t>::max() - 9L) / 10L;

    uint64_t value     = 0;  // for checking overflow
    int32_t exp_offset = 0;
    bool decimal_found = false;

    while (iter < iter_end) {
      auto const ch = *iter++;
      if (ch == '.' && !decimal_found) {
        decimal_found = true;
        continue;
      }
      if (ch < '0' || ch > '9') {
        --iter;
        break;
      }
      if (value > decimal_max) {
        exp_offset += static_cast<int32_t>(!decimal_found);
      } else {
        value = (value * 10) + static_cast<uint64_t>(ch - '0');
        exp_offset -= static_cast<int32_t>(decimal_found);
      }
    }
    return {value, exp_offset};
  }

  /**
   * @brief Return the exponent of a decimal string.
   *
   * This should only be called after the exponent 'e' notation was detected.
   * The return is the exponent (base-10) integer and can only be
   * invalid if `check_only == true` and invalid characters are found or the
   * exponent overflows an int32.
   *
   * @tparam check_only Set to true to verify the characters are valid and the
   *         exponent value in the decimal string does not overflow int32
   * @param[in,out] iter Start of characters to parse
   *                     (points to the character after the 'E' or 'e')
   * @param[in] end End of characters to parse
   * @return Integer value of the exponent
   */
  template <bool check_only = false>
  __device__ thrust::optional<int32_t> parse_exponent(char const* iter, char const* iter_end) const
  {
    constexpr uint32_t exponent_max = static_cast<uint32_t>(std::numeric_limits<int32_t>::max());

    // get optional exponent sign
    int32_t const exp_sign = [&iter] {
      auto const ch = *iter;
      if (ch != '-' && ch != '+') { return 1; }
      ++iter;
      return (ch == '-' ? -1 : 1);
    }();

    // parse exponent integer
    int32_t exp_ten = 0;
    while (iter < iter_end) {
      auto const ch = *iter++;
      if (ch < '0' || ch > '9') {
        if (check_only) { return thrust::nullopt; }
        break;
      }

      uint32_t exp_check = static_cast<uint32_t>(exp_ten * 10) + static_cast<uint32_t>(ch - '0');
      if (check_only && (exp_check > exponent_max)) { return thrust::nullopt; }  // check overflow
      exp_ten = static_cast<int32_t>(exp_check);
    }

    return exp_ten * exp_sign;
  }
};

/**
 * @brief Converts strings into an integers and records decimal places.
 *
 * The conversion uses the provided scale to build the resulting
 * integer. This can prevent overflow for strings with many digits.
 */
template <typename DecimalType>
struct string_to_decimal_fn : string_to_decimal_base {
  column_device_view const d_strings;
  int32_t const scale;

  string_to_decimal_fn(column_device_view const& d_strings, int32_t scale)
    : d_strings(d_strings), scale(scale)
  {
  }

  __device__ DecimalType operator()(size_type idx) const
  {
    if (d_strings.is_null(idx)) { return 0; }
    auto const d_str = d_strings.element<string_view>(idx);
    if (d_str.empty()) { return 0; }

    auto const sign = [&] {
      if (d_str.data()[0] == '-') { return -1; }
      if (d_str.data()[0] == '+') { return 1; }
      return 0;
    }();
    auto iter = d_str.data() + (sign != 0);

    auto const iter_end = d_str.data() + d_str.size_bytes();

    auto [value, exp_offset] = parse_integer(iter, iter_end);
    if (value == 0) { return DecimalType{0}; }

    // check for exponent
    int32_t exp_ten = 0;
    if ((iter < iter_end) && (*iter == 'e' || *iter == 'E')) {
      ++iter;
      if (iter < iter_end) { exp_ten = parse_exponent<false>(iter, iter_end).value(); }
    }
    exp_ten += exp_offset;

    // shift the output value based on the exp_ten and the scale values
    if (exp_ten < scale) {
      value = value / static_cast<uint64_t>(exp10(static_cast<double>(scale - exp_ten)));
    } else {
      value = value * static_cast<uint64_t>(exp10(static_cast<double>(exp_ten - scale)));
    }

    return static_cast<DecimalType>(value) * (sign == 0 ? 1 : sign);
  }
};

/**
 * @brief This only checks the string format for valid decimal characters.
 *
 * This follows closely the logic above but just ensures there are valid
 * characters for conversion and the integer component does not overflow.
 */
template <typename DecimalType>
struct string_to_decimal_check_fn : string_to_decimal_base {
  column_device_view const d_strings;
  int32_t const scale;

  string_to_decimal_check_fn(column_device_view const& d_strings, int32_t scale)
    : d_strings(d_strings), scale(scale)
  {
  }

  __device__ bool operator()(size_type idx) const
  {
    if (d_strings.is_null(idx)) { return false; }
    auto const d_str = d_strings.element<string_view>(idx);
    if (d_str.empty()) { return false; }

    auto iter = d_str.data() + static_cast<int>((d_str.data()[0] == '-' || d_str.data()[0] == '+'));

    auto const iter_end = d_str.data() + d_str.size_bytes();

    auto [value, exp_offset] = parse_integer(iter, iter_end);

    // only exponent notation is expected here
    if ((iter < iter_end) && (*iter != 'e' && *iter != 'E')) { return false; }
    ++iter;

    int32_t exp_ten = 0;  // check exponent overflow
    if (iter < iter_end) {
      auto exp_result = parse_exponent<true>(iter, iter_end);
      if (!exp_result) { return false; }
      exp_ten = exp_result.value();
    }
    exp_ten += exp_offset;

    // finally, check for overflow based on the exp_ten and scale values
    return (exp_ten < scale)
             ? true
             : value <= static_cast<uint64_t>(
                          std::numeric_limits<DecimalType>::max() /
                          static_cast<DecimalType>(exp10(static_cast<double>(exp_ten - scale))));
  }
};

/**
 * @brief The dispatch function for converting strings column to fixed-point column.
 */
struct dispatch_to_fixed_point_fn {
  template <typename T, std::enable_if_t<cudf::is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<column> operator()(strings_column_view const& input,
                                     data_type output_type,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    using DecimalType = device_storage_type_t<T>;

    auto const d_column = column_device_view::create(input.parent(), stream);

    // create output column
    auto results   = make_fixed_point_column(output_type,
                                           input.size(),
                                           cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                           input.null_count(),
                                           stream,
                                           mr);
    auto d_results = results->mutable_view().data<DecimalType>();

    // convert strings into decimal values
    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(input.size()),
                      d_results,
                      string_to_decimal_fn<DecimalType>{*d_column, output_type.scale()});
    results->set_null_count(input.null_count());
    return results;
  }

  template <typename T, std::enable_if_t<not cudf::is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<column> operator()(strings_column_view const&,
                                     data_type,
                                     rmm::cuda_stream_view,
                                     rmm::mr::device_memory_resource*) const
  {
    CUDF_FAIL("Output for to_fixed_point must be a decimal type.");
  }
};

}  // namespace

// This will convert a strings column into any integer column type.
std::unique_ptr<column> to_fixed_point(strings_column_view const& input,
                                       data_type output_type,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  if (input.is_empty()) return make_empty_column(output_type);
  return type_dispatcher(output_type, dispatch_to_fixed_point_fn{}, input, output_type, stream, mr);
}

}  // namespace detail

// external API
std::unique_ptr<column> to_fixed_point(strings_column_view const& strings,
                                       data_type output_type,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::to_fixed_point(strings, output_type, rmm::cuda_stream_default, mr);
}

namespace detail {
namespace {
/**
 * @brief Calculate the size of the each string required for
 * converting each value in base-10 format.
 *
 * ouput format is [-]integer.fraction
 */
template <typename DecimalType>
struct decimal_to_string_size_fn {
  column_device_view const d_column;

  __device__ int32_t operator()(size_type idx) const
  {
    if (d_column.is_null(idx)) return 0;
    auto const value = d_column.element<DecimalType>(idx);
    auto const scale = d_column.type().scale();

    if (scale >= 0) return count_digits(value) + scale;

    auto const abs_value = std::abs(value);
    auto const exp_ten   = static_cast<int64_t>(exp10(static_cast<double>(-scale)));
    auto const fraction  = count_digits(abs_value % exp_ten);
    auto const num_zeros = std::max(0, (-scale - fraction));
    return static_cast<int32_t>(value < 0) +    // sign if negative
           count_digits(abs_value / exp_ten) +  // integer
           1 +                                  // decimal point
           num_zeros +                          // zeros padding
           fraction;                            // size of fraction
  }
};

/**
 * @brief Convert each value into a string.
 *
 * The value is converted into base-10 digits [0-9]
 * plus the decimal point and a negative sign prefix.
 */
template <typename DecimalType>
struct decimal_to_string_fn {
  column_device_view const d_column;
  int32_t const* d_offsets;
  char* d_chars;

  __device__ void operator()(size_type idx)
  {
    if (d_column.is_null(idx)) return;
    auto const value = d_column.element<DecimalType>(idx);
    auto const scale = d_column.type().scale();
    char* d_buffer   = d_chars + d_offsets[idx];

    if (scale >= 0) {
      d_buffer += integer_to_string(value, d_buffer);
      thrust::generate_n(thrust::seq, d_buffer, scale, []() { return '0'; });  // add zeros
      return;
    }

    // scale < 0
    // write format:   [-]integer.fraction
    // where integer  = abs(value) / (10^abs(scale))
    //       fraction = abs(value) % (10^abs(scale))
    auto const abs_value = std::abs(value);
    if (value < 0) *d_buffer++ = '-';  // add sign
    auto const exp_ten   = static_cast<int64_t>(exp10(static_cast<double>(-scale)));
    auto const num_zeros = std::max(0, (-scale - count_digits(abs_value % exp_ten)));

    d_buffer += integer_to_string(abs_value / exp_ten, d_buffer);  // add the integer part
    *d_buffer++ = '.';                                             // add decimal point

    thrust::generate_n(thrust::seq, d_buffer, num_zeros, []() { return '0'; });  // add zeros
    d_buffer += num_zeros;

    integer_to_string(abs_value % exp_ten, d_buffer);  // add the fraction part
  }
};

/**
 * @brief The dispatcher functor for converting fixed-point values into strings.
 */
struct dispatch_from_fixed_point_fn {
  template <typename T, std::enable_if_t<cudf::is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& input,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    using DecimalType = device_storage_type_t<T>;  // underlying value type

    auto const d_column = column_device_view::create(input, stream);

    // build offsets column
    auto offsets_transformer_itr = cudf::detail::make_counting_transform_iterator(
      0, decimal_to_string_size_fn<DecimalType>{*d_column});
    auto offsets_column = detail::make_offsets_child_column(
      offsets_transformer_itr, offsets_transformer_itr + input.size(), stream, mr);
    auto const d_offsets = offsets_column->view().template data<int32_t>();

    // build chars column
    auto const bytes =
      cudf::detail::get_value<int32_t>(offsets_column->view(), input.size(), stream);
    auto chars_column = detail::create_chars_child_column(input.size(), bytes, stream, mr);
    auto d_chars      = chars_column->mutable_view().template data<char>();
    thrust::for_each_n(rmm::exec_policy(stream),
                       thrust::make_counting_iterator<size_type>(0),
                       input.size(),
                       decimal_to_string_fn<DecimalType>{*d_column, d_offsets, d_chars});

    return make_strings_column(input.size(),
                               std::move(offsets_column),
                               std::move(chars_column),
                               input.null_count(),
                               cudf::detail::copy_bitmask(input, stream, mr),
                               stream,
                               mr);
  }

  template <typename T, std::enable_if_t<not cudf::is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<column> operator()(column_view const&,
                                     rmm::cuda_stream_view,
                                     rmm::mr::device_memory_resource*) const
  {
    CUDF_FAIL("Values for from_fixed_point function must be a decimal type.");
  }
};

}  // namespace

std::unique_ptr<column> from_fixed_point(column_view const& input,
                                         rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource* mr)
{
  if (input.is_empty()) return detail::make_empty_strings_column(stream, mr);
  return type_dispatcher(input.type(), dispatch_from_fixed_point_fn{}, input, stream, mr);
}

}  // namespace detail

// external API

std::unique_ptr<column> from_fixed_point(column_view const& input,
                                         rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::from_fixed_point(input, rmm::cuda_stream_default, mr);
}

namespace detail {
namespace {

struct dispatch_is_fixed_point_fn {
  template <typename T, std::enable_if_t<cudf::is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<column> operator()(strings_column_view const& input,
                                     data_type decimal_type,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    using DecimalType = device_storage_type_t<T>;

    auto const d_column = column_device_view::create(input.parent(), stream);

    // create output column
    auto results   = make_numeric_column(data_type{type_id::BOOL8},
                                       input.size(),
                                       cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                       input.null_count(),
                                       stream,
                                       mr);
    auto d_results = results->mutable_view().data<bool>();

    // check strings for valid fixed-point chars
    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(input.size()),
                      d_results,
                      string_to_decimal_check_fn<DecimalType>{*d_column, decimal_type.scale()});
    results->set_null_count(input.null_count());
    return results;
  }

  template <typename T, std::enable_if_t<not cudf::is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<column> operator()(strings_column_view const&,
                                     data_type,
                                     rmm::cuda_stream_view,
                                     rmm::mr::device_memory_resource*) const
  {
    CUDF_FAIL("is_fixed_point is expecting a decimal type");
  }
};

}  // namespace

std::unique_ptr<column> is_fixed_point(strings_column_view const& input,
                                       data_type decimal_type,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  if (input.is_empty()) return cudf::make_empty_column(data_type{type_id::BOOL8});
  return type_dispatcher(
    decimal_type, dispatch_is_fixed_point_fn{}, input, decimal_type, stream, mr);
}
}  // namespace detail

std::unique_ptr<column> is_fixed_point(strings_column_view const& input,
                                       data_type decimal_type,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::is_fixed_point(input, decimal_type, rmm::cuda_stream_default, mr);
}

}  // namespace strings
}  // namespace cudf
