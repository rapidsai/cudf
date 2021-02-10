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
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <strings/convert/utilities.cuh>
#include <strings/utilities.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/extrema.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {
/**
 * @brief Converts strings into an integers and records decimal places.
 */
template <typename DecimalType>
struct string_to_decimal_fn {
  column_device_view const d_strings;
  DecimalType* values;
  int32_t* scales;

  __device__ void operator()(size_type idx)
  {
    values[idx] = DecimalType{0};
    scales[idx] = numeric::scale_type{0};

    if (d_strings.is_null(idx)) return;
    auto const d_str = d_strings.element<string_view>(idx);
    if (d_str.empty()) return;

    auto const sign = [&] {
      auto const first = d_str.data();
      if (*first == '-') return -1;
      if (*first == '+') return 1;
      return 0;
    }();
    auto iter = d_str.begin() + (sign != 0);

    int64_t value      = 0;
    bool decimal_found = false;
    while (iter != d_str.end()) {
      auto const chr = *iter++;
      if (chr >= '0' && chr <= '9') {
        value = (value * 10) + static_cast<int64_t>(chr - '0');
        scales[idx] -= static_cast<int32_t>(decimal_found);
      } else if (chr == '.')
        decimal_found = true;
      else
        break;
    }

    values[idx] = static_cast<DecimalType>(value * (sign == 0 ? 1 : sign));
  }
};

/**
 * @brief This only checks the string format for valid decimal characters.
 */
template <typename DecimalType>
struct string_to_decimal_check_fn {
  column_device_view const d_strings;

  __device__ bool operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) return false;
    auto const d_str = d_strings.element<string_view>(idx);
    if (d_str.empty()) return false;

    auto iter = d_str.begin() + (d_str.data()[0] == '-' || d_str.data()[0] == '+');

    DecimalType value  = 0;
    bool decimal_found = false;
    while (iter != d_str.end()) {
      auto const chr = *iter++;
      if (chr == '.' && !decimal_found) {
        decimal_found = true;
        continue;
      }
      if (chr < '0' || chr > '9') return false;
      auto const digit     = static_cast<DecimalType>(chr - '0');
      auto const max_check = (std::numeric_limits<DecimalType>::max() - digit) / DecimalType{10};
      if (value > max_check) return false;
      value = (value * DecimalType{10}) + digit;
    }
    return true;
  }
};

/**
 * @brief Uses the max_scale to rescale all the integer values.
 */
template <typename DecimalType>
struct rescale_decimals_fn {
  column_device_view const d_strings;
  int32_t const new_scale;
  int32_t const* scales;  // original scales
  DecimalType* values;    // values to rescale

  __device__ void operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) return;

    auto const fp = numeric::fixed_point<DecimalType, numeric::Radix::BASE_10>{
      numeric::scaled_integer<DecimalType>{values[idx], numeric::scale_type{scales[idx]}}};

    values[idx] = fp.rescaled(numeric::scale_type{new_scale}).value();
  }
};

/**
 * @brief The dispatch function for converting strings column to fixed-point column.
 */
struct dispatch_to_fixed_point_fn {
  template <typename T, std::enable_if_t<cudf::is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<column> operator()(strings_column_view const& input,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    using DecimalType = device_storage_type_t<T>;
    rmm::device_uvector<int32_t> d_scales(input.size(), stream);
    rmm::device_uvector<DecimalType> d_values(input.size(), stream, mr);
    auto d_column = column_device_view::create(input.parent(), stream);
    thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator<size_type>(0),
      input.size(),
      string_to_decimal_fn<DecimalType>{*d_column, d_values.data(), d_scales.data()});
    // find the largest scale size -- min is used since all scale values will be <= 0
    auto const min_elem =
      thrust::min_element(rmm::exec_policy(stream), d_scales.begin(), d_scales.end());
    auto const scale = d_scales.element(thrust::distance(d_scales.begin(), min_elem), stream);
    // re-scale all the values to the new scale
    thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator<size_type>(0),
      input.size(),
      rescale_decimals_fn<DecimalType>{*d_column, scale, d_scales.data(), d_values.data()});
    // build output column
    return std::make_unique<column>(data_type{type_to_id<T>(), scale},
                                    input.size(),
                                    d_values.release(),
                                    cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                    input.null_count());
  }

  template <typename T, std::enable_if_t<not cudf::is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<column> operator()(strings_column_view const&,
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
  return type_dispatcher(output_type, dispatch_to_fixed_point_fn{}, input, stream, mr);
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
    auto const exp_ten   = static_cast<int32_t>(exp10(static_cast<double>(-scale)));
    auto const num_zeros = std::max(0, (-scale - count_digits(abs_value % exp_ten)));
    return static_cast<int32_t>(value < 0) + count_digits(abs_value / exp_ten) + 1 + num_zeros +
           count_digits(abs_value % exp_ten);
  }
};

/**
 * @brief Convert each value into a string.
 *
 * The value is converted into base-10 using only characters [0-9].
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
      integer_to_string(value, d_buffer);
      d_buffer += count_digits(value);
      // add zeros
      thrust::generate_n(thrust::seq, d_buffer, scale, []() { return '0'; });
      return;
    }

    // scale < 0
    // write format:   [-]integer.fraction
    // where integer  = abs(value) / (10 ^ abs(scale))
    //       fraction = abs(value) % (10 ^ abs(scale))
    auto const abs_value = std::abs(value);
    if (value < 0) *d_buffer++ = '-';  // add sign
    auto const exp_ten   = static_cast<int32_t>(exp10(static_cast<double>(-scale)));
    auto const num_zeros = std::max(0, (-scale - count_digits(abs_value % exp_ten)));
    // add the integer part
    integer_to_string(abs_value / exp_ten, d_buffer);
    d_buffer += count_digits(abs_value / exp_ten);
    *d_buffer++ = '.';  // add decimal point
    // add zeros
    thrust::generate_n(thrust::seq, d_buffer, num_zeros, []() { return '0'; });
    d_buffer += num_zeros;
    // add the fractional part
    integer_to_string(abs_value % exp_ten, d_buffer);
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
    auto d_column     = column_device_view::create(input, stream);

    // build offsets column
    auto offsets_transformer_itr = cudf::detail::make_counting_transform_iterator(
      0, decimal_to_string_size_fn<DecimalType>{*d_column});
    auto offsets_column = detail::make_offsets_child_column(
      offsets_transformer_itr, offsets_transformer_itr + input.size(), stream, mr);
    auto d_offsets = offsets_column->view().template data<int32_t>();

    // build chars column
    auto const bytes =
      cudf::detail::get_value<int32_t>(offsets_column->view(), input.size(), stream);
    auto chars_column =
      detail::create_chars_child_column(input.size(), input.null_count(), bytes, stream, mr);
    auto chars_view = chars_column->mutable_view();
    auto d_chars    = chars_view.template data<char>();
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
    CUDF_FAIL("Values for from_fixed_point function must be an fixed-point type.");
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
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    using DecimalType = device_storage_type_t<T>;
    auto d_column     = column_device_view::create(input.parent(), stream);

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
                      string_to_decimal_check_fn<DecimalType>{*d_column});
    results->set_null_count(input.null_count());
    return results;
  }

  template <typename T, std::enable_if_t<not cudf::is_fixed_point<T>()>* = nullptr>
  std::unique_ptr<column> operator()(strings_column_view const&,
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
  return type_dispatcher(decimal_type, dispatch_is_fixed_point_fn{}, input, stream, mr);
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
