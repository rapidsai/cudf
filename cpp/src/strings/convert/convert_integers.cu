/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/strings/detail/converters.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <strings/convert/utilities.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {

namespace detail {
namespace {

/**
 * @brief This only checks if a string is a valid integer within the bounds of its storage type.
 */
template <typename IntegerType>
struct string_to_integer_check_fn {
  __device__ bool operator()(thrust::pair<string_view, bool> const& p) const
  {
    if (!p.second || p.first.empty()) { return false; }

    auto const d_str = p.first.data();
    if (d_str[0] == '-' && std::is_unsigned<IntegerType>::value) { return false; }

    auto iter           = d_str + static_cast<int>((d_str[0] == '-' || d_str[0] == '+'));
    auto const iter_end = d_str + p.first.size_bytes();
    if (iter == iter_end) { return false; }

    auto const sign = d_str[0] == '-' ? IntegerType{-1} : IntegerType{1};
    auto const bound_val =
      sign > 0 ? std::numeric_limits<IntegerType>::max() : std::numeric_limits<IntegerType>::min();

    IntegerType value = 0;      // parse the string to integer and check for overflow along the way
    while (iter != iter_end) {  // check all bytes for valid characters
      auto const chr = *iter++;
      // Check for valid character
      if (chr < '0' || chr > '9') { return false; }

      // Check for underflow and overflow:
      auto const digit       = static_cast<IntegerType>(chr - '0');
      auto const bound_check = (bound_val - sign * digit) / IntegerType{10} * sign;
      if (value > bound_check) return false;
      value = value * IntegerType{10} + digit;
    }

    return true;
  }
};

/**
 * @brief The dispatch functions for checking if strings are valid integers.
 */
struct dispatch_is_integer_fn {
  template <typename T, std::enable_if_t<std::is_integral<T>::value>* = nullptr>
  std::unique_ptr<column> operator()(strings_column_view const& strings,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    auto const d_column = column_device_view::create(strings.parent(), stream);
    auto results        = make_numeric_column(data_type{type_id::BOOL8},
                                       strings.size(),
                                       cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                                       strings.null_count(),
                                       stream,
                                       mr);

    auto d_results = results->mutable_view().data<bool>();
    if (strings.has_nulls()) {
      thrust::transform(rmm::exec_policy(stream),
                        d_column->pair_begin<string_view, true>(),
                        d_column->pair_end<string_view, true>(),
                        d_results,
                        string_to_integer_check_fn<T>{});
    } else {
      thrust::transform(rmm::exec_policy(stream),
                        d_column->pair_begin<string_view, false>(),
                        d_column->pair_end<string_view, false>(),
                        d_results,
                        string_to_integer_check_fn<T>{});
    }

    // Calling mutable_view() on a column invalidates it's null count so we need to set it back
    results->set_null_count(strings.null_count());

    return results;
  }

  template <typename T, std::enable_if_t<not std::is_integral<T>::value>* = nullptr>
  std::unique_ptr<column> operator()(strings_column_view const&,
                                     rmm::cuda_stream_view,
                                     rmm::mr::device_memory_resource*) const
  {
    CUDF_FAIL("is_integer is expecting an integer type");
  }
};

}  // namespace

std::unique_ptr<column> is_integer(
  strings_column_view const& strings,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto const d_column = column_device_view::create(strings.parent(), stream);
  auto results        = make_numeric_column(data_type{type_id::BOOL8},
                                     strings.size(),
                                     cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                                     strings.null_count(),
                                     stream,
                                     mr);

  auto d_results = results->mutable_view().data<bool>();
  if (strings.has_nulls()) {
    thrust::transform(
      rmm::exec_policy(stream),
      d_column->pair_begin<string_view, true>(),
      d_column->pair_end<string_view, true>(),
      d_results,
      [] __device__(auto const& p) { return p.second ? string::is_integer(p.first) : false; });
  } else {
    thrust::transform(
      rmm::exec_policy(stream),
      d_column->pair_begin<string_view, false>(),
      d_column->pair_end<string_view, false>(),
      d_results,
      [] __device__(auto const& p) { return p.second ? string::is_integer(p.first) : false; });
  }

  // Calling mutable_view() on a column invalidates it's null count so we need to set it back
  results->set_null_count(strings.null_count());

  return results;
}

std::unique_ptr<column> is_integer(
  strings_column_view const& strings,
  data_type int_type,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  if (strings.is_empty()) { return cudf::make_empty_column(data_type{type_id::BOOL8}); }
  return type_dispatcher(int_type, dispatch_is_integer_fn{}, strings, stream, mr);
}

}  // namespace detail

// external APIs
std::unique_ptr<column> is_integer(strings_column_view const& strings,
                                   rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::is_integer(strings, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> is_integer(strings_column_view const& strings,
                                   data_type int_type,
                                   rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::is_integer(strings, int_type, rmm::cuda_stream_default, mr);
}

namespace detail {
namespace {
/**
 * @brief Converts strings into an integers.
 *
 * Used by the dispatch method to convert to different integer types.
 */
template <typename IntegerType>
struct string_to_integer_fn {
  const column_device_view strings_column;  // strings to convert

  __device__ IntegerType operator()(size_type idx)
  {
    if (strings_column.is_null(idx)) return static_cast<IntegerType>(0);
    // the cast to IntegerType will create predictable results
    // for integers that are larger than the IntegerType can hold
    return static_cast<IntegerType>(string_to_integer(strings_column.element<string_view>(idx)));
  }
};

/**
 * @brief The dispatch functions for converting strings to integers.
 *
 * The output_column is expected to be one of the integer types only.
 */
struct dispatch_to_integers_fn {
  template <typename IntegerType, std::enable_if_t<std::is_integral<IntegerType>::value>* = nullptr>
  void operator()(column_device_view const& strings_column,
                  mutable_column_view& output_column,
                  rmm::cuda_stream_view stream) const
  {
    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(strings_column.size()),
                      output_column.data<IntegerType>(),
                      string_to_integer_fn<IntegerType>{strings_column});
  }
  // non-integral types throw an exception
  template <typename T, std::enable_if_t<not std::is_integral<T>::value>* = nullptr>
  void operator()(column_device_view const&, mutable_column_view&, rmm::cuda_stream_view) const
  {
    CUDF_FAIL("Output for to_integers must be an integral type.");
  }
};

template <>
void dispatch_to_integers_fn::operator()<bool>(column_device_view const&,
                                               mutable_column_view&,
                                               rmm::cuda_stream_view) const
{
  CUDF_FAIL("Output for to_integers must not be a boolean type.");
}

}  // namespace

// This will convert a strings column into any integer column type.
std::unique_ptr<column> to_integers(strings_column_view const& strings,
                                    data_type output_type,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  size_type strings_count = strings.size();
  if (strings_count == 0) return make_numeric_column(output_type, 0);

  // Create integer output column copying the strings null-mask
  auto results = make_numeric_column(output_type,
                                     strings_count,
                                     cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                                     strings.null_count(),
                                     stream,
                                     mr);
  // Fill output column with integers
  auto const strings_dev_view = column_device_view::create(strings.parent(), stream);
  auto results_view           = results->mutable_view();
  type_dispatcher(output_type, dispatch_to_integers_fn{}, *strings_dev_view, results_view, stream);

  // Calling mutable_view() on a column invalidates it's null count so we need to set it back
  results->set_null_count(strings.null_count());

  return results;
}

}  // namespace detail

// external API
std::unique_ptr<column> to_integers(strings_column_view const& strings,
                                    data_type output_type,
                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::to_integers(strings, output_type, rmm::cuda_stream_default, mr);
}

namespace detail {
namespace {
/**
 * @brief Calculate the size of the each string required for
 * converting each integer in base-10 format.
 */
template <typename IntegerType>
struct integer_to_string_size_fn {
  column_device_view d_column;

  __device__ size_type operator()(size_type idx)
  {
    if (d_column.is_null(idx)) return 0;
    IntegerType value = d_column.element<IntegerType>(idx);
    return count_digits(value);
  }
};

/**
 * @brief Convert each integer into a string.
 *
 * The integer is converted into base-10 using only characters [0-9].
 * No formatting is done for the string other than prepending the '-'
 * character for negative values.
 */
template <typename IntegerType>
struct integer_to_string_fn {
  column_device_view d_column;
  const int32_t* d_offsets;
  char* d_chars;

  __device__ void operator()(size_type idx)
  {
    if (d_column.is_null(idx)) return;
    IntegerType value = d_column.element<IntegerType>(idx);
    char* d_buffer    = d_chars + d_offsets[idx];
    integer_to_string(value, d_buffer);
  }
};

/**
 * @brief This dispatch method is for converting integers into strings.
 * The template function declaration ensures only integer types are used.
 */
struct dispatch_from_integers_fn {
  template <typename IntegerType, std::enable_if_t<std::is_integral<IntegerType>::value>* = nullptr>
  std::unique_ptr<column> operator()(column_view const& integers,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
  {
    size_type strings_count = integers.size();
    auto column             = column_device_view::create(integers, stream);
    auto d_column           = *column;

    // copy the null mask
    rmm::device_buffer null_mask = cudf::detail::copy_bitmask(integers, stream, mr);
    // build offsets column
    auto offsets_transformer_itr = thrust::make_transform_iterator(
      thrust::make_counting_iterator<int32_t>(0), integer_to_string_size_fn<IntegerType>{d_column});
    auto offsets_column = detail::make_offsets_child_column(
      offsets_transformer_itr, offsets_transformer_itr + strings_count, stream, mr);
    auto offsets_view  = offsets_column->view();
    auto d_new_offsets = offsets_view.template data<int32_t>();

    // build chars column
    auto const bytes  = cudf::detail::get_value<int32_t>(offsets_view, strings_count, stream);
    auto chars_column = detail::create_chars_child_column(bytes, stream, mr);
    auto chars_view   = chars_column->mutable_view();
    auto d_chars      = chars_view.template data<char>();
    thrust::for_each_n(rmm::exec_policy(stream),
                       thrust::make_counting_iterator<size_type>(0),
                       strings_count,
                       integer_to_string_fn<IntegerType>{d_column, d_new_offsets, d_chars});

    return make_strings_column(strings_count,
                               std::move(offsets_column),
                               std::move(chars_column),
                               integers.null_count(),
                               std::move(null_mask),
                               stream,
                               mr);
  }

  // non-integral types throw an exception
  template <typename T, std::enable_if_t<not std::is_integral<T>::value>* = nullptr>
  std::unique_ptr<column> operator()(column_view const&,
                                     rmm::cuda_stream_view,
                                     rmm::mr::device_memory_resource*) const
  {
    CUDF_FAIL("Values for from_integers function must be an integral type.");
  }
};

template <>
std::unique_ptr<column> dispatch_from_integers_fn::operator()<bool>(
  column_view const&, rmm::cuda_stream_view, rmm::mr::device_memory_resource*) const
{
  CUDF_FAIL("Input for from_integers must not be a boolean type.");
}

}  // namespace

// This will convert all integer column types into a strings column.
std::unique_ptr<column> from_integers(column_view const& integers,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{
  size_type strings_count = integers.size();
  if (strings_count == 0) return make_empty_column(data_type{type_id::STRING});

  return type_dispatcher(integers.type(), dispatch_from_integers_fn{}, integers, stream, mr);
}

}  // namespace detail

// external API
std::unique_ptr<column> from_integers(column_view const& integers,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::from_integers(integers, rmm::cuda_stream_default, mr);
}

}  // namespace strings
}  // namespace cudf
