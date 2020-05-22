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
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/substring.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <strings/utilities.cuh>

namespace {
/**
 * @brief Used as template parameter to divide size calculation from
 * the actual string operation within a function.
 *
 * Useful when most of the logic is identical for both passes.
 */
enum TwoPass {
  SizeOnly = 0,  ///< calculate the size only
  ExecuteOp      ///< run the string operation
};

}  // namespace

namespace cudf {
namespace strings {
namespace detail {
namespace {
/**
 * @brief Function logic for the substring API.
 *
 * This will perform a substring operation on each string
 * using the provided start, stop, and step parameters.
 */
struct substring_fn {
  const column_device_view d_column;
  numeric_scalar_device_view<size_type> d_start, d_stop, d_step;
  const int32_t* d_offsets{};
  char* d_chars{};

  __device__ cudf::size_type operator()(size_type idx)
  {
    if (d_column.is_null(idx)) return 0;  // null string
    string_view d_str = d_column.template element<string_view>(idx);
    auto const length = d_str.length();
    if (length == 0) return 0;  // empty string
    size_type const step = d_step.is_valid() ? d_step.value() : 1;
    auto const begin     = [&] {  // always inclusive
      // when invalid, default depends on step
      if (!d_start.is_valid()) return (step > 0) ? d_str.begin() : (d_str.end() - 1);
      // normal positive position logic
      auto start = d_start.value();
      if (start >= 0) {
        if (start < length) return d_str.begin() + start;
        return d_str.end() + (step < 0 ? -1 : 0);
      }
      // handle negative position here
      auto adjust = length + start;
      if (adjust >= 0) return d_str.begin() + adjust;
      return d_str.begin() + (step < 0 ? -1 : 0);
    }();
    auto const end = [&] {  // always exclusive
      // when invalid, default depends on step
      if (!d_stop.is_valid()) return step > 0 ? d_str.end() : (d_str.begin() - 1);
      // normal positive position logic
      auto stop = d_stop.value();
      if (stop >= 0) return (stop < length) ? (d_str.begin() + stop) : d_str.end();
      // handle negative position here
      auto adjust = length + stop;
      return d_str.begin() + (adjust >= 0 ? adjust : -1);
    }();

    size_type bytes = 0;
    char* d_buffer  = d_chars ? d_chars + d_offsets[idx] : nullptr;
    auto itr        = begin;
    while (step > 0 ? itr < end : end < itr) {
      bytes += bytes_in_char_utf8(*itr);
      if (d_buffer) d_buffer += from_char_utf8(*itr, d_buffer);
      itr += step;
    }
    return bytes;
  }
};

}  // namespace

//
std::unique_ptr<column> slice_strings(
  strings_column_view const& strings,
  numeric_scalar<size_type> const& start = numeric_scalar<size_type>(0, false),
  numeric_scalar<size_type> const& stop  = numeric_scalar<size_type>(0, false),
  numeric_scalar<size_type> const& step  = numeric_scalar<size_type>(1),
  rmm::mr::device_memory_resource* mr    = rmm::mr::get_default_resource(),
  cudaStream_t stream                    = 0)
{
  size_type strings_count = strings.size();
  if (strings_count == 0) return make_empty_strings_column(mr, stream);

  if (step.is_valid()) CUDF_EXPECTS(step.value(stream) != 0, "Step parameter must not be 0");

  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_column       = *strings_column;
  auto d_start        = get_scalar_device_view(const_cast<numeric_scalar<size_type>&>(start));
  auto d_stop         = get_scalar_device_view(const_cast<numeric_scalar<size_type>&>(stop));
  auto d_step         = get_scalar_device_view(const_cast<numeric_scalar<size_type>&>(step));

  // copy the null mask
  rmm::device_buffer null_mask = copy_bitmask(strings.parent(), stream, mr);

  // build offsets column
  auto offsets_transformer_itr = thrust::make_transform_iterator(
    thrust::make_counting_iterator<int32_t>(0), substring_fn{d_column, d_start, d_stop, d_step});
  auto offsets_column = make_offsets_child_column(
    offsets_transformer_itr, offsets_transformer_itr + strings_count, mr, stream);
  auto d_new_offsets = offsets_column->view().data<int32_t>();

  // build chars column
  size_type bytes   = thrust::device_pointer_cast(d_new_offsets)[strings_count];
  auto chars_column = strings::detail::create_chars_child_column(
    strings_count, strings.null_count(), bytes, mr, stream);
  auto d_chars = chars_column->mutable_view().data<char>();
  thrust::for_each_n(rmm::exec_policy(stream)->on(stream),
                     thrust::make_counting_iterator<size_type>(0),
                     strings_count,
                     substring_fn{d_column, d_start, d_stop, d_step, d_new_offsets, d_chars});
  //
  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             std::move(chars_column),
                             strings.null_count(),
                             std::move(null_mask),
                             stream,
                             mr);
}

}  // namespace detail

// external API

std::unique_ptr<column> slice_strings(strings_column_view const& strings,
                                      numeric_scalar<size_type> const& start,
                                      numeric_scalar<size_type> const& stop,
                                      numeric_scalar<size_type> const& step,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::slice_strings(strings, start, stop, step, mr);
}

namespace detail {
namespace {
template <typename PositionType, TwoPass Pass = SizeOnly>
struct substring_from_fn {
  const column_device_view d_column;
  const PositionType* starts;
  const PositionType* stops;
  const int32_t* d_offsets{};
  char* d_chars{};

  /**
   * @brief Function logic for substring_from API.
   * This does both calculate and the execute based on template parameter.
   */
  __device__ size_type operator()(size_type idx)
  {
    if (d_column.is_null(idx)) return 0;  // null string
    string_view d_str = d_column.template element<string_view>(idx);
    size_type length  = d_str.length();
    size_type start   = static_cast<size_type>(starts[idx]);
    if (start >= length) return 0;  // empty string
    size_type stop       = static_cast<size_type>(stops[idx]);
    size_type end        = (((stop < 0) || (stop > length)) ? length : stop);
    string_view d_substr = d_str.substr(start, end - start);
    if (Pass == SizeOnly)
      return d_substr.size_bytes();
    else {
      memcpy(d_chars + d_offsets[idx], d_substr.data(), d_substr.size_bytes());
      return 0;
    }
  }
};

/**
 * Called by the type-dispatcher for resolving the position columns
 * (starts_column and stops_column) to actual types.
 */
struct dispatch_substring_from_fn {
  /**
   * @brief Returns strings column with substrings based on the ranges in the
   * individual starts and stops column position values.
   */
  template <typename PositionType,
            std::enable_if_t<std::is_integral<PositionType>::value>* = nullptr>
  std::unique_ptr<column> operator()(strings_column_view const& strings,
                                     column_view const& starts_column,
                                     column_view const& stops_column,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream) const
  {
    const PositionType* starts = starts_column.data<PositionType>();
    const PositionType* stops  = stops_column.data<PositionType>();

    auto strings_count  = strings.size();
    auto strings_column = column_device_view::create(strings.parent(), stream);
    auto d_column       = *strings_column;

    // copy the null mask
    rmm::device_buffer null_mask;
    size_type null_count = strings.null_count();
    if (d_column.nullable())
      null_mask = rmm::device_buffer(
        d_column.null_mask(), cudf::bitmask_allocation_size_bytes(strings_count), stream, mr);
    // build offsets column
    auto offsets_transformer_itr =
      thrust::make_transform_iterator(thrust::make_counting_iterator<PositionType>(0),
                                      substring_from_fn<PositionType>{d_column, starts, stops});
    auto offsets_column = cudf::strings::detail::make_offsets_child_column(
      offsets_transformer_itr, offsets_transformer_itr + strings_count, mr, stream);
    auto offsets_view  = offsets_column->view();
    auto d_new_offsets = offsets_view.template data<int32_t>();

    // build chars column
    cudf::size_type bytes = thrust::device_pointer_cast(d_new_offsets)[strings_count];
    auto chars_column     = cudf::strings::detail::create_chars_child_column(
      strings_count, null_count, bytes, mr, stream);
    auto chars_view = chars_column->mutable_view();
    auto d_chars    = chars_view.template data<char>();
    thrust::for_each_n(
      rmm::exec_policy(stream)->on(stream),
      thrust::make_counting_iterator<cudf::size_type>(0),
      strings_count,
      substring_from_fn<PositionType, ExecuteOp>{d_column, starts, stops, d_new_offsets, d_chars});
    //
    return make_strings_column(strings_count,
                               std::move(offsets_column),
                               std::move(chars_column),
                               null_count,
                               std::move(null_mask),
                               stream,
                               mr);
  }
  //
  template <typename PositionType,
            std::enable_if_t<not std::is_integral<PositionType>::value>* = nullptr>
  std::unique_ptr<column> operator()(strings_column_view const&,
                                     column_view const&,
                                     column_view const&,
                                     rmm::mr::device_memory_resource*,
                                     cudaStream_t) const
  {
    CUDF_FAIL("Positions values must be an integral type.");
  }
};

template <>
std::unique_ptr<column> dispatch_substring_from_fn::operator()<bool>(
  strings_column_view const&,
  column_view const&,
  column_view const&,
  rmm::mr::device_memory_resource*,
  cudaStream_t) const
{
  CUDF_FAIL("Positions values must not be bool type.");
}

}  // namespace

//
std::unique_ptr<column> slice_strings(
  strings_column_view const& strings,
  column_view const& starts_column,
  column_view const& stops_column,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0)
{
  size_type strings_count = strings.size();
  if (strings_count == 0) return make_empty_strings_column(mr, stream);
  CUDF_EXPECTS(starts_column.size() == strings_count,
               "Parameter starts must have the same number of rows as strings.");
  CUDF_EXPECTS(stops_column.size() == strings_count,
               "Parameter stops must have the same number of rows as strings.");
  CUDF_EXPECTS(starts_column.type() == stops_column.type(),
               "Parameters starts and stops must be of the same type.");
  CUDF_EXPECTS(starts_column.null_count() == 0, "Parameter starts must not contain nulls.");
  CUDF_EXPECTS(stops_column.null_count() == 0, "Parameter stops must not contain nulls.");

  // perhaps another candidate for index-normalizer
  return cudf::type_dispatcher(starts_column.type(),
                               dispatch_substring_from_fn{},
                               strings,
                               starts_column,
                               stops_column,
                               mr,
                               stream);
}

}  // namespace detail

// external API

std::unique_ptr<column> slice_strings(strings_column_view const& strings,
                                      column_view const& starts_column,
                                      column_view const& stops_column,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::slice_strings(strings, starts_column, stops_column, mr);
}

}  // namespace strings
}  // namespace cudf
