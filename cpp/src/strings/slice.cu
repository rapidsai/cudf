/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {
/**
 * @brief Function logic for compute_substrings_from_fn API
 *
 * This computes the output size and resolves the substring
 */
template <typename IndexIterator>
struct substring_from_fn {
  column_device_view const d_column;
  IndexIterator const starts;
  IndexIterator const stops;

  __device__ string_view operator()(size_type idx) const
  {
    if (d_column.is_null(idx)) { return string_view{nullptr, 0}; }
    auto const d_str  = d_column.template element<string_view>(idx);
    auto const length = d_str.length();
    auto const start  = std::max(starts[idx], 0);
    if (start >= length) { return string_view{}; }

    auto const stop = stops[idx];
    auto const end  = (((stop < 0) || (stop > length)) ? length : stop);
    return start < end ? d_str.substr(start, end - start) : string_view{};
  }

  substring_from_fn(column_device_view const& d_column, IndexIterator starts, IndexIterator stops)
    : d_column(d_column), starts(starts), stops(stops)
  {
  }
};

/**
 * @brief Function logic for the substring API.
 *
 * This will perform a substring operation on each string
 * using the provided start, stop, and step parameters.
 */
struct substring_fn {
  column_device_view const d_column;
  numeric_scalar_device_view<size_type> const d_start;
  numeric_scalar_device_view<size_type> const d_stop;
  numeric_scalar_device_view<size_type> const d_step;
  size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;

  __device__ void operator()(size_type idx)
  {
    if (d_column.is_null(idx)) {
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }
    auto const d_str  = d_column.template element<string_view>(idx);
    auto const length = d_str.length();
    if (length == 0) {
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }
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
      if (d_buffer) {
        d_buffer += from_char_utf8(*itr, d_buffer);
      } else {
        bytes += bytes_in_char_utf8(*itr);
      }
      itr += step;
    }
    if (!d_chars) { d_sizes[idx] = bytes; }
  }
};

/**
 * @brief Common utility function for the slice_strings APIs
 *
 * It wraps calling the functors appropriately to build the output strings column.
 *
 * The input iterators may have unique position values per string in `d_column`.
 * This can also be called with constant value iterators to handle special
 * slice functions if possible.
 *
 * @tparam IndexIterator Iterator type for character position values
 *
 * @param d_column Input strings column to substring
 * @param starts Start positions index iterator
 * @param stops Stop positions index iterator
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 */
template <typename IndexIterator>
std::unique_ptr<column> compute_substrings_from_fn(column_device_view const& d_column,
                                                   IndexIterator starts,
                                                   IndexIterator stops,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  auto results = rmm::device_uvector<string_view>(d_column.size(), stream);
  thrust::transform(rmm::exec_policy(stream),
                    thrust::counting_iterator<size_type>(0),
                    thrust::counting_iterator<size_type>(d_column.size()),
                    results.begin(),
                    substring_from_fn{d_column, starts, stops});
  return make_strings_column(results, string_view{nullptr, 0}, stream, mr);
}

}  // namespace

//
std::unique_ptr<column> slice_strings(strings_column_view const& strings,
                                      numeric_scalar<size_type> const& start,
                                      numeric_scalar<size_type> const& stop,
                                      numeric_scalar<size_type> const& step,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  if (strings.is_empty()) return make_empty_column(type_id::STRING);

  auto const step_valid = step.is_valid(stream);
  auto const step_value = step_valid ? step.value(stream) : 0;
  if (step_valid) { CUDF_EXPECTS(step_value != 0, "Step parameter must not be 0"); }

  auto const d_column = column_device_view::create(strings.parent(), stream);

  // optimization for (step==1 and start < stop) -- expect this to be most common
  if (step_value == 1 and start.is_valid(stream) and stop.is_valid(stream)) {
    auto const start_value = start.value(stream);
    auto const stop_value  = stop.value(stream);
    // note that any negative values here must use the alternate function below
    if ((start_value >= 0) && (start_value < stop_value)) {
      // this is about 2x faster on long strings for this common case
      return compute_substrings_from_fn(*d_column,
                                        thrust::constant_iterator<size_type>(start_value),
                                        thrust::constant_iterator<size_type>(stop_value),
                                        stream,
                                        mr);
    }
  }

  auto const d_start = get_scalar_device_view(const_cast<numeric_scalar<size_type>&>(start));
  auto const d_stop  = get_scalar_device_view(const_cast<numeric_scalar<size_type>&>(stop));
  auto const d_step  = get_scalar_device_view(const_cast<numeric_scalar<size_type>&>(step));

  auto [offsets, chars] = make_strings_children(
    substring_fn{*d_column, d_start, d_stop, d_step}, strings.size(), stream, mr);

  return make_strings_column(strings.size(),
                             std::move(offsets),
                             chars.release(),
                             strings.null_count(),
                             cudf::detail::copy_bitmask(strings.parent(), stream, mr));
}

std::unique_ptr<column> slice_strings(strings_column_view const& strings,
                                      column_view const& starts_column,
                                      column_view const& stops_column,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  size_type strings_count = strings.size();
  if (strings_count == 0) return make_empty_column(type_id::STRING);
  CUDF_EXPECTS(starts_column.size() == strings_count,
               "Parameter starts must have the same number of rows as strings.");
  CUDF_EXPECTS(stops_column.size() == strings_count,
               "Parameter stops must have the same number of rows as strings.");
  CUDF_EXPECTS(cudf::have_same_types(starts_column, stops_column),
               "Parameters starts and stops must be of the same type.",
               cudf::data_type_error);
  CUDF_EXPECTS(starts_column.null_count() == 0, "Parameter starts must not contain nulls.");
  CUDF_EXPECTS(stops_column.null_count() == 0, "Parameter stops must not contain nulls.");
  CUDF_EXPECTS(starts_column.type().id() != data_type{type_id::BOOL8}.id(),
               "Positions values must not be bool type.",
               cudf::data_type_error);
  CUDF_EXPECTS(is_fixed_width(starts_column.type()),
               "Positions values must be fixed width type.",
               cudf::data_type_error);

  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto starts_iter    = cudf::detail::indexalator_factory::make_input_iterator(starts_column);
  auto stops_iter     = cudf::detail::indexalator_factory::make_input_iterator(stops_column);
  return compute_substrings_from_fn(*strings_column, starts_iter, stops_iter, stream, mr);
}

}  // namespace detail

// external API

std::unique_ptr<column> slice_strings(strings_column_view const& strings,
                                      numeric_scalar<size_type> const& start,
                                      numeric_scalar<size_type> const& stop,
                                      numeric_scalar<size_type> const& step,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::slice_strings(strings, start, stop, step, stream, mr);
}

std::unique_ptr<column> slice_strings(strings_column_view const& strings,
                                      column_view const& starts_column,
                                      column_view const& stops_column,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::slice_strings(strings, starts_column, stops_column, stream, mr);
}

}  // namespace strings
}  // namespace cudf
