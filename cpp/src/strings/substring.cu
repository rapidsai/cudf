/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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
#include <cudf/copying.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/substring.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

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
  column_device_view const d_column;
  numeric_scalar_device_view<size_type> const d_start;
  numeric_scalar_device_view<size_type> const d_stop;
  numeric_scalar_device_view<size_type> const d_step;
  int32_t* d_offsets{};
  char* d_chars{};

  __device__ void operator()(size_type idx)
  {
    if (d_column.is_null(idx)) {
      if (!d_chars) d_offsets[idx] = 0;
      return;
    }
    auto const d_str  = d_column.template element<string_view>(idx);
    auto const length = d_str.length();
    if (length == 0) {
      if (!d_chars) d_offsets[idx] = 0;
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
      bytes += bytes_in_char_utf8(*itr);
      if (d_buffer) d_buffer += from_char_utf8(*itr, d_buffer);
      itr += step;
    }
    if (!d_chars) d_offsets[idx] = bytes;
  }
};

}  // namespace

//
std::unique_ptr<column> slice_strings(
  strings_column_view const& strings,
  numeric_scalar<size_type> const& start = numeric_scalar<size_type>(0, false),
  numeric_scalar<size_type> const& stop  = numeric_scalar<size_type>(0, false),
  numeric_scalar<size_type> const& step  = numeric_scalar<size_type>(1),
  rmm::cuda_stream_view stream           = cudf::default_stream_value,
  rmm::mr::device_memory_resource* mr    = rmm::mr::get_current_device_resource())
{
  if (strings.is_empty()) return make_empty_column(type_id::STRING);

  if (step.is_valid(stream)) CUDF_EXPECTS(step.value(stream) != 0, "Step parameter must not be 0");

  auto const d_column = column_device_view::create(strings.parent(), stream);
  auto const d_start  = get_scalar_device_view(const_cast<numeric_scalar<size_type>&>(start));
  auto const d_stop   = get_scalar_device_view(const_cast<numeric_scalar<size_type>&>(stop));
  auto const d_step   = get_scalar_device_view(const_cast<numeric_scalar<size_type>&>(step));

  auto children = make_strings_children(
    substring_fn{*d_column, d_start, d_stop, d_step}, strings.size(), stream, mr);

  return make_strings_column(strings.size(),
                             std::move(children.first),
                             std::move(children.second),
                             strings.null_count(),
                             cudf::detail::copy_bitmask(strings.parent(), stream, mr));
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
  return detail::slice_strings(strings, start, stop, step, cudf::default_stream_value, mr);
}

namespace detail {
namespace {

/**
 * @brief Function logic for substring_from API.
 *
 * This both calculates the output size and executes the substring.
 */
struct substring_from_fn {
  column_device_view const d_column;
  cudf::detail::input_indexalator const starts;
  cudf::detail::input_indexalator const stops;
  int32_t* d_offsets{};
  char* d_chars{};

  __device__ void operator()(size_type idx)
  {
    if (d_column.is_null(idx)) {
      if (!d_chars) d_offsets[idx] = 0;
      return;
    }
    auto const d_str  = d_column.template element<string_view>(idx);
    auto const length = d_str.length();
    auto const start  = std::max(starts[idx], 0);
    if (start >= length) {
      if (!d_chars) d_offsets[idx] = 0;
      return;
    }
    auto const stop = stops[idx];
    auto const end  = (((stop < 0) || (stop > length)) ? length : stop);

    auto const d_substr = d_str.substr(start, end - start);
    if (d_chars)
      memcpy(d_chars + d_offsets[idx], d_substr.data(), d_substr.size_bytes());
    else
      d_offsets[idx] = d_substr.size_bytes();
  }
};

/**
 * @brief Common utility function for the slice_strings APIs.
 *
 * It wraps calling the functors appropriately to build the output strings column.
 *
 * The input iterators may have unique position values per string in `d_column`.
 *
 * @param d_column Input strings column to substring.
 * @param null_count Number of nulls for the output column.
 * @param starts Start positions index iterator.
 * @param stops Stop positions index iterator.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 */
std::unique_ptr<column> compute_substrings_from_fn(column_device_view const& d_column,
                                                   size_type null_count,
                                                   cudf::detail::input_indexalator starts,
                                                   cudf::detail::input_indexalator stops,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::mr::device_memory_resource* mr)
{
  auto strings_count = d_column.size();

  // Copy the null mask
  rmm::device_buffer null_mask =
    !d_column.nullable()
      ? rmm::device_buffer{0, stream, mr}
      : rmm::device_buffer(
          d_column.null_mask(), cudf::bitmask_allocation_size_bytes(strings_count), stream, mr);

  auto children =
    make_strings_children(substring_from_fn{d_column, starts, stops}, strings_count, stream, mr);

  return make_strings_column(strings_count,
                             std::move(children.first),
                             std::move(children.second),
                             null_count,
                             std::move(null_mask));
}

/**
 * @brief Compute slice indices for each string.
 *
 * When slice_strings is invoked with a delimiter string and a delimiter count, we need to
 * compute the start and end indices of the substring. This function accomplishes that.
 */
template <typename DelimiterItrT>
void compute_substring_indices(column_device_view const& d_column,
                               DelimiterItrT const delim_itr,
                               size_type delimiter_count,
                               size_type* start_char_pos,
                               size_type* end_char_pos,
                               rmm::cuda_stream_view stream,
                               rmm::mr::device_memory_resource*)
{
  auto strings_count = d_column.size();

  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<size_type>(0),
    strings_count,
    [delim_itr, delimiter_count, start_char_pos, end_char_pos, d_column] __device__(size_type idx) {
      auto const& delim_val_pair = delim_itr[idx];
      auto const& delim_val      = delim_val_pair.first;  // Don't use it yet

      // If the column value for this row is null, result is null.
      // If the delimiter count is 0, result is empty string.
      // If the global delimiter or the row specific delimiter is invalid or if it is empty, row
      // value is empty.
      if (d_column.is_null(idx) || !delim_val_pair.second || delim_val.empty()) return;
      auto const& col_val = d_column.element<string_view>(idx);

      // If the column value for the row is empty, the row value is empty.
      if (!col_val.empty()) {
        auto const col_val_len   = col_val.length();
        auto const delimiter_len = delim_val.length();

        auto nsearches           = (delimiter_count < 0) ? -delimiter_count : delimiter_count;
        bool const left_to_right = (delimiter_count > 0);

        size_type start_pos = start_char_pos[idx];
        size_type end_pos   = col_val_len;
        size_type char_pos  = -1;

        end_char_pos[idx] = col_val_len;

        for (auto i = 0; i < nsearches; ++i) {
          char_pos = left_to_right ? col_val.find(delim_val, start_pos)
                                   : col_val.rfind(delim_val, 0, end_pos);
          if (char_pos == string_view::npos) return;
          if (left_to_right)
            start_pos = char_pos + delimiter_len;
          else
            end_pos = char_pos;
        }
        if (left_to_right)
          end_char_pos[idx] = char_pos;
        else
          start_char_pos[idx] = end_pos + delimiter_len;
      }
    });
}

}  // namespace

//
std::unique_ptr<column> slice_strings(
  strings_column_view const& strings,
  column_view const& starts_column,
  column_view const& stops_column,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  size_type strings_count = strings.size();
  if (strings_count == 0) return make_empty_column(type_id::STRING);
  CUDF_EXPECTS(starts_column.size() == strings_count,
               "Parameter starts must have the same number of rows as strings.");
  CUDF_EXPECTS(stops_column.size() == strings_count,
               "Parameter stops must have the same number of rows as strings.");
  CUDF_EXPECTS(starts_column.type() == stops_column.type(),
               "Parameters starts and stops must be of the same type.");
  CUDF_EXPECTS(starts_column.null_count() == 0, "Parameter starts must not contain nulls.");
  CUDF_EXPECTS(stops_column.null_count() == 0, "Parameter stops must not contain nulls.");
  CUDF_EXPECTS(starts_column.type().id() != data_type{type_id::BOOL8}.id(),
               "Positions values must not be bool type.");
  CUDF_EXPECTS(is_fixed_width(starts_column.type()), "Positions values must be fixed width type.");

  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto starts_iter    = cudf::detail::indexalator_factory::make_input_iterator(starts_column);
  auto stops_iter     = cudf::detail::indexalator_factory::make_input_iterator(stops_column);
  return compute_substrings_from_fn(
    *strings_column, strings.null_count(), starts_iter, stops_iter, stream, mr);
}

template <typename DelimiterItrT>
std::unique_ptr<column> slice_strings(strings_column_view const& strings,
                                      DelimiterItrT const delimiter_itr,
                                      size_type count,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{
  auto strings_count = strings.size();
  // If there aren't any rows, return an empty strings column
  if (strings_count == 0) return empty_like(strings.parent());

  // Compute the substring indices first
  auto start_chars_pos_vec = make_column_from_scalar(numeric_scalar<size_type>(0, true, stream),
                                                     strings_count,
                                                     stream,
                                                     rmm::mr::get_current_device_resource());
  auto stop_chars_pos_vec  = make_column_from_scalar(numeric_scalar<size_type>(0, true, stream),
                                                    strings_count,
                                                    stream,
                                                    rmm::mr::get_current_device_resource());

  auto start_char_pos = start_chars_pos_vec->mutable_view().data<size_type>();
  auto end_char_pos   = stop_chars_pos_vec->mutable_view().data<size_type>();

  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_column       = *strings_column;

  // If delimiter count is 0, the output column will contain empty strings
  if (count != 0) {
    // Compute the substring indices first
    compute_substring_indices(
      d_column, delimiter_itr, count, start_char_pos, end_char_pos, stream, mr);
  }

  // Extract the substrings using the indices next
  auto starts_iter =
    cudf::detail::indexalator_factory::make_input_iterator(start_chars_pos_vec->view());
  auto stops_iter =
    cudf::detail::indexalator_factory::make_input_iterator(stop_chars_pos_vec->view());
  return compute_substrings_from_fn(
    d_column, strings.null_count(), starts_iter, stops_iter, stream, mr);
}

std::unique_ptr<column> slice_strings(strings_column_view const& strings,
                                      strings_column_view const& delimiters,
                                      size_type count,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(strings.size() == delimiters.size(),
               "Strings and delimiters column sizes do not match");
  auto delimiters_dev_view_ptr = cudf::column_device_view::create(delimiters.parent(), stream);
  auto delimiters_dev_view     = *delimiters_dev_view_ptr;
  return (delimiters_dev_view.nullable())
           ? detail::slice_strings(
               strings,
               cudf::detail::make_pair_iterator<string_view, true>(delimiters_dev_view),
               count,
               stream,
               mr)
           : detail::slice_strings(
               strings,
               cudf::detail::make_pair_iterator<string_view, false>(delimiters_dev_view),
               count,
               stream,
               mr);
}

}  // namespace detail

// external API

std::unique_ptr<column> slice_strings(strings_column_view const& strings,
                                      column_view const& starts_column,
                                      column_view const& stops_column,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::slice_strings(
    strings, starts_column, stops_column, cudf::default_stream_value, mr);
}

std::unique_ptr<column> slice_strings(strings_column_view const& strings,
                                      string_scalar const& delimiter,
                                      size_type count,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::slice_strings(strings,
                               cudf::detail::make_pair_iterator<string_view>(delimiter),
                               count,
                               cudf::default_stream_value,
                               mr);
}

std::unique_ptr<column> slice_strings(strings_column_view const& strings,
                                      strings_column_view const& delimiters,
                                      size_type count,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::slice_strings(strings, delimiters, count, cudf::default_stream_value, mr);
}

}  // namespace strings
}  // namespace cudf
