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

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/pad_impl.cuh>
#include <cudf/strings/padding.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {

struct compute_pad_output_length_fn {
  column_device_view d_strings;
  size_type width;
  size_type fill_char_size;

  __device__ size_type operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) return 0;
    string_view d_str = d_strings.element<string_view>(idx);
    return compute_padded_size(d_str, width, fill_char_size);
  }
};

}  // namespace

std::unique_ptr<column> pad(
  strings_column_view const& strings,
  size_type width,
  side_type side                      = side_type::RIGHT,
  std::string_view fill_char          = " ",
  rmm::cuda_stream_view stream        = cudf::default_stream_value,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  size_type strings_count = strings.size();
  if (strings_count == 0) return make_empty_column(type_id::STRING);
  CUDF_EXPECTS(!fill_char.empty(), "fill_char parameter must not be empty");
  char_utf8 d_fill_char    = 0;
  size_type fill_char_size = to_char_utf8(fill_char.data(), d_fill_char);

  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;

  // create null_mask
  rmm::device_buffer null_mask = cudf::detail::copy_bitmask(strings.parent(), stream, mr);

  // build offsets column
  auto offsets_transformer_itr = cudf::detail::make_counting_transform_iterator(
    0, compute_pad_output_length_fn{d_strings, width, fill_char_size});
  auto offsets_column = make_offsets_child_column(
    offsets_transformer_itr, offsets_transformer_itr + strings_count, stream, mr);
  auto d_offsets = offsets_column->view().data<int32_t>();

  // build chars column
  auto const bytes =
    cudf::detail::get_value<int32_t>(offsets_column->view(), strings_count, stream);
  auto chars_column = strings::detail::create_chars_child_column(bytes, stream, mr);
  auto d_chars      = chars_column->mutable_view().data<char>();

  if (side == side_type::LEFT) {
    thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator<cudf::size_type>(0),
      strings_count,
      [d_strings, width, d_fill_char, d_offsets, d_chars] __device__(size_type idx) {
        if (d_strings.is_valid(idx)) {
          pad_impl<side_type::LEFT>(
            d_strings.element<string_view>(idx), width, d_fill_char, d_chars + d_offsets[idx]);
        }
      });
  } else if (side == side_type::RIGHT) {
    thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator<cudf::size_type>(0),
      strings_count,
      [d_strings, width, d_fill_char, d_offsets, d_chars] __device__(size_type idx) {
        if (d_strings.is_valid(idx)) {
          pad_impl<side_type::RIGHT>(
            d_strings.element<string_view>(idx), width, d_fill_char, d_chars + d_offsets[idx]);
        }
      });
  } else if (side == side_type::BOTH) {
    thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator<cudf::size_type>(0),
      strings_count,
      [d_strings, width, d_fill_char, d_offsets, d_chars] __device__(size_type idx) {
        if (d_strings.is_valid(idx)) {
          pad_impl<side_type::BOTH>(
            d_strings.element<string_view>(idx), width, d_fill_char, d_chars + d_offsets[idx]);
        }
      });
  }

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             std::move(chars_column),
                             strings.null_count(),
                             std::move(null_mask));
}

std::unique_ptr<column> zfill(
  strings_column_view const& input,
  size_type width,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  if (input.is_empty()) return make_empty_column(type_id::STRING);

  auto strings_column = column_device_view::create(input.parent(), stream);
  auto d_strings      = *strings_column;

  // build offsets column
  auto offsets_transformer_itr = thrust::make_transform_iterator(
    thrust::make_counting_iterator<int32_t>(0),
    compute_pad_output_length_fn{d_strings, width, 1});  // fillchar is 1 byte
  auto offsets_column = make_offsets_child_column(
    offsets_transformer_itr, offsets_transformer_itr + input.size(), stream, mr);
  auto const d_offsets = offsets_column->view().data<int32_t>();

  // build chars column
  auto const bytes = cudf::detail::get_value<int32_t>(offsets_column->view(), input.size(), stream);
  auto chars_column = strings::detail::create_chars_child_column(bytes, stream, mr);
  auto d_chars      = chars_column->mutable_view().data<char>();

  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<cudf::size_type>(0),
                     input.size(),
                     [d_strings, width, d_offsets, d_chars] __device__(size_type idx) {
                       if (d_strings.is_valid(idx)) {
                         zfill_impl(
                           d_strings.element<string_view>(idx), width, d_chars + d_offsets[idx]);
                       }
                     });

  return make_strings_column(input.size(),
                             std::move(offsets_column),
                             std::move(chars_column),
                             input.null_count(),
                             cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

}  // namespace detail

// Public APIs

std::unique_ptr<column> pad(strings_column_view const& input,
                            size_type width,
                            side_type side,
                            std::string_view fill_char,
                            rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::pad(input, width, side, fill_char, cudf::default_stream_value, mr);
}

std::unique_ptr<column> zfill(strings_column_view const& input,
                              size_type width,
                              rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::zfill(input, width, cudf::default_stream_value, mr);
}

}  // namespace strings
}  // namespace cudf
