/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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
#include <cudf/detail/copy.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/strings/detail/copying.hpp>
#include <cudf/strings/detail/utilities.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace cudf::strings::detail {

namespace {

struct adjust_offsets_fn {
  column_device_view const d_column;
  string_view const d_filler;
  size_type const offset;

  __device__ size_type operator()(size_type idx)
  {
    if (offset < 0) {
      auto const first      = d_column.element<size_type>(-offset);
      auto const last_index = d_column.size() + offset;
      if (idx < last_index) {
        return d_column.element<size_type>(idx - offset) - first;
      } else {
        auto const last = d_column.element<size_type>(d_column.size() - 1);
        return (last - first) + ((idx - last_index + 1) * d_filler.size_bytes());
      }
    } else {
      if (idx < offset) {
        return idx * d_filler.size_bytes();
      } else {
        auto const total_filler = d_filler.size_bytes() * offset;
        return total_filler + d_column.element<size_type>(idx - offset);
      }
    }
  }
};

struct shift_chars_fn {
  column_device_view const d_column;
  string_view const d_filler;
  size_type const offset;

  __device__ char operator()(size_type idx)
  {
    if (offset < 0) {
      auto const last_index = -offset;
      if (idx < last_index) {
        auto const first_index = d_column.size() + offset;
        return d_column.element<char>(idx + first_index);
      } else {
        auto const char_index = idx - last_index;
        return d_filler.data()[char_index % d_filler.size_bytes()];
      }
    } else {
      if (idx < offset) {
        return d_filler.data()[idx % d_filler.size_bytes()];
      } else {
        return d_column.element<char>(idx - offset);
      }
    }
  }
};

}  // namespace

std::unique_ptr<column> shift(strings_column_view const& input,
                              size_type offset,
                              scalar const& fill_value,
                              rmm::cuda_stream_view stream,
                              rmm::mr::device_memory_resource* mr)
{
  auto d_fill_str = static_cast<string_scalar const&>(fill_value).value(stream);

  // adjust offset when greater than the size of the input
  if (std::abs(offset) > input.size()) { offset = input.size(); }

  // output offsets column is the same size as the input
  auto const input_offsets =
    cudf::detail::slice(
      input.offsets(), {input.offset(), input.offset() + input.size() + 1}, stream)
      .front();
  auto const offsets_size = input_offsets.size();
  auto offsets_column     = cudf::detail::allocate_like(
    input_offsets, offsets_size, mask_allocation_policy::NEVER, stream, mr);

  // run kernel to simultaneously shift and adjust the values in the output offsets column
  auto d_offsets = mutable_column_device_view::create(offsets_column->mutable_view(), stream);
  auto const d_input_offsets = column_device_view::create(input_offsets, stream);
  thrust::transform(rmm::exec_policy(stream),
                    thrust::counting_iterator<size_type>(0),
                    thrust::counting_iterator<size_type>(offsets_size),
                    d_offsets->data<size_type>(),
                    adjust_offsets_fn{*d_input_offsets, d_fill_str, offset});

  // compute the shift-offset for the output characters child column
  auto const shift_offset = [&] {
    auto const index = (offset >= 0) ? offset : offsets_size - 1 + offset;
    return (offset < 0 ? -1 : 1) *
           cudf::detail::get_value<size_type>(offsets_column->view(), index, stream);
  }();

  // create output chars child column
  auto const chars_size =
    cudf::detail::get_value<size_type>(offsets_column->view(), offsets_size - 1, stream);
  auto chars_column = create_chars_child_column(chars_size, stream, mr);
  auto d_chars      = mutable_column_device_view::create(chars_column->mutable_view(), stream);
  auto const d_input_chars = column_device_view::create(input.chars(), stream);

  // run kernel to shift the characters
  thrust::transform(rmm::exec_policy(stream),
                    thrust::counting_iterator<size_type>(0),
                    thrust::counting_iterator<size_type>(chars_size),
                    d_chars->data<char>(),
                    shift_chars_fn{*d_input_chars, d_fill_str, shift_offset});

  // caller sets the null-mask
  return make_strings_column(
    input.size(), std::move(offsets_column), std::move(chars_column), 0, rmm::device_buffer{});
}

}  // namespace cudf::strings::detail
