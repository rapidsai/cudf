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
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/copy.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {

string_scalar repeat_strings(string_scalar const& input,
                             size_type repeat_times,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr)
{
  if (!input.is_valid()) { return string_scalar("", false, stream, mr); }
  if (input.size() == 0 || repeat_times <= 0) { return string_scalar("", true, stream, mr); }
  if (repeat_times == 1) { return string_scalar(input, stream, mr); }

  CUDF_EXPECTS(input.size() <= std::numeric_limits<size_type>::max() / repeat_times,
               "The output string has size that exceeds the maximum allowed size.");

  auto buff =
    rmm::device_buffer(repeat_times * input.size(), stream, rmm::mr::get_current_device_resource());
  auto const iter = thrust::make_counting_iterator(0);
  thrust::transform(rmm::exec_policy(stream),
                    iter,
                    iter + repeat_times * input.size(),
                    static_cast<char*>(buff.data()),
                    [in_ptr = input.data(), str_size = input.size()] __device__(const auto idx) {
                      return in_ptr[idx % str_size];
                    });

  return string_scalar(string_view(static_cast<char*>(buff.data()), buff.size()), true, stream, mr);
}

namespace {
/**
 * @brief Functor to compute string sizes and repeat the input strings.
 */
template <bool has_nulls>
struct compute_size_and_repeat_fn {
  column_device_view const strings_dv;
  size_type repeat_times;

  offset_type* d_offsets{nullptr};

  // If d_chars == nullptr: only compute sizes of the output strings.
  // If d_chars != nullptr: only repeat strings.
  char* d_chars{nullptr};

  __device__ void operator()(size_type const idx) const noexcept
  {
    // If the number of repetitions is not positive, the output will be either an empty string,
    // or a null.
    if (repeat_times <= 0) {
      if (!d_chars) { d_offsets[idx] = 0; }
      return;
    }

    auto const is_valid = !has_nulls || strings_dv.is_valid_nocheck(idx);
    if (!d_chars) {
      d_offsets[idx] =
        is_valid ? repeat_times * strings_dv.element<string_view>(idx).size_bytes() : 0;
    } else if (is_valid) {
      auto const d_str = strings_dv.element<string_view>(idx);
      auto output_ptr  = d_chars + d_offsets[idx];
      for (size_type i = 0; i < repeat_times; ++i) {
        output_ptr = detail::copy_string(output_ptr, d_str);
      }
    }
  }
};

}  // namespace

std::unique_ptr<column> repeat_strings(strings_column_view const& input,
                                       size_type repeat_times,
                                       rmm::cuda_stream_view stream,
                                       rmm::mr::device_memory_resource* mr)
{
  auto const num_rows = input.size();
  if (num_rows == 0) { return detail::make_empty_strings_column(stream, mr); }
  if (num_rows == 1) { return std::make_unique<column>(input.parent(), stream, mr); }

  if (repeat_times > 0) {
    auto const size_start =
      cudf::detail::get_value<size_type>(input.offsets(), input.offset(), stream);
    auto const size_end =
      cudf::detail::get_value<size_type>(input.offsets(), input.offset() + num_rows, stream);
    CUDF_EXPECTS(size_end - size_start <= std::numeric_limits<size_type>::max() / repeat_times,
                 "The output strings have total size that exceeds the maximum allowed size.");
  }

  auto [offsets_column, chars_column] = [&] {
    auto const strings_dv_ptr = column_device_view::create(input.parent(), stream);
    if (input.has_nulls()) {
      auto const fn = compute_size_and_repeat_fn<true>{*strings_dv_ptr, repeat_times};
      return make_strings_children(fn, num_rows, stream, mr);
    } else {
      auto const fn = compute_size_and_repeat_fn<false>{*strings_dv_ptr, repeat_times};
      return make_strings_children(fn, num_rows, stream, mr);
    }
  }();

  return make_strings_column(num_rows,
                             std::move(offsets_column),
                             std::move(chars_column),
                             input.null_count(),
                             input.null_count()
                               ? cudf::detail::copy_bitmask(input.parent(), stream, mr)
                               : rmm::device_buffer{},
                             stream,
                             mr);
}

}  // namespace detail

string_scalar repeat_strings(string_scalar const& input,
                             size_type repeat_times,
                             rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::repeat_strings(input, repeat_times, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> repeat_strings(strings_column_view const& input,
                                       size_type repeat_times,
                                       rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::repeat_strings(input, repeat_times, rmm::cuda_stream_default, mr);
}

}  // namespace strings
}  // namespace cudf
