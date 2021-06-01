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
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/copy.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace strings {
namespace detail {

string_scalar repeat_join(string_scalar const& input,
                          size_type repeat_times,
                          rmm::cuda_stream_view stream,
                          rmm::mr::device_memory_resource* mr)
{
  // TODO: handle err

  auto new_str =
    rmm::device_buffer(repeat_times * input.size(), stream, rmm::mr::get_current_device_resource());
  for (size_type i = 0, str_size = input.size(); i < repeat_times; ++i) {
    auto const offset = i * str_size;
    thrust::copy(rmm::exec_policy(stream),
                 input.data(),
                 input.data() + str_size,
                 static_cast<char*>(new_str.data()) + offset);
  }

  return string_scalar(
    string_view(static_cast<char* const>(new_str.data()), new_str.size()), true, stream, mr);
}

namespace {
/**
 * @brief
 */
template <bool has_nulls>
struct compute_size_and_repeat_fn {
  column_device_view const strings_dv;
  size_type repeat_times;

  offset_type* d_offsets{nullptr};

  // If d_chars == nullptr: only compute sizes and validities of the output strings.
  // If d_chars != nullptr: only repeat strings.
  char* d_chars{nullptr};

  // We need to set `1` or `0` for the validities of the output strings.
  int8_t* d_validities{nullptr};

  __device__ void operator()(size_type const idx) const noexcept
  {
    if (!d_chars) {
      auto const is_valid = !has_nulls || strings_dv.is_valid_nocheck(idx);
      d_offsets[idx] =
        is_valid ? repeat_times * strings_dv.element<string_view>(idx).size_bytes() : 0;
      if constexpr (has_nulls) { d_validities[idx] = is_valid; }
    } else if (!has_nulls || d_validities[idx]) {
      auto const d_str = strings_dv.element<string_view>(idx);
      auto output_ptr  = d_chars + d_offsets[idx];
      for (size_type i = 0; i < repeat_times; ++i) {
        output_ptr = detail::copy_string(output_ptr, d_str);
      }
    }
  }
};

}  // namespace

std::unique_ptr<column> repeat_join(strings_column_view const& input,
                                    size_type repeat_times,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  // todo handle err
  auto const num_rows = input.size();
  if (num_rows == 0) { return detail::make_empty_strings_column(stream, mr); }

  auto const strings_dv_ptr = column_device_view::create(input.parent(), stream);
  if (input.has_nulls()) {
    auto const fn = compute_size_and_repeat_fn<true>{*strings_dv_ptr, repeat_times};
    auto [offsets_column, chars_column, null_mask, null_count] =
      make_strings_children_with_null_mask(fn, num_rows, num_rows, stream, mr);
    return make_strings_column(num_rows,
                               std::move(offsets_column),
                               std::move(chars_column),
                               null_count,
                               std::move(null_mask),
                               stream,
                               mr);
  } else {
    auto const fn = compute_size_and_repeat_fn<false>{*strings_dv_ptr, repeat_times};
    auto [offsets_column, chars_column] = make_strings_children(fn, num_rows, stream, mr);
    return make_strings_column(num_rows,
                               std::move(offsets_column),
                               std::move(chars_column),
                               0,
                               rmm::device_buffer{},
                               stream,
                               mr);
  }
}

}  // namespace detail

string_scalar repeat_join(string_scalar const& input,
                          size_type repeat_times,
                          rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::repeat_join(input, repeat_times, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> repeat_join(strings_column_view const& input,
                                    size_type repeat_times,
                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::repeat_join(input, repeat_times, rmm::cuda_stream_default, mr);
}

}  // namespace strings
}  // namespace cudf
