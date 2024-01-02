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
#include <cudf/detail/valid_if.cuh>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/strings/detail/fill.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/iterator/counting_iterator.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {
struct fill_fn {
  column_device_view d_strings;
  size_type begin;
  size_type end;
  string_view d_value;
  size_type* d_offsets{};
  char* d_chars{};

  __device__ void operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) {
      if (!d_chars) d_offsets[idx] = 0;
      return;
    }
    auto const d_str =
      ((begin <= idx) && (idx < end)) ? d_value : d_strings.element<string_view>(idx);
    if (!d_chars) {
      d_offsets[idx] = d_str.size_bytes();
    } else {
      copy_string(d_chars + d_offsets[idx], d_str);
    }
  }
};
}  // namespace

std::unique_ptr<column> fill(strings_column_view const& strings,
                             size_type begin,
                             size_type end,
                             string_scalar const& value,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr)
{
  auto strings_count = strings.size();
  if (strings_count == 0) return make_empty_column(type_id::STRING);
  CUDF_EXPECTS((begin >= 0) && (end <= strings_count),
               "Parameters [begin,end) are outside the range of the provided strings column");
  CUDF_EXPECTS(begin <= end, "Parameters [begin,end) have invalid range values");
  if (begin == end)  // return a copy
    return std::make_unique<column>(strings.parent(), stream, mr);

  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;

  // create resulting null mask
  auto [null_mask, null_count] = [begin, end, &value, d_strings, stream, mr] {
    if (begin == 0 and end == d_strings.size() and value.is_valid(stream)) {
      return std::pair(rmm::device_buffer{}, 0);
    }
    auto d_value = get_scalar_device_view(const_cast<string_scalar&>(value));
    return cudf::detail::valid_if(
      thrust::make_counting_iterator<size_type>(0),
      thrust::make_counting_iterator<size_type>(d_strings.size()),
      [d_strings, begin, end, d_value] __device__(size_type idx) {
        return ((begin <= idx) && (idx < end)) ? d_value.is_valid() : d_strings.is_valid(idx);
      },
      stream,
      mr);
  }();

  auto d_value     = const_cast<string_scalar&>(value);
  auto const d_str = d_value.is_valid(stream) ? d_value.value(stream) : string_view{};
  auto fn          = fill_fn{d_strings, begin, end, d_str};
  auto [offsets_column, chars_column] = make_strings_children(fn, strings_count, stream, mr);

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             std::move(chars_column),
                             null_count,
                             std::move(null_mask));
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
