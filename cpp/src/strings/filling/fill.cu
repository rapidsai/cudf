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
#include <cudf/strings/detail/fill.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/iterator/counting_iterator.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {
struct fill_fn {
  column_device_view const d_strings;
  size_type const begin;
  size_type const end;
  string_view const d_value;
  size_type* d_offsets{};
  char* d_chars{};

  __device__ string_view resolve_string_at(size_type idx) const
  {
    if ((begin <= idx) && (idx < end)) { return d_value; }
    return d_strings.is_valid(idx) ? d_strings.element<string_view>(idx) : string_view{};
  }

  __device__ void operator()(size_type idx) const
  {
    auto const d_str = resolve_string_at(idx);
    if (!d_chars) {
      d_offsets[idx] = d_str.size_bytes();
    } else {
      copy_string(d_chars + d_offsets[idx], d_str);
    }
  }
};
}  // namespace

std::unique_ptr<column> fill(strings_column_view const& input,
                             size_type begin,
                             size_type end,
                             string_scalar const& value,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  auto const strings_count = input.size();
  if (strings_count == 0) { return make_empty_column(type_id::STRING); }
  CUDF_EXPECTS((begin >= 0) && (end <= strings_count),
               "Parameters [begin,end) are outside the range of the provided strings column");
  CUDF_EXPECTS(begin <= end, "Parameters [begin,end) have invalid range values");
  if (begin == end) { return std::make_unique<column>(input.parent(), stream, mr); }

  auto strings_column  = column_device_view::create(input.parent(), stream);
  auto const d_strings = *strings_column;
  auto const is_valid  = value.is_valid(stream);

  // create resulting null mask
  auto [null_mask, null_count] = [begin, end, is_valid, d_strings, stream, mr] {
    if (begin == 0 and end == d_strings.size() and is_valid) {
      return std::pair(rmm::device_buffer{}, 0);
    }
    return cudf::detail::valid_if(
      thrust::make_counting_iterator<size_type>(0),
      thrust::make_counting_iterator<size_type>(d_strings.size()),
      [d_strings, begin, end, is_valid] __device__(size_type idx) {
        return ((begin <= idx) && (idx < end)) ? is_valid : d_strings.is_valid(idx);
      },
      stream,
      mr);
  }();

  auto const d_value = const_cast<string_scalar&>(value);
  auto const d_str   = is_valid ? d_value.value(stream) : string_view{};
  auto fn            = fill_fn{d_strings, begin, end, d_str};

  auto [offsets_column, chars] = make_strings_children(fn, strings_count, stream, mr);

  return make_strings_column(
    strings_count, std::move(offsets_column), chars.release(), null_count, std::move(null_mask));
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
