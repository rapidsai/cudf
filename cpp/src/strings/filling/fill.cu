/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
#include <cudf/detail/valid_if.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/strings/detail/fill.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

#include <cuda/functional>

namespace cudf {
namespace strings {
namespace detail {
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

  // string_scalar.data() is null for valid, empty strings
  auto d_value = get_scalar_device_view(const_cast<string_scalar&>(value));

  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;

  // create resulting null mask
  auto valid_mask = [begin, end, d_value, &value, d_strings, stream, mr] {
    if (begin == 0 and end == d_strings.size() and value.is_valid(stream))
      return std::pair(rmm::device_buffer{}, 0);
    return cudf::detail::valid_if(
      thrust::make_counting_iterator<size_type>(0),
      thrust::make_counting_iterator<size_type>(d_strings.size()),
      [d_strings, begin, end, d_value] __device__(size_type idx) {
        return ((begin <= idx) && (idx < end)) ? d_value.is_valid() : !d_strings.is_null(idx);
      },
      stream,
      mr);
  }();
  auto null_count               = valid_mask.second;
  rmm::device_buffer& null_mask = valid_mask.first;

  // build offsets column
  auto offsets_transformer = cuda::proclaim_return_type<size_type>(
    [d_strings, begin, end, d_value] __device__(size_type idx) {
      if (((begin <= idx) && (idx < end)) ? !d_value.is_valid() : d_strings.is_null(idx)) return 0;
      return ((begin <= idx) && (idx < end)) ? d_value.size()
                                             : d_strings.element<string_view>(idx).size_bytes();
    });
  auto offsets_transformer_itr = thrust::make_transform_iterator(
    thrust::make_counting_iterator<size_type>(0), offsets_transformer);
  auto [offsets_column, bytes] = cudf::detail::make_offsets_child_column(
    offsets_transformer_itr, offsets_transformer_itr + strings_count, stream, mr);
  auto d_offsets = offsets_column->view().data<int32_t>();

  // create the chars column
  auto chars_column = create_chars_child_column(bytes, stream, mr);
  // fill the chars column
  auto d_chars = chars_column->mutable_view().data<char>();
  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<size_type>(0),
    strings_count,
    [d_strings, begin, end, d_value, d_offsets, d_chars] __device__(size_type idx) {
      if (((begin <= idx) && (idx < end)) ? !d_value.is_valid() : d_strings.is_null(idx)) return;
      string_view const d_str =
        ((begin <= idx) && (idx < end)) ? d_value.value() : d_strings.element<string_view>(idx);
      memcpy(d_chars + d_offsets[idx], d_str.data(), d_str.size_bytes());
    });

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             std::move(chars_column),
                             null_count,
                             std::move(null_mask));
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
