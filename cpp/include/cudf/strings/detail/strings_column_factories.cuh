/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/utilities/error.hpp>
#include <strings/utilities.cuh>

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/for_each.h>
#include <thrust/transform_reduce.h>

// clang-format off
namespace cudf {
namespace strings {
namespace detail {

// Create a strings-type column from vector of pointer/size pairs
template<typename IndexPairIterator>
std::unique_ptr<column> make_strings_column(
  IndexPairIterator begin, IndexPairIterator end,
  rmm::mr::device_memory_resource* mr,
  cudaStream_t stream ) 
{
  CUDF_FUNC_RANGE();
  size_type strings_count = thrust::distance(begin,end);
  if (strings_count == 0) return strings::detail::make_empty_strings_column(mr, stream);

  using string_index_pair = thrust::pair<const char*, size_type>;

  auto execpol   = rmm::exec_policy(stream);
  // check total size is not too large for cudf column
  size_t bytes = thrust::transform_reduce(
    execpol->on(stream), begin, end,
    [] __device__(string_index_pair const& item) {
      return (item.first != nullptr) ? item.second : 0;
    },
    0,
    thrust::plus<size_t>());
  CUDF_EXPECTS(bytes < std::numeric_limits<size_type>::max(),
               "total size of strings is too large for cudf column");

  // build offsets column from the strings sizes
  auto offsets_transformer = [begin] __device__(size_type idx) {
    string_index_pair const item = begin[idx];
    return (item.first != nullptr ? static_cast<int32_t>(item.second) : 0);
  };
  auto offsets_transformer_itr = thrust::make_transform_iterator(
    thrust::make_counting_iterator<size_type>(0), offsets_transformer);
  auto offsets_column = strings::detail::make_offsets_child_column(
    offsets_transformer_itr, offsets_transformer_itr + strings_count, mr, stream);
  auto d_offsets = offsets_column->view().template data<int32_t>();

  // create null mask
  auto new_nulls = cudf::detail::valid_if( begin, end,
    [] __device__(string_index_pair const item) { return item.first != nullptr; },
    stream,
    mr);
  auto null_count = new_nulls.second;
  rmm::device_buffer null_mask{0, stream, mr};
  if (null_count > 0) null_mask = std::move(new_nulls.first);

  // build chars column
  auto chars_column =
    strings::detail::create_chars_child_column(strings_count, null_count, bytes, mr, stream);
  auto d_chars = chars_column->mutable_view().template data<char>();
  thrust::for_each_n(execpol->on(stream),
                     thrust::make_counting_iterator<size_type>(0),
                     strings_count,
                     [begin, d_offsets, d_chars] __device__(size_type idx) {
                       string_index_pair const item = begin[idx];
                       if (item.first != nullptr)
                         memcpy(d_chars + d_offsets[idx], item.first, item.second);
                     });

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             std::move(chars_column),
                             null_count,
                             std::move(null_mask),
                             stream,
                             mr);
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
// clang-format on TODO fix
