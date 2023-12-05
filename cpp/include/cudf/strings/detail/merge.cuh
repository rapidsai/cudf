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
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/merge.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/tuple.h>

#include <cuda/functional>

namespace cudf {
namespace strings {
namespace detail {
/**
 * @brief Merges two strings columns.
 *
 * Caller must set the validity mask in the output column.
 *
 * @tparam row_order_iterator This must be an iterator for type thrust::tuple<side,size_type>.
 *
 * @param lhs First column.
 * @param rhs Second column.
 * @param row_order Indexes for each column.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings column.
 */
template <typename index_type, typename row_order_iterator>
std::unique_ptr<column> merge(strings_column_view const& lhs,
                              strings_column_view const& rhs,
                              row_order_iterator begin,
                              row_order_iterator end,
                              rmm::cuda_stream_view stream,
                              rmm::mr::device_memory_resource* mr)
{
  using cudf::detail::side;
  size_type strings_count = static_cast<size_type>(std::distance(begin, end));
  if (strings_count == 0) return make_empty_column(type_id::STRING);

  auto lhs_column = column_device_view::create(lhs.parent(), stream);
  auto d_lhs      = *lhs_column;
  auto rhs_column = column_device_view::create(rhs.parent(), stream);
  auto d_rhs      = *rhs_column;

  // caller will set the null mask
  rmm::device_buffer null_mask{0, stream, mr};
  size_type null_count = lhs.null_count() + rhs.null_count();
  if (null_count > 0)
    null_mask = cudf::detail::create_null_mask(strings_count, mask_state::ALL_VALID, stream, mr);

  // build offsets column
  auto offsets_transformer =
    cuda::proclaim_return_type<size_type>([d_lhs, d_rhs] __device__(auto index_pair) {
      auto const [side, index] = index_pair;
      if (side == side::LEFT ? d_lhs.is_null(index) : d_rhs.is_null(index)) return 0;
      auto d_str =
        side == side::LEFT ? d_lhs.element<string_view>(index) : d_rhs.element<string_view>(index);
      return d_str.size_bytes();
    });
  auto offsets_transformer_itr = thrust::make_transform_iterator(begin, offsets_transformer);
  auto [offsets_column, bytes] = cudf::detail::make_offsets_child_column(
    offsets_transformer_itr, offsets_transformer_itr + strings_count, stream, mr);
  auto d_offsets = offsets_column->view().template data<int32_t>();

  // create the chars column
  auto chars_column = strings::detail::create_chars_child_column(bytes, stream, mr);
  // merge the strings
  auto d_chars = chars_column->mutable_view().template data<char>();
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<size_type>(0),
                     strings_count,
                     [d_lhs, d_rhs, begin, d_offsets, d_chars] __device__(size_type idx) {
                       auto const [side, index] = begin[idx];
                       if (side == side::LEFT ? d_lhs.is_null(index) : d_rhs.is_null(index)) return;
                       auto d_str = side == side::LEFT ? d_lhs.element<string_view>(index)
                                                       : d_rhs.element<string_view>(index);
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
