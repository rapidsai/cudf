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

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/strings/detail/merge.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/tuple.h>

namespace cudf {
namespace strings {
namespace detail {
std::unique_ptr<column> merge(strings_column_view const& lhs,
                              strings_column_view const& rhs,
                              cudf::detail::index_vector const& row_order,
                              rmm::cuda_stream_view stream,
                              rmm::mr::device_memory_resource* mr)
{
  using cudf::detail::side;
  if (row_order.is_empty()) { return make_empty_column(type_id::STRING); }
  auto const strings_count = static_cast<cudf::size_type>(row_order.size());

  auto const lhs_column = column_device_view::create(lhs.parent(), stream);
  auto const d_lhs      = *lhs_column;
  auto const rhs_column = column_device_view::create(rhs.parent(), stream);
  auto const d_rhs      = *rhs_column;

  // caller will set the null mask
  auto const null_count = lhs.null_count() + rhs.null_count();
  auto null_mask        = (null_count > 0) ? cudf::detail::create_null_mask(
                                        strings_count, mask_state::ALL_VALID, stream, mr)
                                           : rmm::device_buffer{};

  // build offsets column
  auto offsets_transformer =
    cuda::proclaim_return_type<size_type>([d_lhs, d_rhs] __device__(auto index_pair) {
      auto const [side, index] = index_pair;
      if (side == side::LEFT ? d_lhs.is_null(index) : d_rhs.is_null(index)) { return 0; }
      auto d_str =
        side == side::LEFT ? d_lhs.element<string_view>(index) : d_rhs.element<string_view>(index);
      return d_str.size_bytes();
    });
  auto const begin             = row_order.begin();
  auto offsets_transformer_itr = thrust::make_transform_iterator(begin, offsets_transformer);
  auto [offsets_column, bytes] = cudf::strings::detail::make_offsets_child_column(
    offsets_transformer_itr, offsets_transformer_itr + strings_count, stream, mr);
  auto d_offsets = cudf::detail::offsetalator_factory::make_input_iterator(offsets_column->view());

  // create the chars column
  rmm::device_uvector<char> chars(bytes, stream, mr);
  auto d_chars = chars.data();
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::counting_iterator<size_type>(0),
                     strings_count,
                     [d_lhs, d_rhs, begin, d_offsets, d_chars] __device__(size_type idx) {
                       auto const [side, index] = begin[idx];
                       if (side == side::LEFT ? d_lhs.is_null(index) : d_rhs.is_null(index)) return;
                       auto d_str = side == side::LEFT ? d_lhs.element<string_view>(index)
                                                       : d_rhs.element<string_view>(index);
                       memcpy(d_chars + d_offsets[idx], d_str.data(), d_str.size_bytes());
                     });

  return make_strings_column(
    strings_count, std::move(offsets_column), chars.release(), null_count, std::move(null_mask));
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
