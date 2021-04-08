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

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/groupby/group_shift.hpp>
#include <cudf/detail/groupby/sort_helper.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/scatter.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/transform_iterator.h>

namespace cudf {
namespace groupby {
namespace detail {

namespace {

constexpr size_type SAFE_GATHER_IDX = 0;

template <bool ForwardShift, typename EdgeIterator>
struct group_shift_fill_functor {
  EdgeIterator group_edges_begin;
  size_type offset;
  size_type group_label, offset_to_edge;

  group_shift_fill_functor(EdgeIterator group_edges_begin, size_type offset)
    : group_edges_begin(group_edges_begin), offset(offset)
  {
  }

  __device__ size_type operator()(size_type i)
  {
    if (ForwardShift) {   // offset > 0
      group_label    = i / offset;
      offset_to_edge = i % offset;
    } else {              // offset < 0
      group_label    = -i / offset;
      offset_to_edge = i % offset + offset + 1;
    }
    return *(group_edges_begin + group_label) + offset_to_edge;
  }
};

}  // namespace

template <bool ForwardShift, typename EdgeIterator>
std::unique_ptr<column> group_shift_impl(column_view const& values,
                                         size_type offset,
                                         EdgeIterator group_edges_begin,
                                         std::size_t num_groups,
                                         cudf::scalar const& fill_value,
                                         rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource* mr)
{
  auto shift_func = [col_size = values.size(), offset] __device__(size_type idx) {
    auto raw_shifted_idx = idx - offset;
    return static_cast<uint32_t>(
      raw_shifted_idx >= 0 and raw_shifted_idx < col_size ? raw_shifted_idx : SAFE_GATHER_IDX);
  };
  auto gather_iter_begin = cudf::detail::make_counting_transform_iterator(0, shift_func);

  auto shifted = cudf::detail::gather(table_view({values}),
                                      gather_iter_begin,
                                      gather_iter_begin + values.size(),
                                      out_of_bounds_policy::DONT_CHECK,
                                      stream,
                                      mr);

  auto scatter_map = make_numeric_column(
    data_type(type_id::UINT32), num_groups * std::abs(offset), mask_state::UNALLOCATED);
  group_shift_fill_functor<ForwardShift, decltype(group_edges_begin)> fill_func{group_edges_begin,
                                                                                offset};
  auto scatter_map_iterator = cudf::detail::make_counting_transform_iterator(0, fill_func);
  thrust::copy(rmm::exec_policy(stream),
               scatter_map_iterator,
               scatter_map_iterator + scatter_map->view().size(),
               scatter_map->mutable_view().begin<size_type>());
               
  auto shifted_filled =
    cudf::detail::scatter({fill_value}, scatter_map->view(), shifted->view(), true, stream, mr);

  return std::move(shifted_filled->release()[0]);
}

std::unique_ptr<column> group_shift(column_view const& values,
                                    size_type offset,
                                    scalar const& fill_value,
                                    sort::sort_groupby_helper& helper,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  if (values.size() == 0) { return make_empty_column(values.type()); }

  rmm::device_uvector<size_type> const& group_offsets = helper.group_offsets(stream);
  if (offset > 0) {
    return group_shift_impl<true>(
      values, offset, group_offsets.begin(), group_offsets.size() - 1, fill_value, stream, mr);
  } else {
    auto redge_iter = thrust::make_transform_iterator(group_offsets.begin() + 1,
                                                      [] __device__(auto i) { return i - 1; });
    return group_shift_impl<false>(
      values, offset, redge_iter, group_offsets.size() - 1, fill_value, stream, mr);
  }
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf
