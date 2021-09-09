/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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
#include <cudf/copying.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/lists/extract.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/transform.h>

namespace cudf {
namespace lists {
namespace detail {

namespace {

/**
 * @brief Convert index value for each sublist into a gather index for
 * the lists column's child column.
 * This specialization is for when the index is a fixed size_type value.
 */
template <bool PositiveIndex = true, typename IndexType = size_type>
struct map_index_fn {
  column_device_view const d_offsets;  // offsets to each sublist (including validity mask)
  IndexType const index;               // index of element within each sublist
  size_type const out_of_bounds;       // value to use to indicate out-of-bounds

  __device__ int32_t operator()(size_type idx)
  {
    if (d_offsets.is_null(idx)) return out_of_bounds;
    auto const offset = d_offsets.element<int32_t>(idx);
    auto const length = d_offsets.element<int32_t>(idx + 1) - offset;
    if constexpr (PositiveIndex) { return index < length ? index + offset : out_of_bounds; }
    return index >= -length ? length + index + offset : out_of_bounds;
  }
};

/**
 * @brief Convert index value for each sublist into a gather index for
 * the lists column's child column.
 * This specialization works on a column_device_view of indices.
 */
template <bool ignored>
struct map_index_fn<ignored, column_view> {
  column_device_view const d_offsets;  // offsets to each sublist (including validity mask)
  column_device_view const d_indices;  // column of indices for each element within a sublist
  size_type const out_of_bounds;       // value to use to indicate out-of-bounds

  __device__ int32_t operator()(size_type idx)
  {
    if (d_offsets.is_null(idx) || d_indices.is_null(idx)) return out_of_bounds;
    auto const offset = d_offsets.element<int32_t>(idx);
    auto const length = d_offsets.element<int32_t>(idx + 1) - offset;
    auto const index  = d_indices.element<int32_t>(idx);

    if (index >= 0) {
      return index < length ? index + offset : out_of_bounds;
    } else {
      return index >= -length ? length + index + offset : out_of_bounds;
    }
  }
};

template <typename IndexType = size_type>
auto get_device_accessible_index(IndexType const& index, rmm::cuda_stream_view)
{
  return &index;  // size_type is accessible in __device__.
}

template <>
auto get_device_accessible_index<column_view>(column_view const& index_column,
                                              rmm::cuda_stream_view stream)
{
  return column_device_view::create(index_column, stream);
}

}  // namespace

template <bool PositiveIndex, typename IndexType = size_type>
std::unique_ptr<column> extract_list_element_impl(lists_column_view lists_column,
                                                  IndexType index,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::mr::device_memory_resource* mr)
{
  if (lists_column.is_empty()) return empty_like(lists_column.child());
  auto const offsets_column = lists_column.offsets();

  // create a column_view with attributes of the parent and data from the offsets
  column_view annotated_offsets(data_type{type_id::INT32},
                                lists_column.size() + 1,
                                offsets_column.data<int32_t>(),
                                lists_column.null_mask(),
                                lists_column.null_count(),
                                lists_column.offset());

  // create a gather map for extracting elements from the child column
  auto gather_map = make_fixed_width_column(
    data_type{type_id::INT32}, annotated_offsets.size() - 1, mask_state::UNALLOCATED, stream);
  auto d_gather_map       = gather_map->mutable_view().data<int32_t>();
  auto const child_column = lists_column.child();

  // build the gather map using the offsets and the provided index
  auto const d_column = column_device_view::create(annotated_offsets, stream);
  auto const d_index  = get_device_accessible_index(index, stream);
  thrust::transform(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<size_type>(0),
    thrust::make_counting_iterator<size_type>(gather_map->size()),
    d_gather_map,
    map_index_fn<PositiveIndex, IndexType>{*d_column, *d_index, child_column.size()});

  // call gather on the child column
  auto result = cudf::detail::gather(table_view({child_column}),
                                     d_gather_map,
                                     d_gather_map + gather_map->size(),
                                     out_of_bounds_policy::NULLIFY,  // nullify-out-of-bounds
                                     stream,
                                     mr)
                  ->release();
  if (result.front()->null_count() == 0)
    result.front()->set_null_mask(rmm::device_buffer{0, stream, mr}, 0);
  return std::unique_ptr<column>(std::move(result.front()));
}

/**
 * @copydoc cudf::lists::extract_list_element
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> extract_list_element(lists_column_view lists_column,
                                             size_type index,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  return index < 0 ? extract_list_element_impl<false>(lists_column, index, stream, mr)
                   : extract_list_element_impl<true>(lists_column, index, stream, mr);
}

std::unique_ptr<column> extract_list_element(lists_column_view lists_column,
                                             column_view const& indices,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(lists_column.size() == indices.size(),
               "Index column must have as many elements as lists column.");
  return extract_list_element_impl<false, column_view>(lists_column, indices, stream, mr);
}

}  // namespace detail

/**
 * @copydoc cudf::lists::extract_list_element
 */
std::unique_ptr<column> extract_list_element(lists_column_view const& lists_column,
                                             size_type index,
                                             rmm::mr::device_memory_resource* mr)
{
  return detail::extract_list_element(lists_column, index, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> extract_list_element(lists_column_view const& lists_column,
                                             column_view const& indices,
                                             rmm::mr::device_memory_resource* mr)
{
  return detail::extract_list_element(lists_column, indices, rmm::cuda_stream_default, mr);
}

}  // namespace lists
}  // namespace cudf
