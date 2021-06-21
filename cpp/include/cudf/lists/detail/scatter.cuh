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

#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/lists/detail/scatter_helper.cuh>
#include <cudf/lists/list_device_view.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/types.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cinttypes>

namespace cudf {
namespace lists {
namespace detail {

template <typename IndexIterator>
rmm::device_uvector<unbound_list_view> list_vector_from_column(
  unbound_list_view::label_type label,
  cudf::detail::lists_column_device_view const& lists_column,
  IndexIterator index_begin,
  IndexIterator index_end,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  auto n_rows = thrust::distance(index_begin, index_end);

  auto vector = rmm::device_uvector<unbound_list_view>(n_rows, stream, mr);

  thrust::transform(rmm::exec_policy(stream),
                    index_begin,
                    index_end,
                    vector.begin(),
                    [label, lists_column] __device__(size_type row_index) {
                      return unbound_list_view{label, lists_column, row_index};
                    });

  return vector;
}

/**
 * @brief General implementation of scattering into list column
 *
 * Scattering `source` into `target` according to `scatter_map`.
 * The view order of `source` and `target` can be specified by
 * `source_vector` and `target_vector` respectively.
 *
 * @tparam MapIterator must produce index values within the target column.
 *
 * @param source_vector A vector of `unbound_list_view` into source column
 * @param target_vector A vector of `unbound_list_view` into target column
 * @param scatter_map_begin Start iterator of scatter map
 * @param scatter_map_end End iterator of scatter map
 * @param source Source column view
 * @param target Target column view
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New lists column.
 */
template <typename MapIterator>
std::unique_ptr<column> scatter_impl(
  rmm::device_uvector<unbound_list_view> const& source_vector,
  rmm::device_uvector<unbound_list_view>& target_vector,
  MapIterator scatter_map_begin,
  MapIterator scatter_map_end,
  column_view const& source,
  column_view const& target,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  assert_same_data_type(source, target);

  auto const child_column_type = lists_column_view(target).child().type();

  // Scatter.
  thrust::scatter(rmm::exec_policy(stream),
                  source_vector.begin(),
                  source_vector.end(),
                  scatter_map_begin,
                  target_vector.begin());

  auto const source_lists_column_view =
    lists_column_view(source);  // Checks that this is a list column.
  auto const target_lists_column_view =
    lists_column_view(target);  // Checks that target is a list column.

  auto list_size_begin = thrust::make_transform_iterator(
    target_vector.begin(), [] __device__(unbound_list_view l) { return l.size(); });
  auto offsets_column = cudf::strings::detail::make_offsets_child_column(
    list_size_begin, list_size_begin + target.size(), stream, mr);

  auto child_column = build_lists_child_column_recursive(child_column_type,
                                                         target_vector,
                                                         offsets_column->view(),
                                                         source_lists_column_view,
                                                         target_lists_column_view,
                                                         stream,
                                                         mr);

  auto null_mask =
    target.has_nulls() ? copy_bitmask(target, stream, mr) : rmm::device_buffer{0, stream, mr};

  return cudf::make_lists_column(target.size(),
                                 std::move(offsets_column),
                                 std::move(child_column),
                                 cudf::UNKNOWN_NULL_COUNT,
                                 std::move(null_mask),
                                 stream,
                                 mr);
}

/**
 * @brief Scatters lists into a copy of the target column
 * according to a scatter map.
 *
 * The scatter is performed according to the scatter iterator such that row
 * `scatter_map[i]` of the output column is replaced by the source list-row.
 * All other rows of the output column equal corresponding rows of the target table.
 *
 * If the same index appears more than once in the scatter map, the result is
 * undefined.
 *
 * The caller must update the null mask in the output column.
 *
 * @tparam MapIterator must produce index values within the target column.
 *
 * @param source Source column view
 * @param scatter_map_begin Start iterator of scatter map
 * @param scatter_map_end End iterator of scatter map
 * @param target Target column view
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New lists column.
 */
template <typename MapIterator>
std::unique_ptr<column> scatter(
  column_view const& source,
  MapIterator scatter_map_begin,
  MapIterator scatter_map_end,
  column_view const& target,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto const num_rows = target.size();
  if (num_rows == 0) { return cudf::empty_like(target); }

  auto const source_device_view = column_device_view::create(source, stream);
  auto const scatter_map_size   = thrust::distance(scatter_map_begin, scatter_map_end);
  auto const source_vector =
    list_vector_from_column(unbound_list_view::label_type::SOURCE,
                            cudf::detail::lists_column_device_view(*source_device_view),
                            thrust::make_counting_iterator<size_type>(0),
                            thrust::make_counting_iterator<size_type>(scatter_map_size),
                            stream,
                            mr);

  auto const target_device_view = column_device_view::create(target, stream);
  auto target_vector =
    list_vector_from_column(unbound_list_view::label_type::TARGET,
                            cudf::detail::lists_column_device_view(*target_device_view),
                            thrust::make_counting_iterator<size_type>(0),
                            thrust::make_counting_iterator<size_type>(num_rows),
                            stream,
                            mr);

  return scatter_impl(
    source_vector, target_vector, scatter_map_begin, scatter_map_end, source, target, stream, mr);
}

/**
 * @brief Scatters list scalar (a single row) into a copy of the target column
 * according to a scatter map.
 *
 * Returns a copy of the target column where every row specified in the `scatter_map`
 * is replaced by the row value.
 *
 * If the same index appears more than once in the scatter map, the result is
 * undefined.
 *
 * The caller must update the null mask in the output column.
 *
 * @tparam MapIterator must produce index values within the target column.
 *
 * @param slr Source scalar, specifying row data
 * @param scatter_map_begin Start iterator of scatter map
 * @param scatter_map_end End iterator of scatter map
 * @param target Target column view
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New lists column.
 */
template <typename MapIterator>
std::unique_ptr<column> scatter(
  scalar const& slr,
  MapIterator scatter_map_begin,
  MapIterator scatter_map_end,
  column_view const& target,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto const num_rows = target.size();
  if (num_rows == 0) { return cudf::empty_like(target); }

  auto lv        = static_cast<list_scalar const*>(&slr);
  bool slr_valid = slr.is_valid(stream);
  rmm::device_buffer null_mask =
    slr_valid ? cudf::detail::create_null_mask(1, mask_state::UNALLOCATED, stream, mr)
              : cudf::detail::create_null_mask(1, mask_state::ALL_NULL, stream, mr);
  auto offset_column = make_numeric_column(
    data_type{type_to_id<offset_type>()}, 2, mask_state::UNALLOCATED, stream, mr);
  thrust::sequence(rmm::exec_policy(stream),
                   offset_column->mutable_view().begin<offset_type>(),
                   offset_column->mutable_view().end<offset_type>(),
                   0,
                   lv->view().size());
  auto wrapped = column_view(data_type{type_id::LIST},
                             1,
                             nullptr,
                             static_cast<bitmask_type const*>(null_mask.data()),
                             slr_valid ? 0 : 1,
                             0,
                             {offset_column->view(), lv->view()});

  auto const source_device_view = column_device_view::create(wrapped, stream);
  auto const scatter_map_size   = thrust::distance(scatter_map_begin, scatter_map_end);
  auto const source_vector =
    list_vector_from_column(unbound_list_view::label_type::SOURCE,
                            cudf::detail::lists_column_device_view(*source_device_view),
                            thrust::make_constant_iterator<size_type>(0),
                            thrust::make_constant_iterator<size_type>(0) + scatter_map_size,
                            stream,
                            mr);

  auto const target_device_view = column_device_view::create(target, stream);
  auto target_vector =
    list_vector_from_column(unbound_list_view::label_type::TARGET,
                            cudf::detail::lists_column_device_view(*target_device_view),
                            thrust::make_counting_iterator<size_type>(0),
                            thrust::make_counting_iterator<size_type>(num_rows),
                            stream,
                            mr);

  return scatter_impl(
    source_vector, target_vector, scatter_map_begin, scatter_map_end, wrapped, target, stream, mr);
}

}  // namespace detail
}  // namespace lists
}  // namespace cudf
