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
#pragma once

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/lists/lists_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/transform_scan.h>

namespace cudf {
namespace lists {
namespace detail {

/**
 * @brief The information needed to create an iterator to gather level N+1
 *
 * @ref make_gather_data
 */
struct gather_data {
  // The offsets column from our parent list (level N)
  std::unique_ptr<column> offsets;
  // For each offset in the above offsets column, the original offset value
  // prior to being gathered.
  // Example:
  // If the offsets[3] == 6  (representing row 3 of the new column)
  // And the original value it was itself gathered from was 15, then
  // base_offsets[3] == 15
  rmm::device_uvector<int32_t> base_offsets;
  // size of the gather map that will be generated from this data
  size_type gather_map_size;
};

/**
 * @copydoc cudf::make_gather_data(cudf::lists_column_view const& source_column,
 *                                 MapItType gather_map,
 *                                 size_type gather_map_size,
 *                                 cudaStream_t stream,
 *                                 rmm::mr::device_memory_resource* mr)
 *
 * @param prev_base_offsets The buffer backing the base offsets used in the gather map. We can
 *                          free this buffer before allocating the new one to keep peak memory
 *                          usage down.
 */
template <bool NullifyOutOfBounds, typename MapItType>
gather_data make_gather_data(cudf::lists_column_view const& source_column,
                             MapItType gather_map,
                             size_type gather_map_size,
                             rmm::device_uvector<int32_t>&& prev_base_offsets,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr)
{
  // size of the gather map is the # of output rows
  size_type output_count = gather_map_size;
  size_type offset_count = output_count + 1;

  // offsets of the source column
  int32_t const* src_offsets{source_column.offsets().data<int32_t>() + source_column.offset()};
  size_type const src_size = source_column.size();

  // outgoing offsets.  these will persist as output from the entire gather operation
  auto dst_offsets_c = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, offset_count, mask_state::UNALLOCATED, stream, mr);
  mutable_column_view dst_offsets_v = dst_offsets_c->mutable_view();

  // generate the compacted outgoing offsets.
  auto count_iter = thrust::make_counting_iterator<int32_t>(0);
  thrust::transform_exclusive_scan(
    rmm::exec_policy(stream)->on(stream.value()),
    count_iter,
    count_iter + offset_count,
    dst_offsets_v.begin<int32_t>(),
    [gather_map, output_count, src_offsets, src_size] __device__(int32_t index) -> int32_t {
      int32_t offset_index = index < output_count ? gather_map[index] : 0;

      // if this is an invalid index, this will be a NULL list
      if (NullifyOutOfBounds && ((offset_index < 0) || (offset_index >= src_size))) { return 0; }

      // the length of this list
      return src_offsets[offset_index + 1] - src_offsets[offset_index];
    },
    0,
    thrust::plus<int32_t>());

  // handle sliced columns
  size_type const shift =
    source_column.offset() > 0
      ? cudf::detail::get_value<size_type>(source_column.offsets(), source_column.offset(), stream)
      : 0;

  // generate the base offsets
  rmm::device_uvector<int32_t> base_offsets = rmm::device_uvector<int32_t>(output_count, stream);
  thrust::transform(rmm::exec_policy(stream)->on(stream.value()),
                    gather_map,
                    gather_map + output_count,
                    base_offsets.data(),
                    [src_offsets, src_size, shift] __device__(int32_t index) {
                      // if this is an invalid index, this will be a NULL list
                      if (NullifyOutOfBounds && ((index < 0) || (index >= src_size))) { return 0; }
                      return src_offsets[index] - shift;
                    });

  // now that we are done using the gather_map, we can release the underlying prev_base_offsets.
  // doing this prevents this (potentially large) memory buffer from sitting around unused as the
  // recursion continues.
  prev_base_offsets.release();

  // Retrieve size of the resulting gather map for level N+1 (the last offset)
  size_type child_gather_map_size =
    cudf::detail::get_value<size_type>(dst_offsets_c->view(), output_count, stream);

  return {std::move(dst_offsets_c), std::move(base_offsets), child_gather_map_size};
}

/**
 * @brief Generates the data needed to create a `gather_map` for the next level of
 * recursion in a hierarchy of list columns.
 *
 * Gathering from a single level of a list column is similar to gathering from
 * a string column.  Each row represents a list bounded by offsets.
 *
 * @code{.pseudo}
 * Example:
 * Level 0 : List<List<int>>
 *           Size : 3
 *           Offsets : [0, 2, 5, 10]
 *
 * This represents a column with 3 rows.
 * Row 0 has 2 elements (bounded by offsets 0,2)
 * Row 1 has 3 elements (bounded by offsets 2,5)
 * Row 2 has 5 elements (bounded by offsets 5,10)
 *
 * If we wanted to gather rows 0 and 2 the offsets for our outgoing column
 * would be the compacted ranges (0,2) and (5,10). The level 1 column
 * then looks like
 *
 * Level 1 : List<int>
 *           Size : 2
 *           Offsets : [0, 2, 7]
 *
 * However, we need to then gather one level further, because at the bottom we have
 * a column of integers.  We cannot gather the elements in the ranges (0, 2) and (2, 7).
 * Instead, we have to gather elements in the ranges from the Level 0 column (0, 2) and (5, 10).
 * So our gather_map iterator will need to know these "base" offsets to index properly.
 * Specifically:
 *
 * Offsets        : [0, 2, 7]    The offsets for Level 1
 * Base Offsets   : [0, 5]       The corresponding base offsets from Level 0
 *
 * Using this we can create an iterator that generates the sequence which properly indexes the
 * final integer values we want to gather.
 *
 * [0, 1, 5, 6, 7, 8, 9]
 * @endcode
 *
 * Thinking generally, this means that to produce a gather_map for level N+1, we need to use the
 * offsets from level N and the "base" offsets from level N-1. So we are always carrying along
 * one extra buffer of these "base" offsets which keeps our memory usage well controlled.
 *
 * @code{.pseudo}
 * Example:
 *
 * "column of lists of lists of ints"
 * {
 *   {
 *      {2, 3}, {4, 5}
 *   },
 *   {
 *      {6, 7, 8}, {9, 10, 11}, {12, 13, 14}
 *   },
 *   {
 *      {15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}
 *   }
 * }
 *
 * List<List<int32_t>>:
 * Length : 3
 * Offsets : 0, 2, 5, 10
 * Children :
 *    List<int32_t>:
 *    Length : 10
 *    Offsets : 0, 2, 4, 7, 10, 13, 15, 17, 19, 21, 23
 *       Children :
 *           2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 17, 18, 17, 18, 17, 18
 *
 * Final column, doing gather([0, 2])
 *
 * {
 *   {
 *      {2, 3}, {4, 5}
 *   },
 *   {
 *      {15, 16}, {17, 18}, {17, 18}, {17, 18}, {17, 18}
 *   }
 * }
 *
 * List<List<int32_t>>:
 * Length : 2
 * Offsets : 0, 2, 7
 * Children :
 *    List<int32_t>:
 *    Length : 7
 *    Offsets : 0, 2, 4, 6, 8, 10, 12, 14
 *       Children :
 *          2, 3, 4, 5, 15, 16, 17, 18, 17, 18, 17, 18, 17, 18
 * @endcode
 *
 * @tparam MapItType Iterator type to access the incoming column.
 * @tparam NullifyOutOfBounds Nullify values in `gather_map` that are out of bounds
 * @param source_column View into the column to gather from
 * @param gather_map Iterator access to the gather map for `source_column` map
 * @param gather_map_size Size of the gather map.
 * @param stream CUDA stream on which to execute kernels
 * @param mr Memory resource to use for all allocations
 *
 * @returns The gather_data struct needed to construct the gather map for the
 *          next level of recursion.
 *
 */
template <bool NullifyOutOfBounds, typename MapItType>
gather_data make_gather_data(cudf::lists_column_view const& source_column,
                             MapItType gather_map,
                             size_type gather_map_size,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr)
{
  return make_gather_data<NullifyOutOfBounds, MapItType>(
    source_column,
    gather_map,
    gather_map_size,
    rmm::device_uvector<int32_t>{0, stream, mr},
    stream,
    mr);
}

/**
 * @brief Gather a list column from a hierarchy of list columns.
 *
 * The recursion continues from here at least 1 level further.
 *
 * @param list View into the list column to gather from
 * @param gd The gather_data needed to construct a gather map iterator for this level
 * @param stream CUDA stream on which to execute kernels
 * @param mr Memory resource to use for all allocations
 *
 * @returns column with elements gathered based on `gather_data`
 *
 */
std::unique_ptr<column> gather_list_nested(
  lists_column_view const& list,
  gather_data& gd,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Gather a leaf column from a hierarchy of list columns.
 *
 * The recursion terminates here.
 *
 * @param column View into the column to gather from
 * @param gd The gather_data needed to construct a gather map iterator for this level
 * @param stream CUDA stream on which to execute kernels
 * @param mr Memory resource to use for all allocations
 *
 * @returns column with elements gathered based on `gather_data`
 *
 */
std::unique_ptr<column> gather_list_leaf(
  column_view const& column,
  gather_data const& gd,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace detail
}  // namespace lists
}  // namespace cudf
