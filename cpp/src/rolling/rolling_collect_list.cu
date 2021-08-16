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

#include <rolling/rolling_collect_list.cuh>

#include <cudf/detail/get_value.cuh>

#include <rmm/device_uvector.hpp>

#include <thrust/copy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scatter.h>

namespace cudf {
namespace detail {

/**
 * @see cudf::detail::get_list_child_to_list_row_mapping
 */
std::unique_ptr<column> get_list_child_to_list_row_mapping(cudf::column_view const& offsets,
                                                           rmm::cuda_stream_view stream)
{
  // First, reduce offsets column by key, to identify the number of times
  // an offset appears.
  // Next, scatter the count for each offset (except the first and last),
  // into a column of N `0`s, where N == number of child rows.
  // For the example above:
  //   offsets        == [0, 2, 5, 8, 11, 13]
  //   scatter result == [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
  //
  // If the above example had an empty list row at index 2,
  // the same columns would look as follows:
  //   offsets        == [0, 2, 5, 5, 8, 11, 13]
  //   scatter result == [0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 1, 0]
  //
  // Note: To correctly handle null list rows at the beginning of
  // the output column, care must be taken to skip the first `0`
  // in the offsets column, when running `reduce_by_key()`.
  // This accounts for the `0` added by default to the offsets
  // column, marking the beginning of the column.

  auto const num_child_rows{
    cudf::detail::get_value<size_type>(offsets, offsets.size() - 1, stream)};

  rmm::device_uvector<size_type> scatter_values(offsets.size(), stream);
  rmm::device_uvector<size_type> scatter_keys(offsets.size(), stream);
  auto reduced_by_key =
    thrust::reduce_by_key(rmm::exec_policy(stream),
                          offsets.begin<size_type>() + 1,  // Skip first 0 in offsets.
                          offsets.end<size_type>(),
                          thrust::make_constant_iterator<size_type>(1),
                          scatter_keys.begin(),
                          scatter_values.begin());
  auto scatter_values_end = reduced_by_key.second;
  rmm::device_uvector<size_type> scatter_output(num_child_rows + 1, stream);
  thrust::fill_n(rmm::exec_policy(stream), scatter_output.begin(), num_child_rows, 0);
  thrust::scatter(rmm::exec_policy(stream),
                  scatter_values.begin(),
                  scatter_values_end,
                  scatter_keys.begin(),
                  scatter_output.begin());  // [0,0,1,0,0,1,...]

  // Next, generate mapping with inclusive_scan() on scatter() result.
  // For the example above:
  //   scatter result == [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
  //   inclusive_scan == [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4]
  //
  // For the case with an empty list at index 3:
  //   scatter result == [0, 0, 1, 0, 0, 2, 0, 0, 1, 0, 0, 1, 0]
  //   inclusive_scan == [0, 0, 1, 1, 1, 3, 3, 3, 4, 4, 4, 5, 5]
  auto per_row_mapping = make_fixed_width_column(
    data_type{type_to_id<size_type>()}, num_child_rows, mask_state::UNALLOCATED, stream);
  thrust::inclusive_scan(rmm::exec_policy(stream),
                         scatter_output.begin(),
                         scatter_output.begin() + num_child_rows,
                         per_row_mapping->mutable_view().template begin<size_type>());
  return per_row_mapping;
}

/**
 * @see cudf::detail::count_child_nulls
 */
size_type count_child_nulls(column_view const& input,
                            std::unique_ptr<column> const& gather_map,
                            rmm::cuda_stream_view stream)
{
  auto input_device_view = column_device_view::create(input, stream);

  auto input_row_is_null = [d_input = *input_device_view] __device__(auto i) {
    return d_input.is_null_nocheck(i);
  };

  return thrust::count_if(rmm::exec_policy(stream),
                          gather_map->view().begin<size_type>(),
                          gather_map->view().end<size_type>(),
                          input_row_is_null);
}

/**
 * @see cudf::detail::rolling_collect_list
 */
std::pair<std::unique_ptr<column>, std::unique_ptr<column>> purge_null_entries(
  column_view const& input,
  column_view const& gather_map,
  column_view const& offsets,
  size_type num_child_nulls,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  auto input_device_view = column_device_view::create(input, stream);

  auto input_row_not_null = [d_input = *input_device_view] __device__(auto i) {
    return d_input.is_valid_nocheck(i);
  };

  // Purge entries in gather_map that correspond to null input.
  auto new_gather_map = make_fixed_width_column(data_type{type_to_id<size_type>()},
                                                gather_map.size() - num_child_nulls,
                                                mask_state::UNALLOCATED,
                                                stream,
                                                mr);
  thrust::copy_if(rmm::exec_policy(stream),
                  gather_map.template begin<size_type>(),
                  gather_map.template end<size_type>(),
                  new_gather_map->mutable_view().template begin<size_type>(),
                  input_row_not_null);

  // Recalculate offsets after null entries are purged.
  auto new_sizes = make_fixed_width_column(
    data_type{type_to_id<size_type>()}, input.size(), mask_state::UNALLOCATED, stream, mr);

  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(input.size()),
                    new_sizes->mutable_view().template begin<size_type>(),
                    [d_gather_map  = gather_map.template begin<size_type>(),
                     d_old_offsets = offsets.template begin<size_type>(),
                     input_row_not_null] __device__(auto i) {
                      return thrust::count_if(thrust::seq,
                                              d_gather_map + d_old_offsets[i],
                                              d_gather_map + d_old_offsets[i + 1],
                                              input_row_not_null);
                    });

  auto new_offsets =
    strings::detail::make_offsets_child_column(new_sizes->view().template begin<size_type>(),
                                               new_sizes->view().template end<size_type>(),
                                               stream,
                                               mr);

  return std::make_pair<std::unique_ptr<column>, std::unique_ptr<column>>(std::move(new_gather_map),
                                                                          std::move(new_offsets));
}

}  // namespace detail
}  // namespace cudf
