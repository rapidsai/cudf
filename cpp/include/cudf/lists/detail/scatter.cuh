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

namespace {

rmm::device_uvector<unbound_list_view> list_vector_from_column(
  unbound_list_view::label_type label,
  cudf::detail::lists_column_device_view const& lists_column,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  auto n_rows = lists_column.size();

  auto vector = rmm::device_uvector<unbound_list_view>(n_rows, stream, mr);

  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(n_rows),
                    vector.begin(),
                    [label, lists_column] __device__(size_type row_index) {
                      return unbound_list_view{label, lists_column, row_index};
                    });

  return vector;
}

#ifndef NDEBUG
void print(std::string const& msg, column_view const& col, rmm::cuda_stream_view stream)
{
  if (col.type().id() != type_id::INT32) {
    std::cout << "[Cannot print non-INT32 column.]" << std::endl;
    return;
  }

  std::cout << msg << " = [";
  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<size_type>(0),
    col.size(),
    [c = col.template data<int32_t>()] __device__(auto const& i) { printf("%d,", c[i]); });
  std::cout << "]" << std::endl;
}

void print(std::string const& msg,
           rmm::device_uvector<unbound_list_view> const& scatter,
           rmm::cuda_stream_view stream)
{
  std::cout << msg << " == [";

  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<size_type>(0),
                     scatter.size(),
                     [s = scatter.begin()] __device__(auto const& i) {
                       auto si = s[i];
                       printf("%s[%d](%d), ",
                              (si.label() == unbound_list_view::label_type::SOURCE ? "S" : "T"),
                              si.row_index(),
                              si.size());
                     });
  std::cout << "]" << std::endl;
}
#endif  // NDEBUG

/**
 * @brief Checks that the specified columns have matching schemas, all the way down.
 */
void assert_same_data_type(column_view const& lhs, column_view const& rhs)
{
  CUDF_EXPECTS(lhs.type().id() == rhs.type().id(), "Mismatched Data types.");
  CUDF_EXPECTS(lhs.num_children() == rhs.num_children(), "Mismatched number of child columns.");

  for (int i{0}; i < lhs.num_children(); ++i) { assert_same_data_type(lhs.child(i), rhs.child(i)); }
}

}  // namespace

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
 * @tparam SourceIterator must produce list_view objects
 * @tparam MapIterator must produce index values within the target column.
 *
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
  rmm::cuda_stream_view stream        = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto const num_rows = target.size();

  if (num_rows == 0) { return cudf::empty_like(target); }

  auto const child_column_type = lists_column_view(target).child().type();

  assert_same_data_type(source, target);

  using lists_column_device_view = cudf::detail::lists_column_device_view;
  using unbound_list_view        = cudf::lists::detail::unbound_list_view;

  auto const source_device_view = column_device_view::create(source, stream);
  auto const source_vector      = list_vector_from_column(unbound_list_view::label_type::SOURCE,
                                                     lists_column_device_view(*source_device_view),
                                                     stream,
                                                     mr);

  auto const target_device_view = column_device_view::create(target, stream);
  auto target_vector            = list_vector_from_column(unbound_list_view::label_type::TARGET,
                                               lists_column_device_view(*target_device_view),
                                               stream,
                                               mr);

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

  auto child_column = build_child_column(child_column_type,
                                         target_vector,
                                         offsets_column->view(),
                                         source_lists_column_view,
                                         target_lists_column_view,
                                         stream,
                                         mr);

  auto null_mask =
    target.has_nulls() ? copy_bitmask(target, stream, mr) : rmm::device_buffer{0, stream, mr};

  return cudf::make_lists_column(num_rows,
                                 std::move(offsets_column),
                                 std::move(child_column),
                                 cudf::UNKNOWN_NULL_COUNT,
                                 std::move(null_mask),
                                 stream,
                                 mr);
}

}  // namespace detail
}  // namespace lists
}  // namespace cudf
