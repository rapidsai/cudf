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

#include <algorithm>
#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <memory>

namespace cudf {
namespace lists {
namespace detail {

namespace {

/**
 * @brief Merges the offsets child columns of multiple list columns into one.
 *
 * Since offsets are all relative to the start of their respective column, 
 * all offsets are shifted to account for the new starting position
 *
 * @param columns               Vector of lists columns to concatenate
 * @param total_list_count      Total number of lists contains in the columns
 * @param print_all_differences If true display all differences
 **/
std::unique_ptr<column> merge_offsets(std::vector<lists_column_view> const& columns,
                                      size_type total_list_count,
                                      rmm::mr::device_memory_resource* mr,
                                      cudaStream_t stream)
{
  // outgoing offsets
  auto merged_offsets = cudf::make_fixed_width_column(data_type{INT32}, total_list_count + 1);
  mutable_column_device_view d_merged_offsets(*merged_offsets, 0, 0);

  // merge offsets
  size_type shift = 0;
  size_type count = 0;
  thrust::for_each(columns.begin(), columns.end(), [&](lists_column_view const& c) {
    column_device_view offsets(c.offsets(), 0, 0);
    thrust::transform(rmm::exec_policy(0)->on(0),
                      offsets.begin<size_type>(),
                      offsets.end<size_type>(),
                      d_merged_offsets.begin<size_type>() + count,
                      [shift] __device__(size_type offset) { return offset + shift; });
    shift += c.child().size();
    count += offsets.size() - 1;
  });

  return merged_offsets;
}

}  // namespace

/**
 * @copydoc cudf::lists::detail::concatenate
 *
 */
std::unique_ptr<column> concatenate(
  std::vector<column_view> const& columns,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0)
{
  std::vector<lists_column_view> lists_columns;
  std::transform(
    columns.begin(), columns.end(), std::back_inserter(lists_columns), [](column_view const& c) {
      return lists_column_view(c);
    });

  // concatenate data. also prep data needed for offset merging
  std::vector<column_view> leaves;
  size_type total_list_count = 0;
  std::transform(lists_columns.begin(),
                 lists_columns.end(),
                 std::back_inserter(leaves),
                 [&](lists_column_view const& l) {
                   // count total # of lists
                   total_list_count += l.size();
                   // child data. a leaf type (float, int, string, etc)
                   return l.child();
                 });
  auto data = cudf::concatenate(leaves);

  // merge offsets
  auto offsets = merge_offsets(lists_columns, total_list_count, mr, stream);

  // assemble into outgoing list column
  return make_lists_column(total_list_count,
                           std::move(offsets),
                           std::move(data),
                           0,
                           rmm::device_buffer{0, stream, mr},
                           stream,
                           mr);
}

}  // namespace detail
}  // namespace lists
}  // namespace cudf
