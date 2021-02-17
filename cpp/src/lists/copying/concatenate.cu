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

#include <algorithm>
#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/concatenate.cuh>
#include <cudf/detail/get_value.cuh>
#include <cudf/lists/lists_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

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
 * @param[in] columns               Vector of lists columns to concatenate
 * @param[in] total_list_count      Total number of lists contained in the columns
 * @param[in] stream                CUDA stream used for device memory operations
 * and kernel launches.
 * @param[in] mr                    Device memory resource used to allocate the
 * returned column's device memory.
 */
std::unique_ptr<column> merge_offsets(std::vector<lists_column_view> const& columns,
                                      size_type total_list_count,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{
  // outgoing offsets
  auto merged_offsets = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, total_list_count + 1, mask_state::UNALLOCATED, stream, mr);
  mutable_column_device_view d_merged_offsets(*merged_offsets, 0, 0);

  // merge offsets
  // TODO : this could probably be done as a single gpu operation if done as a kernel.
  size_type shift = 0;
  size_type count = 0;
  std::for_each(columns.begin(), columns.end(), [&](lists_column_view const& c) {
    if (c.size() > 0) {
      // handle sliced columns
      int const local_shift =
        shift -
        (c.offset() > 0 ? cudf::detail::get_value<size_type>(c.offsets(), c.offset(), stream) : 0);
      column_device_view offsets(c.offsets(), nullptr, nullptr);
      thrust::transform(
        rmm::exec_policy(stream),
        offsets.begin<size_type>() + c.offset(),
        offsets.begin<size_type>() + c.offset() + c.size() + 1,
        d_merged_offsets.begin<size_type>() + count,
        [local_shift] __device__(size_type offset) { return offset + local_shift; });

      shift += c.get_sliced_child(stream).size();
      count += c.size();
    }
  });

  return merged_offsets;
}

}  // namespace

/**
 * @copydoc cudf::lists::detail::concatenate
 */
std::unique_ptr<column> concatenate(
  std::vector<column_view> const& columns,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  std::vector<lists_column_view> lists_columns;
  lists_columns.reserve(columns.size());
  std::transform(
    columns.begin(), columns.end(), std::back_inserter(lists_columns), [](column_view const& c) {
      return lists_column_view(c);
    });

  // concatenate children. also prep data needed for offset merging
  std::vector<column_view> children;
  children.reserve(columns.size());
  size_type total_list_count = 0;
  std::for_each(lists_columns.begin(),
                lists_columns.end(),
                [&total_list_count, &children, stream](lists_column_view const& l) {
                  // count total # of lists
                  total_list_count += l.size();
                  children.push_back(l.get_sliced_child(stream));
                });
  auto data = cudf::detail::concatenate(children, stream, mr);

  // merge offsets
  auto offsets = merge_offsets(lists_columns, total_list_count, stream, mr);

  // if any of the input columns have nulls, construct the output mask
  bool const has_nulls =
    std::any_of(columns.cbegin(), columns.cend(), [](auto const& col) { return col.has_nulls(); });
  rmm::device_buffer null_mask = create_null_mask(
    total_list_count, has_nulls ? mask_state::UNINITIALIZED : mask_state::UNALLOCATED);
  if (has_nulls) {
    cudf::detail::concatenate_masks(columns, static_cast<bitmask_type*>(null_mask.data()), stream);
  }

  // assemble into outgoing list column
  return make_lists_column(total_list_count,
                           std::move(offsets),
                           std::move(data),
                           has_nulls ? UNKNOWN_NULL_COUNT : 0,
                           std::move(null_mask),
                           stream,
                           mr);
}

}  // namespace detail
}  // namespace lists
}  // namespace cudf
