/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/strings_column_view.hpp>

namespace cudf {
namespace strings {
namespace detail {
/**
 * @brief Scatters strings into a copy of the target column
 * according to a scatter map.
 *
 * The scatter is performed according to the scatter iterator such that row
 * `scatter_map[i]` of the output column is replaced by the source string.
 * All other rows of the output column equal corresponding rows of the target table.
 *
 * If the same index appears more than once in the scatter map, the result is
 * undefined.
 *
 * The caller must update the null mask in the output column.
 *
 * @tparam SourceIterator must produce string_view objects
 * @tparam MapIterator must produce index values within the target column.
 *
 * @param source The iterator of source strings to scatter into the output column.
 * @param scatter_map Iterator of indices into the output column.
 * @param target The set of columns into which values from the source column
 *        are to be scattered.
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return New strings column.
 */
template <typename SourceIterator, typename MapIterator>
std::unique_ptr<column> scatter(
  SourceIterator begin,
  SourceIterator end,
  MapIterator scatter_map,
  strings_column_view const& target,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0)
{
  auto strings_count = target.size();
  if (strings_count == 0) return make_empty_strings_column(mr, stream);

  // create null mask -- caller must update this
  rmm::device_buffer null_mask{0, stream, mr};
  if (target.has_nulls()) null_mask = copy_bitmask(target.parent(), stream, mr);

  // create string vectors
  rmm::device_vector<string_view> target_vector = create_string_vector_from_column(target, stream);
  // do the scatter
  thrust::scatter(
    rmm::exec_policy(stream)->on(stream), begin, end, scatter_map, target_vector.begin());

  // build offsets column
  auto offsets_column = child_offsets_from_string_vector(target_vector, mr, stream);
  // build chars column
  auto chars_column = child_chars_from_string_vector(
    target_vector, offsets_column->view().data<int32_t>(), 0, mr, stream);

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             std::move(chars_column),
                             UNKNOWN_NULL_COUNT,
                             std::move(null_mask),
                             stream,
                             mr);
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
