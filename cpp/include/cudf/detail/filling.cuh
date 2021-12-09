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

#pragma once

#include <cudf/filling.hpp>

#include <rmm/device_uvector.hpp>

namespace cudf {
namespace detail {

/**
 * @brief Generate a column containing complement of the indices given from the input ranges defined
 * by the given pair of begin/end iterators.
 *
 * TBA
 * // Note: This internal API is used to call on each of the input column of the public API.
 *
 * The output buffer starting by @p output_begin must be large enough to hold the output data (i.e.,
 * it has the size of at least the largest input index + 1).
 *
 * @tparam Iterator Used as input to scan for the input indices.
 * TBA
 * @param size Size that defines the range of indices in both input and output.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 *
 * @return An array in device memory containing the complement indices.
 */
template <typename InputIterator, typename OutputIterator>
rmm::device_uvector<size_type> complement(
  host_span<std::pair<InputIterator, OutputIterator> const> input_ranges,
  size_type size,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  //
}

}  // namespace detail
}  // namespace cudf
