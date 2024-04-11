/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace strings {
namespace detail {

/**
 * @brief Internal API to copy a range of string elements out-of-place from
 * a source column to a target column
 *
 * Creates a new column as if an in-place copy was performed into `target`.
 * The elements indicated by the indices `source_begin`, `source_end`)
 * replace with the elements in the target column starting at `target_begin`.
 * Elements outside the range are copied from `target` into the new target
 * column to return.
 *
 * @throws cudf::logic_error for invalid range (if `target_begin < 0`,
 * or `target_begin >= target.size()`,
 * or `target_begin + (source_end-source_begin)` > target.size()`).
 *
 * @param source The strings column to copy from inside the `target_begin` range
 * @param target The strings column to copy from outside the range
 * @param source_end The index of the first element in the source range
 * @param source_end The index of the last element in the source range (exclusive)
 * @param target_begin The starting index of the target range (inclusive)
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return The result target column
 */
std::unique_ptr<column> copy_range(strings_column_view const& source,
                                   strings_column_view const& target,
                                   size_type source_begin,
                                   size_type source_end,
                                   size_type target_begin,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr);

}  // namespace detail
}  // namespace strings
}  // namespace cudf
