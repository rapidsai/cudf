/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

//! NVText APIs
namespace CUDF_EXPORT nvtext {
/**
 * @addtogroup nvtext_replace
 * @{
 * @file
 */

/**
 * @brief Returns a duplicate strings found in the given input
 *
 * The internal implementation creates a suffix array of the input which
 * requires ~10x the input size for temporary memory.
 *
 * The output includes any strings of at least `min_width` bytes that
 * appear more than once in the entire input.
 *
 * @param input Strings column to dedup
 * @param min_width Minimum number of bytes must match to specify a duplicate
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column with updated strings
 */
std::unique_ptr<cudf::column> substring_deduplicate(
  cudf::strings_column_view const& input,
  cudf::size_type min_width,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Builds a suffix array for the input strings column
 *
 * @param input Strings column to build suffix array for
 * @param min_width Minimum number of bytes must match to specify a duplicate
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Sorted suffix array and corresponding sizes
 */
std::unique_ptr<rmm::device_uvector<int64_t>> build_suffix_array(
  cudf::strings_column_view const& input,
  cudf::size_type min_width,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace CUDF_EXPORT nvtext
