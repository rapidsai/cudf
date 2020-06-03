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

#include <cuda_runtime.h>
#include <rmm/thrust_rmm_allocator.h>
#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>

namespace cudf {
namespace strings {
namespace detail {
/**
 * @brief Create a chars column to be a child of a strings column.
 * This will return the properly sized column to be filled in by the caller.
 *
 * @param strings_count Number of strings in the column.
 * @param null_count Number of null string entries in the column.
 * @param bytes Number of bytes for the chars column.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return The chars child column for a strings column.
 */
std::unique_ptr<column> create_chars_child_column(
  size_type strings_count,
  size_type null_count,
  size_type bytes,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0);

/**
 * @brief Create a strings column with no strings.
 *
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return Empty strings column
 */
std::unique_ptr<column> make_empty_strings_column(
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(), cudaStream_t stream = 0);

/**
 * @brief Creates a string_view vector from a strings column.
 *
 * @param strings Strings column instance.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return Device vector of string_views
 */
rmm::device_vector<string_view> create_string_vector_from_column(cudf::strings_column_view strings,
                                                                 cudaStream_t stream = 0);

/**
 * @brief Creates an offsets column from a string_view vector.
 *
 * @param strings Strings column
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return Child offsets column
 */
std::unique_ptr<cudf::column> child_offsets_from_string_vector(
  const rmm::device_vector<string_view>& strings,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0);

/**
 * @brief Creates a chars column from a string_view vector.
 *
 * @param strings Strings vector
 * @param d_offsets Offsets vector for placing strings into column's memory.
 * @param null_count Number of null strings.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return Child chars column
 */
std::unique_ptr<cudf::column> child_chars_from_string_vector(
  const rmm::device_vector<string_view>& strings,
  const int32_t* d_offsets,
  cudf::size_type null_count,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0);

}  // namespace detail
}  // namespace strings
}  // namespace cudf
