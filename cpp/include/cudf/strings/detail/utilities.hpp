/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/cuda_stream_view.hpp>

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
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return The chars child column for a strings column.
 */
std::unique_ptr<column> create_chars_child_column(
  size_type strings_count,
  size_type null_count,
  size_type bytes,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Create a strings column with no strings.
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return Empty strings column
 */
std::unique_ptr<column> make_empty_strings_column(
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Creates a string_view vector from a strings column.
 *
 * @param strings Strings column instance.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @return Device vector of string_views
 */
rmm::device_vector<string_view> create_string_vector_from_column(
  cudf::strings_column_view strings, rmm::cuda_stream_view stream = rmm::cuda_stream_default);

/**
 * @brief Creates an offsets column from a string_view vector.
 *
 * @param strings Strings column
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return Child offsets column
 */
std::unique_ptr<cudf::column> child_offsets_from_string_vector(
  const rmm::device_vector<string_view>& strings,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @brief Creates a chars column from a string_view vector.
 *
 * @param strings Strings vector
 * @param d_offsets Offsets vector for placing strings into column's memory.
 * @param null_count Number of null strings.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return Child chars column
 */
std::unique_ptr<cudf::column> child_chars_from_string_vector(
  const rmm::device_vector<string_view>& strings,
  const int32_t* d_offsets,
  cudf::size_type null_count,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace detail
}  // namespace strings
}  // namespace cudf
