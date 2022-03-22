/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cudf/lists/contains.hpp>
#include <cudf/lists/lists_column_view.hpp>

namespace cudf {
namespace lists {
namespace detail {

/**
 * @copydoc cudf::lists::index_of(cudf::lists_column_view const&,
 *                                cudf::scalar const&,
 *                                duplicate_find_option,
 *                                rmm::mr::device_memory_resource*)
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> index_of(
  cudf::lists_column_view const& lists,
  cudf::scalar const& search_key,
  cudf::lists::duplicate_find_option find_option,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @copydoc cudf::lists::index_of(cudf::lists_column_view const&,
 *                                cudf::column_view const&,
 *                                duplicate_find_option,
 *                                rmm::mr::device_memory_resource*)
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> index_of(
  cudf::lists_column_view const& lists,
  cudf::column_view const& search_keys,
  cudf::lists::duplicate_find_option find_option,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @copydoc cudf::lists::contains(cudf::lists_column_view const&,
 *                                cudf::scalar const&,
 *                                rmm::mr::device_memory_resource*)
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> contains(
  cudf::lists_column_view const& lists,
  cudf::scalar const& search_key,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

/**
 * @copydoc cudf::lists::contains(cudf::lists_column_view const&,
 *                                cudf::column_view const&,
 *                                rmm::mr::device_memory_resource*)
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> contains(
  cudf::lists_column_view const& lists,
  cudf::column_view const& search_keys,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());
}  // namespace detail
}  // namespace lists
}  // namespace cudf
