/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <vector>

namespace cudf {
namespace detail {

/**
 * @copydoc cudf::sorted_order
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> sorted_order(
  table_view const& input,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::cuda_stream_view stream                   = cudf::default_stream_value,
  rmm::mr::device_memory_resource* mr            = rmm::mr::get_current_device_resource());

/**
 * @copydoc cudf::stable_sorted_order
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> stable_sorted_order(
  table_view const& input,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::cuda_stream_view stream                   = cudf::default_stream_value,
  rmm::mr::device_memory_resource* mr            = rmm::mr::get_current_device_resource());

/**
 * @copydoc cudf::sort_by_key
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> sort_by_key(
  table_view const& values,
  table_view const& keys,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::cuda_stream_view stream                   = cudf::default_stream_value,
  rmm::mr::device_memory_resource* mr            = rmm::mr::get_current_device_resource());

/**
 * @copydoc cudf::stable_sort_by_key
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> stable_sort_by_key(
  table_view const& values,
  table_view const& keys,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::cuda_stream_view stream                   = cudf::default_stream_value,
  rmm::mr::device_memory_resource* mr            = rmm::mr::get_current_device_resource());

/**
 * @copydoc cudf::segmented_sorted_order
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> segmented_sorted_order(
  table_view const& keys,
  column_view const& segment_offsets,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::cuda_stream_view stream                   = cudf::default_stream_value,
  rmm::mr::device_memory_resource* mr            = rmm::mr::get_current_device_resource());

/**
 * @copydoc cudf::stable_segmented_sorted_order
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> stable_segmented_sorted_order(
  table_view const& keys,
  column_view const& segment_offsets,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::cuda_stream_view stream                   = cudf::default_stream_value,
  rmm::mr::device_memory_resource* mr            = rmm::mr::get_current_device_resource());

/**
 * @copydoc cudf::segmented_sort_by_key
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> segmented_sort_by_key(
  table_view const& values,
  table_view const& keys,
  column_view const& segment_offsets,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::cuda_stream_view stream                   = cudf::default_stream_value,
  rmm::mr::device_memory_resource* mr            = rmm::mr::get_current_device_resource());

/**
 * @copydoc cudf::stable_segmented_sort_by_key
 *
 * @param[in] stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> stable_segmented_sort_by_key(
  table_view const& values,
  table_view const& keys,
  column_view const& segment_offsets,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::cuda_stream_view stream                   = cudf::default_stream_value,
  rmm::mr::device_memory_resource* mr            = rmm::mr::get_current_device_resource());

/**
 * @copydoc cudf::sort
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> sort(
  table_view const& values,
  std::vector<order> const& column_order         = {},
  std::vector<null_order> const& null_precedence = {},
  rmm::cuda_stream_view stream                   = cudf::default_stream_value,
  rmm::mr::device_memory_resource* mr            = rmm::mr::get_current_device_resource());

}  // namespace detail
}  // namespace cudf
