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
#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/table/table_view.hpp>

#include <vector>

namespace cudf {
//! Inner interfaces and implementations
namespace detail {
/**
 * @copydoc cudf::concatenate_masks(std::vector<column_view>
 * const&,rmm::mr::device_memory_resource*)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
void concatenate_masks(rmm::device_vector<column_device_view> const& d_views,
                       rmm::device_vector<size_t> const& d_offsets,
                       bitmask_type* dest_mask,
                       size_type output_size,
                       cudaStream_t stream);

/**
 * @copydoc cudf::concatenate_masks(std::vector<column_view> const&,bitmask_type*)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
void concatenate_masks(std::vector<column_view> const& views,
                       bitmask_type* dest_mask,
                       cudaStream_t stream);

/**
 * @copydoc cudf::concatenate(std::vector<column_view> const&,rmm::mr::device_memory_resource*)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> concatenate(
  std::vector<column_view> const& columns_to_concat,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0);

/**
 * @copydoc cudf::concatenate(std::vector<table_view> const&,rmm::mr::device_memory_resource*)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> concatenate(
  std::vector<table_view> const& tables_to_concat,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0);

}  // namespace detail
}  // namespace cudf
