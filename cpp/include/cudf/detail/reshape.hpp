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

#include <cudf/types.hpp>

#include <memory>

namespace cudf {
namespace detail {
/**
 * @copydoc cudf::tile
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
std::unique_ptr<table> tile(
  table_view const& input,
  size_type count,
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());
}  // namespace detail

/**
 * @brief Converts primitive types and string columns to lists of bytes, mimics Spark's cast to
 * binary type.
 *
 * @param inpu_column column to be converted to lists of bytes.
 * @param configuration configuration to retain or flip the endianness of a row.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @param  stream CUDA stream to launch kernels within
 *
 * @return The column containing the lists of bytes.
 */
std::unique_ptr<column> byte_cast(
  column_view const& input_column,
  endianess_policy configuration,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0);

}  // namespace cudf
