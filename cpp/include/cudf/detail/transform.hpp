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

#include <cudf/transform.hpp>

namespace cudf {
namespace detail {
/**
 * @copydoc cudf::transform
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 **/
std::unique_ptr<column> transform(
  column_view const& input,
  std::string const& unary_udf,
  data_type output_type,
  bool is_ptx,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0);

/**
 * @copydoc cudf::nans_to_nulls
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 **/
std::pair<std::unique_ptr<rmm::device_buffer>, size_type> nans_to_nulls(
  column_view const& input,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0);

/**
 * @copydoc cudf::bools_to_mask
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 **/
std::pair<std::unique_ptr<rmm::device_buffer>, cudf::size_type> bools_to_mask(
  column_view const& input,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0);

/**
 * @copydoc cudf::encode
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 **/
std::pair<std::unique_ptr<cudf::column>, std::unique_ptr<cudf::column>> encode(
  cudf::column_view const& input,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0);

/**
 * @copydoc cudf::mask_to_bools
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 **/
std::unique_ptr<column> mask_to_bools(
  bitmask_type const* null_mask,
  size_type begin_bit,
  size_type end_bit,
  cudaStream_t stream                 = 0,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource());

}  // namespace detail
}  // namespace cudf
