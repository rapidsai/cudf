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

#include <cudf/hashing.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {

/**
 * @copydoc cudf::hash
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> hash(
  table_view const& input,
  hash_id hash_function                     = hash_id::HASH_MURMUR3,
  std::vector<uint32_t> const& initial_hash = {},
  uint32_t seed                             = 0,
  rmm::cuda_stream_view stream              = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr       = rmm::mr::get_current_device_resource());

std::unique_ptr<column> murmur_hash3_32(
  table_view const& input,
  std::vector<uint32_t> const& initial_hash = {},
  rmm::cuda_stream_view stream              = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr       = rmm::mr::get_current_device_resource());

std::unique_ptr<column> md5_hash(
  table_view const& input,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

template <template <typename> class hash_function>
std::unique_ptr<column> serial_murmur_hash3_32(
  table_view const& input,
  uint32_t seed                       = 0,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

}  // namespace detail
}  // namespace cudf
