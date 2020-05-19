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

#include <cudf/hashing.hpp>

namespace cudf {
namespace detail {
/**
 * @copydoc cudf::experimental::hash_partition
 *
 * @param stream Optional stream to use for allocations and copies
 */
std::pair<std::unique_ptr<experimental::table>, std::vector<size_type>> hash_partition(
  table_view const& input,
  std::vector<size_type> const& columns_to_hash,
  int num_partitions,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
  cudaStream_t stream                 = 0);

/**
 * @copydoc cudf::hash
 *
 * @param stream Optional stream to use for allocations and copies
 */
std::unique_ptr<column> hash(table_view const& input,
                             std::vector<uint32_t> const& initial_hash = {},
                             rmm::mr::device_memory_resource* mr = rmm::mr::get_default_resource(),
                             cudaStream_t stream                 = 0);

}  // namespace detail
}  // namespace cudf
