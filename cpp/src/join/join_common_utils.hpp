/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

#include <cudf/detail/join.hpp>
#include <cudf/detail/utilities/device_atomics.cuh>
#include <cudf/detail/utilities/hash_functions.cuh>
#include <cudf/join.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_view.hpp>

#include <hash/hash_allocator.cuh>
#include <hash/helper_functions.cuh>

#include <rmm/mr/device/polymorphic_allocator.hpp>

#include <cuco/static_map.cuh>
#include <cuco/static_multimap.cuh>

#include <limits>

namespace cudf {
namespace detail {

constexpr int DEFAULT_JOIN_BLOCK_SIZE = 128;
constexpr int DEFAULT_JOIN_CACHE_SIZE = 128;
constexpr size_type JoinNoneValue     = std::numeric_limits<size_type>::min();

using pair_type = cuco::pair<hash_value_type, size_type>;

using hash_type = cuco::murmurhash3_32<hash_value_type>;

using hash_table_allocator_type = rmm::mr::stream_allocator_adaptor<default_allocator<char>>;

using multimap_type = cudf::hash_join::impl_type::map_type;

// Multimap type used for mixed joins. TODO: This is a temporary alias used
// until the mixed joins are converted to using CGs properly. Right now it's
// using a cooperative group of size 1.
using mixed_multimap_type = cuco::static_multimap<hash_value_type,
                                                  size_type,
                                                  cuda::thread_scope_device,
                                                  hash_table_allocator_type,
                                                  cuco::double_hashing<1, hash_type, hash_type>>;

using semi_map_type = cuco::
  static_map<hash_value_type, size_type, cuda::thread_scope_device, hash_table_allocator_type>;

using row_hash_legacy = cudf::row_hasher<default_hash, cudf::nullate::DYNAMIC>;

using row_equality_legacy = cudf::row_equality_comparator<cudf::nullate::DYNAMIC>;

bool is_trivial_join(table_view const& left, table_view const& right, join_kind join_type);
}  // namespace detail
}  // namespace cudf
