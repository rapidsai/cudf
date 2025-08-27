/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/join/join.hpp>
#include <cudf/hashing.hpp>
#include <cudf/table/table_view.hpp>

#include <cuco/static_multiset.cuh>
#include <cuda/atomic>

namespace cudf::detail {

constexpr int DEFAULT_JOIN_BLOCK_SIZE = 128;

using pair_type = cuco::pair<hash_value_type, size_type>;

using hash_type = cuco::murmurhash3_32<hash_value_type>;

// Comparator that always returns false to ensure all values are inserted (like hash_join)
struct mixed_join_always_not_equal {
  __device__ constexpr bool operator()(cuco::pair<hash_value_type, size_type> const&,
                                       cuco::pair<hash_value_type, size_type> const&) const noexcept
  {
    // multiset always insert
    return false;
  }
};

struct mixed_join_hasher1 {
  __device__ constexpr hash_value_type operator()(
    cuco::pair<hash_value_type, size_type> const& key) const noexcept
  {
    return key.first;
  }
};

struct mixed_join_hasher2 {
  mixed_join_hasher2(hash_value_type seed) : _hash{seed} {}

  __device__ constexpr hash_value_type operator()(
    cuco::pair<hash_value_type, size_type> const& key) const noexcept
  {
    return _hash(key.first);
  }

 private:
  hash_type _hash;
};

// Multimap type used for mixed joins
using mixed_multimap_type =
  cuco::static_multiset<cuco::pair<hash_value_type, size_type>,
                        cuco::extent<std::size_t>,
                        cuda::thread_scope_device,
                        mixed_join_always_not_equal,
                        cuco::double_hashing<1, mixed_join_hasher1, mixed_join_hasher2>,
                        cudf::detail::cuco_allocator<char>,
                        cuco::storage<2>>;
template <typename Tag>
using mixed_join_hash_table_ref_t = mixed_multimap_type::ref_type<Tag>;

bool is_trivial_join(table_view const& left, table_view const& right, join_kind join_type);
}  // namespace cudf::detail
