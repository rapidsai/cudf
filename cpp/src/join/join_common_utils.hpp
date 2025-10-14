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

#include <cuco/static_multimap.cuh>
#include <cuco/static_multiset.cuh>
#include <cuda/atomic>

namespace cudf::detail {

constexpr int DEFAULT_JOIN_BLOCK_SIZE = 128;

using pair_type = cuco::pair<hash_value_type, size_type>;

using hash_type = cuco::murmurhash3_32<hash_value_type>;

// Multimap type used for mixed joins. TODO: This is a temporary alias used
// until the mixed joins are converted to using CGs properly. Right now it's
// using a cooperative group of size 1.
using mixed_multimap_type =
  cuco::static_multimap<hash_value_type,
                        size_type,
                        cuda::thread_scope_device,
                        rmm::mr::polymorphic_allocator<char>,
                        cuco::legacy::double_hashing<1, hash_type, hash_type>>;

/**
 * @brief Remaps a hash value to avoid collisions with sentinel values.
 *
 * @param hash The hash value to potentially remap
 * @param sentinel The reserved value
 */
template <typename H, typename S>
constexpr auto remap_sentinel_hash(H hash, S sentinel)
{
  // Arbitrarily choose hash - 1
  return (hash == sentinel) ? (hash - 1) : hash;
}

/**
 * @brief Device functor to create a pair of hash value and row index for use with cuco data
 * structures.
 *
 * @tparam T Type of row index, must be convertible to `size_type`.
 * @tparam Hasher The type of internal hasher to compute row hash.
 */
template <typename Hasher, typename T = size_type>
class make_pair_function {
 public:
  CUDF_HOST_DEVICE make_pair_function(Hasher const& hash, hash_value_type const empty_key_sentinel)
    : _hash{hash}, _empty_key_sentinel{empty_key_sentinel}
  {
  }

  __device__ __forceinline__ auto operator()(size_type i) const noexcept
  {
    // Compute the hash value of row `i`
    auto row_hash_value = remap_sentinel_hash(_hash(i), _empty_key_sentinel);
    return cuco::make_pair(row_hash_value, T{i});
  }

 private:
  Hasher _hash;
  hash_value_type const _empty_key_sentinel;
};

bool is_trivial_join(table_view const& left, table_view const& right, join_kind join_type);
}  // namespace cudf::detail
