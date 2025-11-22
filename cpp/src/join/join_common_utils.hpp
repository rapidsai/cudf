/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/hashing.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>

#include <cuco/static_multimap.cuh>
#include <cuco/static_multiset.cuh>
#include <cuda/atomic>

namespace cudf::detail {

constexpr int DEFAULT_JOIN_BLOCK_SIZE = 128;

using pair_type = cuco::pair<hash_value_type, size_type>;

using hash_type = cuco::murmurhash3_32<hash_value_type>;

/**
 * @brief A custom comparator used for the mixed join multiset insertion
 */
struct mixed_join_always_not_equal {
  __device__ constexpr bool operator()(pair_type const&, pair_type const&) const noexcept
  {
    // multiset always insert
    return false;
  }
};

/**
 * @brief Hash functions for double hashing in mixed joins.
 *
 * These hashers implement a double hashing scheme for the mixed join multiset:
 *
 * - mixed_join_hasher1: Determines the initial probe slot for a given key. We simply use
 *   the precomputed row hash value, which is the first element of our (row_hash, row_index) pair.
 *
 * - mixed_join_hasher2: Determines the step size for the probing sequence. This allows keys
 *   with the same hash value to have different step sizes, helping to avoid secondary clustering.
 *
 * Note: Strictly speaking, this setup does not truly avoid secondary clustering because rows with
 * the same hash value still receive the same step size. A true secondary clustering avoidance
 * method would compute a different hash value for each row. However, based on performance testing,
 * this current approach actually delivers better performance than computing row hashes with a
 * different hasher.
 */
struct mixed_join_hasher1 {
  __device__ constexpr hash_value_type operator()(pair_type const& key) const noexcept
  {
    return key.first;
  }
};

struct mixed_join_hasher2 {
  mixed_join_hasher2(hash_value_type seed) : _hash{seed} {}

  __device__ constexpr hash_value_type operator()(pair_type const& key) const noexcept
  {
    return _hash(key.first);
  }

 private:
  hash_type _hash;
};

using mixed_multiset_type =
  cuco::static_multiset<pair_type,
                        cuco::extent<std::size_t>,
                        cuda::thread_scope_device,
                        mixed_join_always_not_equal,
                        cuco::double_hashing<1, mixed_join_hasher1, mixed_join_hasher2>,
                        rmm::mr::polymorphic_allocator<char>,
                        cuco::storage<2>>;

bool is_trivial_join(table_view const& left, table_view const& right, join_kind join_type);
}  // namespace cudf::detail
