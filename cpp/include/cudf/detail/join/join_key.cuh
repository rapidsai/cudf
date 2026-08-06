/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// The key type stored in cudf's cuco-backed join hash tables.
//
// Joins mix build-side and probe-side rows in a single table, so a bare row index would be
// ambiguous; cudf therefore stores each row's index next to its precomputed hash. cuco updates a
// slot with a single atomic, which limits a key to 8 bytes, or to 16 bytes where 128-bit atomics
// exist (sm_90+), and requires that keys have unique object representations so slots can be
// compared bitwise.
//
// A 32-bit size_type pairs with a 32-bit hash in exactly 8 bytes, which is why the shipping build
// has never had to think about this. A 64-bit size_type does not fit, and the naive widening to
// `pair<uint32_t, int64_t>` fails on both counts: 16 bytes exceeds the limit below sm_90, and its
// four bytes of tail padding leave it without unique object representations even above sm_90. So
// there are two viable shapes, selected by what the target architecture supports:
//
//   sm_90+ : widen the cached hash to 64 bits as well, giving a padding-free 16-byte key that
//            keeps the cached-hash optimization intact.
//   below  : keep the 8-byte key and narrow the stored row index to 32 bits, which caps a join
//            build side at 2^31 rows.
//
// Lifting that cap below sm_90 needs either 16-byte key support from cuco or a re-keying of the
// join tables on the row index alone, hashing rows during probing. See rapidsai/cudf#13159.

#pragma once

#include <cudf/detail/row_operator/common_utils.cuh>
#include <cudf/hashing.hpp>
#include <cudf/join/join.hpp>
#include <cudf/types.hpp>

#include <cuco/constraints.cuh>
#include <cuco/pair.cuh>
#include <cuda/std/limits>

#include <cstdint>
#include <type_traits>

namespace cudf::detail {

/// True where a cuco slot can hold a 16-byte key, i.e. where 128-bit atomics are available.
inline constexpr bool join_wide_key_available = cuco::open_addressing_max_key_size >= 16;

/// True where the stored row index has to be narrower than `size_type`.
inline constexpr bool join_narrow_row_index =
  (CUDF_SIZE_TYPE_BITS == 64) && !join_wide_key_available;

/// Hash cached alongside the row index inside a join key.
using join_hash_type = std::
  conditional_t<(CUDF_SIZE_TYPE_BITS == 64) && join_wide_key_available, uint64_t, hash_value_type>;

/// Row index as stored inside a join key.
using join_row_type = std::conditional_t<join_narrow_row_index, int32_t, size_type>;

/// Number of build rows a join hash table can address.
inline constexpr size_type join_max_build_rows =
  static_cast<size_type>(cuda::std::numeric_limits<join_row_type>::max());

/// Sentinel stored in an empty slot's row index, mirroring `cudf::JoinNoMatch`.
inline constexpr join_row_type join_no_match_key = cuda::std::numeric_limits<join_row_type>::min();

/// `join_no_match_key` as a particular stored index type.
template <typename StoredIndex>
inline constexpr StoredIndex join_no_match_index = static_cast<StoredIndex>(join_no_match_key);

namespace join_key_detail {
enum class narrow_lhs_index : int32_t {};
enum class narrow_rhs_index : int32_t {};
}  // namespace join_key_detail

/// Left- and right-side row indices as stored inside a join key.
///
/// These stay distinct types so the comparators that dispatch on which table a key came from keep
/// working. Where the stored index is `size_type` wide they are the row operator's own strong index
/// types, so the key layout is unchanged from the 32-bit build.
using join_lhs_index_type =
  std::conditional_t<join_narrow_row_index, join_key_detail::narrow_lhs_index, row::lhs_index_type>;
using join_rhs_index_type =
  std::conditional_t<join_narrow_row_index, join_key_detail::narrow_rhs_index, row::rhs_index_type>;

/// A join hash table key: a row index next to that row's precomputed hash.
///
/// @tparam StoredIndex How the row index is stored, which is `join_row_type` unless the table needs
/// to distinguish the two input tables at compile time
template <typename StoredIndex = join_row_type>
using join_key = cuco::pair<join_hash_type, StoredIndex>;

/// Narrows a row index for storage in a join key.
///
/// The caller is responsible for rejecting tables with more than `join_max_build_rows` rows.
template <typename StoredIndex = join_row_type>
CUDF_HOST_DEVICE constexpr StoredIndex to_join_index(size_type row) noexcept
{
  return static_cast<StoredIndex>(static_cast<join_row_type>(row));
}

/// Caches a row's hash in the width the join key stores it in.
///
/// The hash itself is only ever 32 bits wide; the wide key zero-extends it purely so that the pair
/// has no padding.
CUDF_HOST_DEVICE constexpr join_hash_type to_join_hash(hash_value_type hash) noexcept
{
  return static_cast<join_hash_type>(hash);
}

/// Recovers the 32-bit hash cached in a join key.
CUDF_HOST_DEVICE constexpr hash_value_type from_join_hash(join_hash_type hash) noexcept
{
  return static_cast<hash_value_type>(hash);
}

/// Widens a row index read back out of a join key.
template <typename StoredIndex>
CUDF_HOST_DEVICE constexpr size_type from_join_index(StoredIndex stored) noexcept
{
  return static_cast<size_type>(static_cast<join_row_type>(stored));
}

/// Recovers the row operator's left-side index from the one stored in a join key.
CUDF_HOST_DEVICE constexpr row::lhs_index_type as_lhs_index(join_lhs_index_type stored) noexcept
{
  return static_cast<row::lhs_index_type>(from_join_index(stored));
}

/// Recovers the row operator's right-side index from the one stored in a join key.
CUDF_HOST_DEVICE constexpr row::rhs_index_type as_rhs_index(join_rhs_index_type stored) noexcept
{
  return static_cast<row::rhs_index_type>(from_join_index(stored));
}

}  // namespace cudf::detail
