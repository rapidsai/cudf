/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Ported from cuco's open_addressing kernels and ref_impl to give cudf direct
// control over hash-join probe kernel launches.  The device-side probing logic
// is identical to cuco's static_multiset::count / count_each / retrieve /
// retrieve_outer.  We keep the cuco ref type for hash-table access (storage,
// probing scheme, predicate) and only replace the host-side launch.

#pragma once

#include <cudf/detail/join/join.hpp>
#include <cudf/hashing.hpp>
#include <cudf/types.hpp>

#include <cuco/pair.cuh>

namespace cudf::detail {

// Constants matching the cuco static_multiset configuration used by hash_join.
inline constexpr int PROBE_BLOCK_SIZE = 128;                   // cuco::detail::default_block_size()
inline constexpr int PROBE_CG_SIZE    = DEFAULT_JOIN_CG_SIZE;  // 2

/// The probe key type stored in the hash table: {hash_value, row_index}.
using probe_key_type = cuco::pair<hash_value_type, size_type>;

}  // namespace cudf::detail
