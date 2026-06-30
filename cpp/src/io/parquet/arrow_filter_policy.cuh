/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/hashing/detail/xxhash_64.cuh>

#include <cuco/bloom_filter_policies.cuh>

#include <cstdint>

namespace cudf::io::parquet::detail {

/**
 * @brief Hasher adapter that exposes the `argument_type` member required by cuco's
 * `parametric_filter_policy`.
 *
 * @tparam Key The type of the values to generate a fingerprint for.
 */
template <class Key>
struct arrow_hasher : cudf::hashing::detail::XXHash_64<Key> {
  using argument_type = Key;
  using cudf::hashing::detail::XXHash_64<Key>::XXHash_64;
};

/**
 * @brief A policy that defines how the Apache Arrow Block-Split Bloom Filter generates and stores a
 * key's fingerprint.
 *
 * Implemented in terms of cuco's `parametric_filter_policy` with the Apache Arrow layout: 256-bit
 * blocks (8 x `uint32_t`), 8 fingerprint bits per key, fully horizontal add (Theta=8) and fully
 * vertical contains (Phi=8). This is bit-compatible with Apache Arrow, as verified by cuCollections
 * `tests/bloom_filter/arrow_compat_test.cu`.
 *
 * Reference:
 * https://github.com/apache/arrow/blob/be1dcdb96b030639c0b56955c4c62f9d6b03f473/cpp/src/parquet/bloom_filter.cc#L219-L230
 *
 * @tparam Key The type of the values to generate a fingerprint for.
 */
template <class Key>
using arrow_filter_policy =
  cuco::parametric_filter_policy<arrow_hasher<Key>, std::uint32_t, 8, 8, 8, 1, 1, 8, false, false>;

}  // namespace cudf::io::parquet::detail
