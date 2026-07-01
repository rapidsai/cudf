/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/utilities/arrow_bloom_filter_policy.cuh>
#include <cudf/hashing/detail/xxhash_64.cuh>

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
 * @brief Policy describing the Apache Arrow Block-Split Bloom Filter, hashing keys with cudf's
 * `XXHash_64` (so that `cudf::string_view` and other cudf types are hashed by content, matching the
 * Apache Parquet/Arrow bloom filter specification).
 *
 * @tparam Key The type of the values to generate a fingerprint for.
 */
template <class Key>
using arrow_filter_policy = cudf::detail::arrow_bloom_filter_policy<arrow_hasher<Key>>;

}  // namespace cudf::io::parquet::detail
