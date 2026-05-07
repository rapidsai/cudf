/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/hashing.hpp>
#include <cudf/types.hpp>

namespace cudf::hashing::detail {

/**
 * @brief Forward declarations for the Murmur JIT link graph.
 *
 * Per-type explicit specializations must appear in the TU before any use of that specialization.
 * Hasher / noop fragment TUs include this header, then define `template <> ...
 * murmur_jit_hasher<T>`. The ``murmurhash_jit_dispatch`` fatbin defines
 * ``murmur_jit_hash_dispatcher``; the entry kernel and hasher fragments only declare it here.
 *
 * `murmur_jit_hash_dispatcher` is declared here so `murmurhash3_x86_32_lto.cuh` can call it from
 * nested/dictionary paths without including the dispatcher body before `murmur_jit_hasher`
 * specializations in hasher fragment TUs.
 */
extern __device__ hash_value_type murmur_jit_hash_dispatcher(column_device_view col,
                                                             uint32_t seed,
                                                             bool nullable,
                                                             size_type row_index);

template <typename T>
extern __device__ hash_value_type
murmur_jit_hasher(column_device_view col, uint32_t seed, bool nullable, size_type row_index);

}  // namespace cudf::hashing::detail
