/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/hashing.hpp>
#include <cudf/types.hpp>

namespace cudf::hashing::detail {

/**
 * @brief Forward declaration only (no `murmur_jit_hash_dispatcher` here).
 *
 * Per-type explicit specializations must appear in the TU before any use of that specialization.
 * Hasher / noop fragment TUs include this header, then define `template <> ... murmur_jit_hasher<T>`.
 * The entry kernel includes `murmurhash3_x86_32_jit_device.cuh`, which pulls this in and adds the
 * dispatcher that calls `murmur_jit_hasher` for every storage type.
 */
template <typename T>
extern __device__ hash_value_type murmur_jit_hasher(column_device_view col,
                                                    uint32_t seed,
                                                    bool nullable,
                                                    size_type row_index);

}  // namespace cudf::hashing::detail
