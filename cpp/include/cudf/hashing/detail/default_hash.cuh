/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>

namespace cudf::hashing::detail {

/**
 * @brief The default hash algorithm for use within libcudf internal functions
 *
 * This is declared here so it may be changed to another algorithm without modifying
 * all those places that use it. Internal function implementations are encourage to
 * use the `cudf::hashing::detail::default_hash` where possible.
 *
 * @tparam Key The key type for use by the hash class
 */
template <typename Key>
using default_hash = MurmurHash3_x86_32<Key>;

}  // namespace cudf::hashing::detail
