/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>

#include <cuda/atomic>

namespace cudf::io::orc::detail {

/**
 * @brief Record a decoding failure without a shared error count wrapping back to zero.
 *
 * The reader checks for any error after synchronizing the decoding stream. Columns and
 * stripes share the same status, so their combined number of failures is not row-count bounded.
 */
inline __device__ void set_decode_error(size_type* status)
{
  cuda::atomic_ref<size_type, cuda::thread_scope_device> error{*status};
  error.fetch_or(1, cuda::std::memory_order_relaxed);
}

}  // namespace cudf::io::orc::detail
