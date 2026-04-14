/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "kernels_common.cuh"

#include <rmm/cuda_stream_view.hpp>

#include <cuda/std/cstdint>

namespace cudf::detail {

/// Launch the count_each kernel.
template <bool IsOuter, typename Ref>
void launch_count_each(probe_key_type const* keys,
                       cuda::std::int64_t n,
                       size_type* output,
                       Ref ref,
                       rmm::cuda_stream_view stream);

}  // namespace cudf::detail
