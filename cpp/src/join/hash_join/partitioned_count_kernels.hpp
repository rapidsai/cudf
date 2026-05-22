/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "kernels_common.cuh"

#include <rmm/cuda_stream_view.hpp>

namespace cudf::detail {

/// Launch the partitioned_count kernel.
template <bool IsOuter, typename Ref>
void launch_partitioned_count(probe_key_type const* keys,
                              thread_index_type n,
                              size_type* output,
                              Ref ref,
                              rmm::cuda_stream_view stream);

}  // namespace cudf::detail
