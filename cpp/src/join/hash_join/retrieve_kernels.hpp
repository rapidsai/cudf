/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "kernels_common.cuh"

#include <rmm/cuda_stream_view.hpp>

namespace cudf::detail {

template <bool IsOuter, typename Ref>
std::size_t launch_retrieve(probe_key_type const* keys,
                            cuda::std::int64_t n,
                            size_type* left_output,
                            size_type* right_output,
                            size_type const* match_counts,
                            Ref ref,
                            rmm::cuda_stream_view stream);

}  // namespace cudf::detail
