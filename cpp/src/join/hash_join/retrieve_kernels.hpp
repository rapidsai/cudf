/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "kernels_common.cuh"

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <memory>
#include <utility>

namespace cudf::detail {

/**
 * @brief Probes the hash table for each key and writes matching index pairs.
 *
 * Internally computes per-row output offsets via exclusive scan on match_counts,
 * derives the total output size, allocates output buffers, and launches the
 * retrieve kernel.
 *
 * @return A pair of device vectors [left_indices, right_indices].
 */
template <bool IsOuter, typename Ref>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
launch_retrieve(probe_key_type const* keys,
                cuda::std::int64_t n,
                size_type const* match_counts,
                Ref ref,
                rmm::cuda_stream_view stream,
                rmm::device_async_resource_ref mr);

}  // namespace cudf::detail
