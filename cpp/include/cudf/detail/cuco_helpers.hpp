/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

namespace cudf::detail {

/// Sentinel value for `cudf::size_type`
static cudf::size_type constexpr CUDF_SIZE_TYPE_SENTINEL = -1;

/// Default load factor for cuco data structures
static double constexpr CUCO_DESIRED_LOAD_FACTOR = 0.5;

}  // namespace cudf::detail
