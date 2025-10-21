/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>

namespace cudf::detail {

/// Sentinel value for `cudf::size_type`
static cudf::size_type constexpr CUDF_SIZE_TYPE_SENTINEL = -1;

/// Default load factor for cuco data structures
static double constexpr CUCO_DESIRED_LOAD_FACTOR = 0.5;

/**
 * @brief Stream-ordered allocator adaptor used for cuco data structures
 *
 * The stream-ordered `rmm::mr::polymorphic_allocator` cannot be used in `cuco` directly since the
 * later expects a standard C++ `Allocator` interface. This allocator helper provides a simple way
 * to handle cuco memory allocation/deallocation with the given `stream` and the rmm default memory
 * resource.
 *
 * @tparam T The allocator's value type.
 */
template <typename T>
using cuco_allocator = rmm::mr::stream_allocator_adaptor<rmm::mr::polymorphic_allocator<T>>;

}  // namespace cudf::detail
