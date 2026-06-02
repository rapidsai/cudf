/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/utilities/export.hpp>

#include <cuda/std/cstdint>

namespace CUDF_EXPORT cudf {
namespace detail {
namespace ops {

/**
 * @brief An enumeration of error codes that can occur during operations.
 */
enum class errc : cuda::std::int8_t { SUCCESS = 0, OVERFLOW = 1, DIVISION_BY_ZERO = 2 };

}  // namespace ops
}  // namespace detail
}  // namespace CUDF_EXPORT cudf
