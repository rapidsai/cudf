/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/utilities/export.hpp>

#include <cuda/std/cstdint>

namespace CUDF_EXPORT cudf {

/**
 * @brief An enumeration of error codes that can occur during operations.
 */
enum class errc : cuda::std::int8_t {
  SUCCESS          = 0,
  OVERFLOW         = 1,
  DIVISION_BY_ZERO = 2,
};

/**
 * @brief Convert an `errc` error code to a human-readable string.
 * @param error The error code to convert
 * @return A C-string representing the error code
 */
[[nodiscard]] constexpr char const* to_string(errc error)
{
  switch (error) {
    case errc::SUCCESS: return "SUCCESS";
    case errc::OVERFLOW: return "OVERFLOW";
    case errc::DIVISION_BY_ZERO: return "DIVISION_BY_ZERO";
    default: return "UNKNOWN_ERROR";
  }
}

}  // namespace CUDF_EXPORT cudf
