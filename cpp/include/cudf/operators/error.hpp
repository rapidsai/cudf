
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/utilities/export.hpp>

namespace CUDF_EXPORT cudf {
namespace ops {

enum errc : int { OK = 0, OVERFLOW = 1, DIVISION_BY_ZERO = 2 };

inline char const* to_string(errc error_code)
{
  switch (error_code) {
    case errc::OK: return "cudf::ops::errc::OK";
    case errc::OVERFLOW: return "cudf::ops::errc::OVERFLOW";
    case errc::DIVISION_BY_ZERO: return "cudf::ops::errc::DIVISION_BY_ZERO";
    default: return "UNKNOWN_ERROR";
  }
}

enum class error_mode : char { IGNORE = 0, ANY_ROW = 1 };

}  // namespace ops
}  // namespace CUDF_EXPORT cudf
