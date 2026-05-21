
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/utilities/export.hpp>

namespace CUDF_EXPORT cudf {
namespace ops {

enum class errc : int { OK = 0, OVERFLOW = 1, DIVISION_BY_ZERO = 2 };

}  // namespace ops
}  // namespace CUDF_EXPORT cudf
