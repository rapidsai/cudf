/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/types.hpp>

#include <limits>

namespace CUDF_EXPORT cudf {
namespace detail {

constexpr int DEFAULT_JOIN_CG_SIZE = 2;

constexpr size_type JoinNoneValue = std::numeric_limits<size_type>::min();

enum class join_kind { INNER_JOIN, LEFT_JOIN, FULL_JOIN, LEFT_SEMI_JOIN, LEFT_ANTI_JOIN };

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
