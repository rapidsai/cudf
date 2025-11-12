/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace cudf {
namespace detail {

/**
 * @brief The enum specifying which sorting method to use (stable or unstable).
 */
enum class sort_method : bool { STABLE, UNSTABLE };

}  // namespace detail
}  // namespace cudf
