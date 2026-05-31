/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace cudf {
namespace lto {

/// @brief The operator for the transform operation.
/// @tparam T the types of the operator arguments, which can be either input or output arguments.
/// The output argument must be the first argument if there are multiple arguments.
/// @param args The arguments for the operator.
template <typename... T>
__device__ void transform(T... args);

}  // namespace lto
}  // namespace cudf
