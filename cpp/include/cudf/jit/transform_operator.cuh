/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

namespace cudf {
namespace lto {

/// @brief The unary operator for the transform operation.
/// @tparam Out The output type of the operator.
/// @tparam In0 The input type of the operator.
/// @param out The output destination for the operator result.
/// @param a The input value for the operator.
template <typename Out, typename In0>
__device__ void unary_operator(Out* __restrict__ out, In0 a);

/// @brief The binary operator for the transform operation.
/// @tparam Out The output type of the operator.
/// @tparam In0 The first input type of the operator.
/// @tparam In1 The second input type of the operator.
/// @param out The output destination for the operator result.
/// @param a The first input value for the operator.
/// @param b The second input value for the operator.
template <typename Out, typename In0, typename In1>
__device__ void binary_operator(Out* __restrict__ out, In0 a, In1 b);

}  // namespace lto
}  // namespace cudf
