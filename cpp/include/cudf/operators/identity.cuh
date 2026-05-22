/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/utilities/export.hpp>

#include <cuda/std/optional>

namespace CUDF_EXPORT cudf {
namespace ops {

/**
 * @brief Copies an input value to the output.
 *
 * @tparam T Value type.
 * @param out Destination value.
 * @param a Input value.
 */
template <typename T>
__device__ void identity(T* out, T const* a)
{
  *out = *a;
}

}  // namespace ops
}  // namespace CUDF_EXPORT cudf
