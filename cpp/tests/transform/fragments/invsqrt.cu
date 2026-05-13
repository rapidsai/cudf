/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/jit/transform_operator.cuh>

template <>
__device__ void cudf::lto::unary_operator<float, float>(float* __restrict__ out, float a)
{
  *out = 1.0F / sqrtf(a);
}
