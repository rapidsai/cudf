/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/jit/transform_operator.cuh>

template <>
__device__ void cudf::lto::transform<float*, float, float>(float* out, float a, float b)
{
  *out = a * a + b * b;
}
