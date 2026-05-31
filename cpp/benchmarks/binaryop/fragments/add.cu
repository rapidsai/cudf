/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/jit/transform_operator.cuh>
#include <cudf/wrappers/timestamps.hpp>

template <>
__device__ void cudf::lto::transform<int32_t*, int32_t, int32_t>(int32_t* out, int32_t a, int32_t b)
{
  *out = a + b;
}

template <>
__device__ void cudf::lto::transform<float*, float, float>(float* out, float a, float b)
{
  *out = a + b;
}
