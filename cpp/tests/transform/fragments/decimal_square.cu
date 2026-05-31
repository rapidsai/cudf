/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/jit/transform_operator.cuh>

template <>
__device__ void cudf::lto::transform<numeric::decimal32*, numeric::decimal32>(
  numeric::decimal32* out, numeric::decimal32 a)
{
  *out = a * a;
}

template <>
__device__ void cudf::lto::transform<numeric::decimal64*, numeric::decimal64>(
  numeric::decimal64* out, numeric::decimal64 a)
{
  *out = a * a;
}

template <>
__device__ void cudf::lto::transform<numeric::decimal128*, numeric::decimal128>(
  numeric::decimal128* out, numeric::decimal128 a)
{
  *out = a * a;
}
