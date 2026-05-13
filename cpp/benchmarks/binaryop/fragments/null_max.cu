

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/jit/transform_operator.cuh>

#include <cuda/std/optional>

template <>
__device__ void cudf::lto::binary_operator<cuda::std::optional<numeric::decimal32>,
                                           cuda::std::optional<numeric::decimal32>,
                                           cuda::std::optional<numeric::decimal32>>(
  cuda::std::optional<numeric::decimal32>* __restrict__ out,
  cuda::std::optional<numeric::decimal32> a,
  cuda::std::optional<numeric::decimal32> b)
{
  if (a.has_value() || b.has_value()) {
    *out = (a.has_value() && (!b.has_value() || (*a > *b))) ? *a : *b;
  } else {
    *out = cuda::std::nullopt;
  }
}
