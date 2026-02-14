/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/jit/lto/transform_params.cuh>
#include <cudf/types.hpp>

extern "C" {

__device__ void transform_operator(cudf::lto::transform_params params);

__global__ void transform_kernel(void* __restrict__ const* __restrict__ scope, int32_t num_rows)
{
  auto start  = cudf::detail::grid_1d::global_thread_id();
  auto stride = cudf::detail::grid_1d::grid_stride();
  auto size   = num_rows;

  for (auto i = start; i < size; i += stride) {
    cudf::lto::transform_params params{.scope     = scope,
                                       .row_index = static_cast<cudf::size_type>(i)};
    transform_operator(params);
  }
}
}
