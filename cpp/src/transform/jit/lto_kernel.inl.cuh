/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/jit/transform_params.cuh>
#include <cudf/types.hpp>

extern "C" {
__device__ void transform_operator(cudf::lto::transform_params const* params);

__global__ void transform_kernel(void const* outputs,
                                 void const* span_outputs,
                                 void const* inputs,
                                 void* user_data,
                                 cudf::size_type num_rows)
{
  auto const start  = cudf::detail::grid_1d::global_thread_id();
  auto const stride = cudf::detail::grid_1d::grid_stride();
  auto const size   = num_rows;

  for (auto i = start; i < size; i += stride) {
    cudf::lto::transform_params p{.inputs       = inputs,
                                  .user_data    = user_data,
                                  .outputs      = outputs,
                                  .span_outputs = span_outputs,
                                  .row_index    = static_cast<cudf::size_type>(i)};
    transform_operator(&p);
  }
}
}
