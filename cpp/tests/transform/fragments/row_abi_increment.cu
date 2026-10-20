/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf/row_operators.cuh>
using namespace cudf::row_abi;
extern "C" __device__ status transform_operator(void const* inputs,
                                                void const* outputs,
                                                void*,
                                                i32 row)
{
  auto input = load_i32_nullable(inputs, 0, row);
  set_validity(outputs, 0, row, input.validity);
  if (!input.validity) { return success; }
  auto value = cudf_row_op_ADD_OVERFLOW_i32_i32(input.value, 1);
  if (value.error != success) { return value.error; }
  return set_i32(outputs, 0, row, value.value);
}
