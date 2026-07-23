/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column.h>
#include <cudf/core/c_api.h>
#include <cudf/core/export.h>
#include <cudf/types.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  CUDF_BINARY_OP_ADD = 0,
  CUDF_BINARY_OP_SUB,
  CUDF_BINARY_OP_MUL,
  CUDF_BINARY_OP_DIV,
  CUDF_BINARY_OP_TRUE_DIV,
  CUDF_BINARY_OP_FLOOR_DIV,
  CUDF_BINARY_OP_MOD,
  CUDF_BINARY_OP_EQUAL,
  CUDF_BINARY_OP_NOT_EQUAL,
  CUDF_BINARY_OP_LESS,
  CUDF_BINARY_OP_GREATER,
  CUDF_BINARY_OP_LESS_EQUAL,
  CUDF_BINARY_OP_GREATER_EQUAL
} cudfBinaryOp_t;

CUDF_C_EXPORT cudfError_t cudfBinaryOpColumns(cudfColumn_t lhs,
                                              cudfColumn_t rhs,
                                              cudfBinaryOp_t op,
                                              cudfDataType_t output_type,
                                              cudaStream_t stream,
                                              cudfColumn_t* result);

#ifdef __cplusplus
}
#endif
