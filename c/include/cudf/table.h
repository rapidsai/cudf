/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column.h>
#include <cudf/core/c_api.h>
#include <cudf/core/export.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  uintptr_t addr;
} cudfTable;

typedef cudfTable* cudfTable_t;

CUDF_C_EXPORT cudfError_t cudfTableDestroy(cudfTable_t table);
/** @brief Allocate an empty table handle (no columns). Destroy with cudfTableDestroy. */
CUDF_C_EXPORT cudfError_t cudfTableCreate(cudfTable_t* table);
CUDF_C_EXPORT cudfError_t cudfTableGetNumColumns(cudfTable_t table, int32_t* num_cols);
CUDF_C_EXPORT cudfError_t cudfTableGetNumRows(cudfTable_t table, int64_t* num_rows);
CUDF_C_EXPORT cudfError_t cudfTableGetColumn(cudfTable_t table, int32_t index, cudfColumn_t* col);

#ifdef __cplusplus
}
#endif
