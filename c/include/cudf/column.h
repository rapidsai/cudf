/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/core/c_api.h>
#include <cudf/core/export.h>
#include <cudf/types.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  uintptr_t addr;
} cudfColumn;

typedef cudfColumn* cudfColumn_t;

// WARNING: Non-owning columns from cudfTableGetColumn are valid only while the parent table exists.
CUDF_C_EXPORT cudfError_t cudfColumnDestroy(cudfColumn_t col);
/** @brief Allocate an empty column handle (no data). Destroy with cudfColumnDestroy. */
CUDF_C_EXPORT cudfError_t cudfColumnCreate(cudfColumn_t* col);
CUDF_C_EXPORT cudfError_t cudfColumnGetType(cudfColumn_t col, cudfDataType_t* type);
CUDF_C_EXPORT cudfError_t cudfColumnGetSize(cudfColumn_t col, int64_t* size);
CUDF_C_EXPORT cudfError_t cudfColumnGetNullCount(cudfColumn_t col, int64_t* null_count);

#ifdef __cplusplus
}
#endif
