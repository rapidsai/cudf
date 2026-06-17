/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column.h>
#include <cudf/core/c_api.h>
#include <cudf/core/export.h>
#include <cudf/table.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ArrowSchema;
typedef struct ArrowSchema ArrowSchema;

struct ArrowDeviceArray;
typedef struct ArrowDeviceArray ArrowDeviceArray;

struct ArrowArray;
typedef struct ArrowArray ArrowArray;

CUDF_C_EXPORT cudfError_t cudfTableFromArrow(ArrowSchema* schema,
                                             ArrowDeviceArray* input,
                                             cudaStream_t stream,
                                             cudfTable_t* table);

CUDF_C_EXPORT cudfError_t cudfTableToArrow(cudfTable_t table,
                                           ArrowSchema* schema,
                                           ArrowDeviceArray* output,
                                           cudaStream_t stream);

CUDF_C_EXPORT cudfError_t cudfColumnFromArrow(ArrowSchema* schema,
                                              ArrowDeviceArray* input,
                                              cudaStream_t stream,
                                              cudfColumn_t* col);

CUDF_C_EXPORT cudfError_t cudfColumnToArrow(cudfColumn_t col,
                                            ArrowSchema* schema,
                                            ArrowDeviceArray* output,
                                            cudaStream_t stream);

CUDF_C_EXPORT cudfError_t cudfTableFromArrowHost(ArrowSchema* schema,
                                                 ArrowArray* input,
                                                 cudaStream_t stream,
                                                 cudfTable_t* table);

CUDF_C_EXPORT cudfError_t cudfTableToArrowHost(cudfTable_t table,
                                               ArrowSchema* schema,
                                               ArrowArray* output,
                                               cudaStream_t stream);

#ifdef __cplusplus
}
#endif
