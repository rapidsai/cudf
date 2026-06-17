/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/core/c_api.h>
#include <cudf/core/export.h>
#include <cudf/table.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { CUDF_JOIN_INNER = 0, CUDF_JOIN_LEFT, CUDF_JOIN_FULL } cudfJoinType_t;

CUDF_C_EXPORT cudfError_t cudfTableJoin(cudfTable_t left,
                                        cudfTable_t right,
                                        const int32_t* left_on,
                                        const int32_t* right_on,
                                        int32_t num_keys,
                                        cudfJoinType_t join_type,
                                        cudaStream_t stream,
                                        cudfTable_t* result);

#ifdef __cplusplus
}
#endif
