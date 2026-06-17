/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  CUDF_TYPE_EMPTY                  = 0,
  CUDF_TYPE_INT8                   = 1,
  CUDF_TYPE_INT16                  = 2,
  CUDF_TYPE_INT32                  = 3,
  CUDF_TYPE_INT64                  = 4,
  CUDF_TYPE_UINT8                  = 5,
  CUDF_TYPE_UINT16                 = 6,
  CUDF_TYPE_UINT32                 = 7,
  CUDF_TYPE_UINT64                 = 8,
  CUDF_TYPE_FLOAT32                = 9,
  CUDF_TYPE_FLOAT64                = 10,
  CUDF_TYPE_BOOL8                  = 11,
  CUDF_TYPE_TIMESTAMP_DAYS         = 12,
  CUDF_TYPE_TIMESTAMP_SECONDS      = 13,
  CUDF_TYPE_TIMESTAMP_MILLISECONDS = 14,
  CUDF_TYPE_TIMESTAMP_MICROSECONDS = 15,
  CUDF_TYPE_TIMESTAMP_NANOSECONDS  = 16,
  CUDF_TYPE_DURATION_DAYS          = 17,
  CUDF_TYPE_DURATION_SECONDS       = 18,
  CUDF_TYPE_DURATION_MILLISECONDS  = 19,
  CUDF_TYPE_DURATION_MICROSECONDS  = 20,
  CUDF_TYPE_DURATION_NANOSECONDS   = 21,
  CUDF_TYPE_DICTIONARY32           = 22,
  CUDF_TYPE_STRING                 = 23,
  CUDF_TYPE_LIST                   = 24,
  CUDF_TYPE_DECIMAL32              = 25,
  CUDF_TYPE_DECIMAL64              = 26,
  CUDF_TYPE_DECIMAL128             = 27,
  CUDF_TYPE_STRUCT                 = 28
} cudfTypeId_t;

typedef struct {
  cudfTypeId_t id;
  int32_t scale;
} cudfDataType_t;

#ifdef __cplusplus
}
#endif
