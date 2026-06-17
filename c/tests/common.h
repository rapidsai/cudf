/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CUDF_TEST_COMMON_H
#define CUDF_TEST_COMMON_H

#include <cudf/core/c_api.h>

#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDF(call)                                                                        \
  do {                                                                                          \
    cudfError_t err = (call);                                                                   \
    if (err != CUDF_SUCCESS) {                                                                  \
      fprintf(stderr, "CUDF error at %s:%d: %s\n", __FILE__, __LINE__, cudfGetLastErrorText()); \
      exit(1);                                                                                  \
    }                                                                                           \
  } while (0)

#endif /* CUDF_TEST_COMMON_H */
