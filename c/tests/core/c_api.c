/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Pure C test for cuDF C API core functions:
 * - cudfVersionGet
 * - cudfResourcesCreate / cudfResourcesDestroy
 * - cudfGetLastErrorText
 */

#include "../common.h"

#include <cudf/core/c_api.h>

#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(void)
{
  /* Test version */
  uint16_t major, minor, patch;
  CHECK_CUDF(cudfVersionGet(&major, &minor, &patch));
  printf("cuDF version: %u.%u.%u\n", major, minor, patch);
  assert(major > 0);

  /* Test resources create/destroy */
  cudfResources_t res;
  CHECK_CUDF(cudfResourcesCreate(&res));
  assert(res != 0);
  CHECK_CUDF(cudfResourcesDestroy(res));

  /* Test error text (after success, should be NULL or empty) */
  const char* err_text = cudfGetLastErrorText();
  printf("Error text after success: %s\n", err_text ? err_text : "(null)");

  printf("c_api tests PASSED\n");
  return 0;
}
