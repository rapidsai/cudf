/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

extern "C" __device__ int transform(float* out, float a, float b)
{
  *out = a * a + b * b;
  return 0;
}
