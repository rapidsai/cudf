/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

extern "C" __device__ int transform(float* out, float a)
{
  *out = 1.0F / sqrtf(a);
  return 0;
}
