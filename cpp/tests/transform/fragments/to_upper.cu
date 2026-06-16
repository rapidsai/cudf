

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda/std/cstdint>

extern "C" __device__ int transform(uint8_t* output, uint8_t input)
{
  if (input > 96 && input < 123) {
    *output = input - 32;
  } else {
    *output = input;
  }
  return 0;
}
