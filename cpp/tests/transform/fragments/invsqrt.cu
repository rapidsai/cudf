/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cstdint>

template <typename T>
__device__ T load(void const* inputs, int input_stride, int arg)
{
  auto p = reinterpret_cast<T const*>(static_cast<char const*>(inputs) + arg * input_stride);
  return *p;
}

template <typename T>
__device__ void store(void* outputs, int output_stride, int arg, T value)
{
  auto p = reinterpret_cast<T*>(static_cast<char*>(outputs) + arg * output_stride);
  *p     = value;
}

extern "C" __device__ int cudf_transform_operation(
  void*, long int, void const* inputs, int input_stride, void* outputs, int output_stride)
{
  auto input  = load<float>(inputs, input_stride, 0);
  auto result = 1.0f / sqrtf(input);
  store(outputs, output_stride, 0, result);
  return 0;
}
