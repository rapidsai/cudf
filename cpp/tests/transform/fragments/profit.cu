/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

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
  // Input schema:
  //   0: extended_price(double),
  //   1: discount(float),
  //   2: tax(float),
  //   3: ship_date(int32 YYYYMMDD)
  auto extended_price = load<double>(inputs, input_stride, 0);
  auto discount       = load<float>(inputs, input_stride, 1);
  auto tax            = load<float>(inputs, input_stride, 2);
  auto ship_date      = load<int>(inputs, input_stride, 3);

  // Parameters:
  //   4: ship_date_cutoff(int32 YYYYMMDD)
  auto ship_date_cutoff = load<int>(inputs, input_stride, 4);

  auto base_price = extended_price;
  auto disc_price = extended_price * (1.0 - discount);
  auto charge     = disc_price * (1.0 + tax);

  auto before_cutoff = ship_date <= ship_date_cutoff;

  store(outputs, output_stride, 0, base_price);
  store(outputs, output_stride, 1, charge);
  store(outputs, output_stride, 2, disc_price);
  store(outputs, output_stride, 3, before_cutoff);

  return 0;
}
