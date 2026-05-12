
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

/**
 * TPC-DS Q7: "Report the profit of each returned item, and the total profit for all returned items,
 * for a given date range."
 *
 */
extern "C" __device__ int cudf_transform_operation(
  void*, long int, void const* inputs, int input_stride, void* outputs, int output_stride)
{
  // Input schema:
  //   0: store_sales(double),
  //   1: catalog_sales(double),
  //   2: web_sales(double),
  //   3: store_returns(double),
  //   4: catalog_returns(double),
  //   5: web_returns(double),
  //   6: profit(double),
  //   7: profit_loss(double)
  auto store_sales     = load<double>(inputs, input_stride, 0);
  auto catalog_sales   = load<double>(inputs, input_stride, 1);
  auto web_sales       = load<double>(inputs, input_stride, 2);
  auto store_returns   = load<double>(inputs, input_stride, 3);
  auto catalog_returns = load<double>(inputs, input_stride, 4);
  auto web_returns     = load<double>(inputs, input_stride, 5);
  auto profit          = load<double>(inputs, input_stride, 6);
  auto profit_loss     = load<double>(inputs, input_stride, 7);

  double sales      = store_sales + catalog_sales + web_sales;
  double returns    = store_returns + catalog_returns + web_returns;
  double net_sales  = sales - returns;
  double net_profit = profit - profit_loss;

  store(outputs, output_stride, 0, net_profit);
  store(outputs, output_stride, 1, net_sales);
  return 0;
}
