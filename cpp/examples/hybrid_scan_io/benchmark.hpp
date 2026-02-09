/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "timer.hpp"

#include <iostream>

#pragma once

template <std::invocable F>
void benchmark(F&& f, std::size_t iterations)
{
  double total_time_millis{0.0};
  for (std::size_t i = 0; i < iterations; ++i) {
    timer timer;
    timer.reset();

    f();

    auto elapsed_time_ms =
      static_cast<double>(std::chrono::duration_cast<timer::micros>(timer.elapsed()).count()) /
      1000.0;
    std::cout << "Iteration: " << i << ", time: " << elapsed_time_ms << " ms\n";
    if (i != 0) { total_time_millis += elapsed_time_ms; }
  }

  std::cout << "Average time (first iteration excluded): " << total_time_millis / (iterations - 1)
            << " ms\n\n";
}
