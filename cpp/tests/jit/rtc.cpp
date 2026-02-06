/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf_test/column_wrapper.hpp"
#include "jit/row_ir.hpp"

#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <jit/rtc/cudf.hpp>

#include <chrono>

using namespace cudf;

struct RTCTest : public ::testing::Test {};

TEST_F(RTCTest, CreateFragment)
{
  auto fn = []() {
    auto begin = std::chrono::high_resolution_clock::now();
    auto kern  = rtc::compile_and_link_udf("test_fragment",
                                          R"***(
    #include "cudf/jit/transform_params.cuh"
    #include "cudf/jit/lto/operators.cuh"

    extern "C" __device__ void transform_operator(cudf::lto::transform_params const* p){
      int a = 1;
      int b = 2;
      int c = 3;
      int * out = (int *)p->outputs;
      cudf::lto::operators::add(&c, &a, &b);
      cudf::lto::operators::sub(&c, &a, &b);
      cudf::lto::operators::mul(&c, &a, &b);
      cudf::lto::operators::mul(&c, &a, &b);
      *out = a + b * c;
    }

    )***",
                                          "test_udf_key",
                                          "transform_kernel");

    (void)kern;
    auto end = std::chrono::high_resolution_clock::now();
    auto dur = end - begin;
    std::cout << "RTC compilation took "
              << std::chrono::duration_cast<std::chrono::microseconds>(dur).count() << " us\n";
  };

  fn();
  fn();
}

CUDF_TEST_PROGRAM_MAIN()
