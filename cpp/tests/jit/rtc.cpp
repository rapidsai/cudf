/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf_test/column_wrapper.hpp"
#include "jit/row_ir.hpp"

#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <cudf/jit/lto/transform_params.cuh>

#include <jit/rtc/cudf.hpp>

#include <chrono>

using namespace cudf;

struct RTCTest : public ::testing::Test {};

TEST_F(RTCTest, CreateFragment)
{

  // TODO: add configuration parameters necessary for testing
  // and profiling cache behaviour
  auto fn = []() {
    auto lib = rtc::compile_and_link_udf("test_fragment",
                                         R"***(
    #include "cudf/jit/lto/transform_params.cuh"
    #include "cudf/jit/lto/operators.cuh"

    // TODO: declare getters and setters required for the specific LTO context of the operator
    // they should use the provided LTO functions; might need a planner

    extern "C" __device__ void transform_operator(cudf::lto::transform_params p){
      int a = 1;
      int b = 2;
      int c = 3;
      double d;
      int * out = (int *)p.scope[0];
      cudf::lto::operators::add(&c, &a, &b);
      cudf::lto::operators::sub(&c, &a, &b);
      cudf::lto::operators::mul(&c, &a, &b);
      cudf::lto::operators::mul(&c, &a, &b);
      cudf::lto::operators::cast_to_float64(&d, &c);
      c = (int)a;
      c = c * c;
      *out = a + b * c;
    }

    )***",
                                         "test_udf_key",
                                         "transform_kernel");

    auto kernel = lib->get_kernel("transform_kernel");

    EXPECT_EQ("transform_kernel", kernel.get_name());
    void** scope     = nullptr;
    int32_t num_rows = 0;
    void* args[]     = {&scope, &num_rows};
    kernel.launch(1, 1, 1, 1, 1, 1, 0, nullptr, args);
  };

  fn();
  fn();
  fn();
}

CUDF_TEST_PROGRAM_MAIN()
