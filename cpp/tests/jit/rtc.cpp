/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cudf_test/column_wrapper.hpp"
#include "jit/row_ir.hpp"

#include <cudf_test/debug_utilities.hpp>
#include <cudf_test/testing_main.hpp>

#include <jit/rtc/cudf.hpp>

using namespace cudf;

struct RTCTest : public ::testing::Test {};

TEST_F(RTCTest, CreateFragment)
{
  auto kern = rtc::compile_and_link_udf("test_fragment",
                                        "transform_kernel",
                                        "test_kernel_key",
                                        R"***(

    #include "cudf/jit/transform_params.cuh"
    #include "cudf/jit/lto/operators.inl.cuh"
    #include "cudf/jit/lto/types.inl.cuh"

    extern "C" __device__ transform_operation(){
    }

    )***",
                                        "test_udf_key");

  (void)kern;
}

CUDF_TEST_PROGRAM_MAIN()
