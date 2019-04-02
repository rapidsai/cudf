/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <gtest/gtest.h>

#include <rmm/rmm.h>

#define ASSERT_CUDA_SUCCEEDED(expr) ASSERT_EQ(cudaSuccess, expr)
#define EXPECT_CUDA_SUCCEEDED(expr) EXPECT_EQ(cudaSuccess, expr)

#define ASSERT_RMM_SUCCEEDED(expr)  ASSERT_EQ(RMM_SUCCESS, expr)
#define EXPECT_RMM_SUCCEEDED(expr)  EXPECT_EQ(RMM_SUCCESS, expr)

// Base class fixture for GDF google tests that initializes / finalizes the
// RAPIDS memory manager
struct GdfTest : public ::testing::Test
{
    static void SetUpTestCase() {
        ASSERT_RMM_SUCCEEDED( rmmInitialize(nullptr) );
    }

    static void TearDownTestCase() {
        ASSERT_RMM_SUCCEEDED( rmmFinalize() );
    }
};
