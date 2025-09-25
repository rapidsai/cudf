/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cudf/context.hpp>
#include <cudf_test/base_fixture.hpp>

#include <gtest/gtest.h>

struct ContextTest : public cudf::test::BaseFixture {
  ~ContextTest() override {
    // Automatically deinitialize on test completion
    try {
      cudf::deinitialize();
    } catch (...) {
      // Ignore exceptions during cleanup
    }
  }
};

TEST_F(ContextTest, MultipleInitializeCalls)
{
  // First initialization with JIT cache only
  cudf::initialize(cudf::init_flags::INIT_JIT_CACHE);
  
  // Second initialization with nvCOMP - should not throw
  EXPECT_NO_THROW(cudf::initialize(cudf::init_flags::LOAD_NVCOMP));
  
  // Third initialization with all flags - should not throw and not reinitialize
  EXPECT_NO_THROW(cudf::initialize(cudf::init_flags::ALL));
}

TEST_F(ContextTest, InitializeAfterDeinitialize)
{
  cudf::initialize(cudf::init_flags::ALL);
  cudf::deinitialize();
  
  // Should be able to initialize again
  EXPECT_NO_THROW(cudf::initialize(cudf::init_flags::INIT_JIT_CACHE));
}

TEST_F(ContextTest, MultipleDeinitializeCalls)
{
  cudf::initialize(cudf::init_flags::ALL);
  cudf::deinitialize();
  
  EXPECT_NO_THROW(cudf::deinitialize());
}
