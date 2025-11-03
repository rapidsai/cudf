/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>

#include <cudf/context.hpp>

#include <gtest/gtest.h>

struct ContextTest : public cudf::test::BaseFixture {
  ~ContextTest() override
  {
    try {
      cudf::deinitialize();
    } catch (...) {
    }
  }
};

TEST_F(ContextTest, MultipleInitializeCalls)
{
  cudf::initialize(cudf::init_flags::INIT_JIT_CACHE);

  EXPECT_NO_THROW(cudf::initialize(cudf::init_flags::LOAD_NVCOMP));
  EXPECT_NO_THROW(cudf::initialize(cudf::init_flags::ALL));
}

TEST_F(ContextTest, InitializeAfterDeinitialize)
{
  cudf::initialize(cudf::init_flags::ALL);
  cudf::deinitialize();

  EXPECT_NO_THROW(cudf::initialize(cudf::init_flags::INIT_JIT_CACHE));
}

TEST_F(ContextTest, DeinitializeWithoutInitialize) { EXPECT_NO_THROW(cudf::deinitialize()); }

TEST_F(ContextTest, MultipleDeinitializeCalls)
{
  cudf::initialize(cudf::init_flags::ALL);
  cudf::deinitialize();

  EXPECT_NO_THROW(cudf::deinitialize());
}
