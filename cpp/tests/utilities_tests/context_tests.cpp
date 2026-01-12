/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>

#include <cudf/context.hpp>

#include <gtest/gtest.h>

struct ContextTest : public cudf::test::BaseFixture {};

TEST_F(ContextTest, MultipleInitializeCalls)
{
  cudf::initialize(cudf::init_flags::INIT_JIT_CACHE);

  EXPECT_THROW(cudf::initialize(cudf::init_flags::LOAD_NVCOMP), std::runtime_error);
  EXPECT_THROW(cudf::initialize(cudf::init_flags::ALL), std::runtime_error);
  EXPECT_NO_THROW(cudf::initialize(cudf::init_flags::INIT_JIT_CACHE));
}
