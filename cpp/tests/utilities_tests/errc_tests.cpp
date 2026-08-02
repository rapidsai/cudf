/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Simulates an includer that has an unrelated OVERFLOW macro in scope (glibc's math.h
// still defines it as a legacy SVID constant). cudf/errc.hpp must parse and its
// enumerators must remain nameable regardless.
#define OVERFLOW 3

#include <cudf/errc.hpp>

#include <gtest/gtest.h>

static_assert(OVERFLOW == 3, "the includer's macro must be left untouched");
static_assert(static_cast<int>(cudf::errc::ARITHMETIC_OVERFLOW) == 1);

TEST(ErrcTest, EnumeratorsSurviveMacroCollision)
{
  EXPECT_STREQ(cudf::to_string(cudf::errc::ARITHMETIC_OVERFLOW), "ARITHMETIC_OVERFLOW");
  EXPECT_STREQ(cudf::to_string(cudf::errc::SUCCESS), "SUCCESS");
  EXPECT_STREQ(cudf::to_string(cudf::errc::DIVISION_BY_ZERO), "DIVISION_BY_ZERO");
}
