/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>

#include <cudf/context.hpp>

#include <gtest/gtest.h>

struct ContextTest : public cudf::test::BaseFixture {
  ~ContextTest() override
  {
    try {
      cudf::teardown();
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

TEST_F(ContextTest, InitializeAfterTeardown)
{
  cudf::initialize(cudf::init_flags::ALL);
  cudf::teardown();

  EXPECT_NO_THROW(cudf::initialize(cudf::init_flags::INIT_JIT_CACHE));
}

TEST_F(ContextTest, TeardownWithoutInitialize) { EXPECT_NO_THROW(cudf::teardown()); }

TEST_F(ContextTest, MultipleTeardownCalls)
{
  cudf::initialize(cudf::init_flags::ALL);
  cudf::teardown();

  EXPECT_NO_THROW(cudf::teardown());
}

template <typename Lambda>
void run_multithreaded(Lambda func)
{
  std::vector<std::thread> threads;
  auto concurrency = std::max(std::thread::hardware_concurrency(), 2U) - 1U;

  for (size_t i = 0; i < concurrency; ++i) {
    threads.emplace_back([i, f = std::move(func)]() { f(i); });
  }

  for (auto& t : threads) {
    t.join();
  }
}

TEST_F(ContextTest, MultipleInitializeCallsMultiThreaded)
{
  auto init_task = [](size_t thread_id) {
    auto role = thread_id % 3;
    if (role == 0) {
      EXPECT_NO_THROW(cudf::initialize(cudf::init_flags::INIT_JIT_CACHE));
    } else if (role == 1) {
      EXPECT_NO_THROW(cudf::initialize(cudf::init_flags::LOAD_NVCOMP));
    } else {
      EXPECT_NO_THROW(cudf::initialize(cudf::init_flags::ALL));
    }
  };
  EXPECT_NO_FATAL_FAILURE(run_multithreaded(init_task));
}
