/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/column/column.hpp>
#include <cudf/context.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>

#include <gtest/gtest.h>

#include <barrier>
#include <future>
#include <vector>

struct ContextTest : public cudf::test::BaseFixture {};

namespace {

cudf::size_type compute_column_jit()
{
  auto c_0        = cudf::test::fixed_width_column_wrapper<cudf::size_type>{3, 20, 1, 50};
  auto c_1        = cudf::test::fixed_width_column_wrapper<cudf::size_type>{10, 7, 20, 0};
  auto table      = cudf::table_view{{c_0, c_1}};
  auto col_ref_0  = cudf::ast::column_reference(0);
  auto col_ref_1  = cudf::ast::column_reference(1);
  auto expression = cudf::ast::operation(cudf::ast::ast_operator::ADD, col_ref_0, col_ref_1);

  return cudf::compute_column_jit(table, expression)->size();
}

}  // namespace

TEST_F(ContextTest, MultipleInitializeCalls)
{
  cudf::detail::initialize(cudf::detail::init_flags::DEFAULT);

  EXPECT_NO_THROW(cudf::detail::initialize(cudf::detail::init_flags::LOAD_NVCOMP));
  EXPECT_NO_THROW(cudf::detail::initialize(cudf::detail::init_flags::ALL));
}

TEST_F(ContextTest, ConcurrentFirstJitCacheUse)
{
  constexpr auto num_threads = 4;
  std::barrier start{num_threads};
  std::vector<std::future<cudf::size_type>> results;
  results.reserve(num_threads);

  for (auto i = 0; i < num_threads; ++i) {
    results.push_back(std::async(std::launch::async, [&] {
      start.arrive_and_wait();
      return compute_column_jit();
    }));
  }

  for (auto& result : results) {
    EXPECT_EQ(result.get(), cudf::size_type{4});
  }
}

TEST_F(ContextTest, JitCacheUse)
{
  cudf::detail::initialize(cudf::detail::init_flags::DEFAULT);
  EXPECT_EQ(compute_column_jit(), cudf::size_type{4});

  cudf::detail::initialize(cudf::detail::init_flags::DEFAULT);
  EXPECT_EQ(compute_column_jit(), cudf::size_type{4});
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
      EXPECT_NO_THROW(cudf::detail::initialize(cudf::detail::init_flags::NONE));
    } else if (role == 1) {
      EXPECT_NO_THROW(cudf::detail::initialize(cudf::detail::init_flags::LOAD_NVCOMP));
    } else {
      EXPECT_NO_THROW(cudf::detail::initialize(cudf::detail::init_flags::ALL));
    }
  };
  EXPECT_NO_FATAL_FAILURE(run_multithreaded(init_task));
}
