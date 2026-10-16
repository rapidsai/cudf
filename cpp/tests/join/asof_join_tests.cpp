/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/copying.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/join/asof_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <array>
#include <atomic>
#include <exception>
#include <limits>
#include <memory>
#include <thread>
#include <vector>

namespace {

using cudf::experimental::asof_join;
using cudf::experimental::asof_join_exact_matches;
using cudf::experimental::asof_join_strategy;
using cudf::test::fixed_width_column_wrapper;
using cudf::test::strings_column_wrapper;

class AsofJoinTest : public cudf::test::BaseFixture {};

void expect_gather_map(rmm::device_uvector<cudf::size_type> const& actual,
                       std::vector<cudf::size_type> const& expected)
{
  auto const host_actual = cudf::detail::make_std_vector(actual, cudf::get_default_stream());
  EXPECT_EQ(host_actual, expected);
}

template <typename T>
class AsofJoinOrderedKeyTest : public cudf::test::BaseFixture {};

using OrderedKeyTypes =
  cudf::test::Concat<cudf::test::IntegralTypesNotBool, cudf::test::ChronoTypes>;
TYPED_TEST_SUITE(AsofJoinOrderedKeyTest, OrderedKeyTypes);

TYPED_TEST(AsofJoinOrderedKeyTest, Backward)
{
  fixed_width_column_wrapper<TypeParam, int32_t> right_on{1, 3, 3, 7};
  fixed_width_column_wrapper<TypeParam, int32_t> left_on{0, 1, 2, 3, 8};

  asof_join join(cudf::table_view{}, right_on);
  auto result = join.join(
    cudf::table_view{}, left_on, asof_join_strategy::BACKWARD, asof_join_exact_matches::YES);

  expect_gather_map(*result, {cudf::JoinNoMatch, 0, 0, 2, 3});
}

TEST_F(AsofJoinTest, FloatingPointGroupingKeys)
{
  auto const nan = std::numeric_limits<float>::quiet_NaN();
  auto const inf = std::numeric_limits<float>::infinity();
  fixed_width_column_wrapper<float> right_by({0.0f, 1.0f, 1.0f, inf, nan, nan},
                                             cudf::test::iterators::nulls_at({0}));
  fixed_width_column_wrapper<int32_t> right_on{1, 1, 4, 2, 1, 5};
  fixed_width_column_wrapper<float> left_by({nan, 1.0f, nan, 0.0f, 2.0f, inf},
                                            cudf::test::iterators::nulls_at({3}));
  fixed_width_column_wrapper<int32_t> left_on{5, 3, 0, 9, 9, 2};

  asof_join join(cudf::table_view{{right_by}}, right_on);
  auto result = join.join(cudf::table_view{{left_by}},
                          left_on,
                          asof_join_strategy::BACKWARD,
                          asof_join_exact_matches::YES);

  expect_gather_map(*result, {5, 1, cudf::JoinNoMatch, cudf::JoinNoMatch, cudf::JoinNoMatch, 3});
}

TEST_F(AsofJoinTest, DecimalGroupingKeys)
{
  cudf::test::fixed_point_column_wrapper<int32_t> right_by{{120, 120, 250},
                                                           numeric::scale_type{-2}};
  fixed_width_column_wrapper<int32_t> right_on{1, 4, 2};
  cudf::test::fixed_point_column_wrapper<int32_t> left_by{{250, 120, 175, 120},
                                                          numeric::scale_type{-2}};
  fixed_width_column_wrapper<int32_t> left_on{3, 4, 9, 0};

  asof_join join(cudf::table_view{{right_by}}, right_on);
  auto result = join.join(cudf::table_view{{left_by}},
                          left_on,
                          asof_join_strategy::BACKWARD,
                          asof_join_exact_matches::YES);

  expect_gather_map(*result, {2, 1, cudf::JoinNoMatch, cudf::JoinNoMatch});
}

TEST_F(AsofJoinTest, Ungrouped)
{
  fixed_width_column_wrapper<int32_t> right_on{1, 3, 3, 7};
  fixed_width_column_wrapper<int32_t> left_on{0, 1, 2, 3, 8};

  asof_join join(cudf::table_view{}, right_on);
  auto result = join.join(
    cudf::table_view{}, left_on, asof_join_strategy::BACKWARD, asof_join_exact_matches::YES);

  expect_gather_map(*result, {cudf::JoinNoMatch, 0, 0, 2, 3});
}

TEST_F(AsofJoinTest, SingleStringGroupingKey)
{
  strings_column_wrapper right_by{"a", "a", "b", "b"};
  fixed_width_column_wrapper<int32_t> right_on{1, 3, 2, 4};
  strings_column_wrapper left_by{"b", "a", "c", "a", "b"};
  fixed_width_column_wrapper<int32_t> left_on{3, 0, 5, 3, 5};

  asof_join join(cudf::table_view{{right_by}}, right_on);
  auto result = join.join(cudf::table_view{{left_by}},
                          left_on,
                          asof_join_strategy::BACKWARD,
                          asof_join_exact_matches::YES);

  expect_gather_map(*result, {2, cudf::JoinNoMatch, cudf::JoinNoMatch, 1, 3});
}

TEST_F(AsofJoinTest, MultipleGroupingKeys)
{
  strings_column_wrapper right_by_1{"a", "a", "a", "b", "b"};
  fixed_width_column_wrapper<int32_t> right_by_2{1, 1, 2, 1, 1};
  fixed_width_column_wrapper<int32_t> right_on{1, 4, 2, 3, 5};
  strings_column_wrapper left_by_1{"b", "a", "a"};
  fixed_width_column_wrapper<int32_t> left_by_2{1, 2, 1};
  fixed_width_column_wrapper<int32_t> left_on{4, 1, 4};

  asof_join join(cudf::table_view{{right_by_1, right_by_2}}, right_on);
  auto result = join.join(cudf::table_view{{left_by_1, left_by_2}},
                          left_on,
                          asof_join_strategy::BACKWARD,
                          asof_join_exact_matches::YES);

  expect_gather_map(*result, {3, cudf::JoinNoMatch, 1});
}

TEST_F(AsofJoinTest, SequentialProbes)
{
  strings_column_wrapper right_by{"a", "a", "b", "b"};
  fixed_width_column_wrapper<int32_t> right_on{1, 4, 2, 6};
  strings_column_wrapper left_by_1{"a", "b", "c"};
  fixed_width_column_wrapper<int32_t> left_on_1{4, 5, 7};
  strings_column_wrapper left_by_2{"b", "a", "a"};
  fixed_width_column_wrapper<int32_t> left_on_2{6, 0, 3};

  asof_join join(cudf::table_view{{right_by}}, right_on);
  auto result_1 = join.join(cudf::table_view{{left_by_1}},
                            left_on_1,
                            asof_join_strategy::BACKWARD,
                            asof_join_exact_matches::YES);
  auto result_2 = join.join(cudf::table_view{{left_by_2}},
                            left_on_2,
                            asof_join_strategy::BACKWARD,
                            asof_join_exact_matches::YES);

  expect_gather_map(*result_1, {1, 2, cudf::JoinNoMatch});
  expect_gather_map(*result_2, {3, cudf::JoinNoMatch, 0});
}

TEST_F(AsofJoinTest, ConcurrentMultiStreamProbes)
{
  constexpr std::size_t num_probes = 4;
  fixed_width_column_wrapper<int32_t> right_by{0, 0, 1, 1, 2, 2};
  fixed_width_column_wrapper<int32_t> right_on{1, 4, 2, 6, 3, 8};

  fixed_width_column_wrapper<int32_t> left_by_0{0, 1, 2};
  fixed_width_column_wrapper<int32_t> left_on_0{4, 5, 9};
  fixed_width_column_wrapper<int32_t> left_by_1{2, 0, 3};
  fixed_width_column_wrapper<int32_t> left_on_1{2, 2, 7};
  fixed_width_column_wrapper<int32_t> left_by_2{1, 2, 0};
  fixed_width_column_wrapper<int32_t> left_on_2{6, 8, 0};
  fixed_width_column_wrapper<int32_t> left_by_3{0, 2, 1};
  fixed_width_column_wrapper<int32_t> left_on_3{3, 4, 1};

  auto const left_by =
    std::array<cudf::column_view, num_probes>{left_by_0, left_by_1, left_by_2, left_by_3};
  auto const left_on =
    std::array<cudf::column_view, num_probes>{left_on_0, left_on_1, left_on_2, left_on_3};
  auto const expected = std::array<std::vector<cudf::size_type>, num_probes>{
    std::vector<cudf::size_type>{1, 2, 5},
    std::vector<cudf::size_type>{cudf::JoinNoMatch, 0, cudf::JoinNoMatch},
    std::vector<cudf::size_type>{3, 5, cudf::JoinNoMatch},
    std::vector<cudf::size_type>{0, 4, cudf::JoinNoMatch}};

  int device{};
  CUDF_CUDA_TRY(cudaGetDevice(&device));
  auto const stream_device = cuda::device_ref{device};
  std::vector<std::unique_ptr<cuda::stream>> streams;
  streams.reserve(num_probes);
  for (std::size_t i = 0; i < num_probes; ++i) {
    streams.push_back(std::make_unique<cuda::stream>(stream_device));
  }

  // Construct on a stream of its own so the probes below run on different streams than the
  // group index was built on, then synchronize it as the `join()` docs require.
  cuda::stream const build_stream{stream_device};
  asof_join join(cudf::table_view{{right_by}}, right_on, build_stream);
  build_stream.sync();

  std::vector<std::unique_ptr<rmm::device_uvector<cudf::size_type>>> results(num_probes);
  std::vector<std::thread> threads;
  std::vector<std::exception_ptr> errors(num_probes);
  // `ready` lets the main thread wait until every worker is spawned and spinning, and `start`
  // then releases them together, so the probes actually overlap instead of trickling in as
  // each thread is created.
  std::atomic<std::size_t> ready{0};
  std::atomic<bool> start{false};
  threads.reserve(num_probes);
  for (std::size_t i = 0; i < num_probes; ++i) {
    threads.emplace_back([&, i] {
      CUDF_CUDA_TRY(cudaSetDevice(device));
      ready.fetch_add(1, std::memory_order_relaxed);
      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }
      try {
        results[i] = join.join(cudf::table_view{{left_by[i]}},
                               left_on[i],
                               asof_join_strategy::BACKWARD,
                               asof_join_exact_matches::YES,
                               *streams[i]);
      } catch (...) {
        errors[i] = std::current_exception();
      }
    });
  }
  while (ready.load(std::memory_order_relaxed) != num_probes) {
    std::this_thread::yield();
  }
  start.store(true, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }
  for (auto const& probe_stream : streams) {
    probe_stream->sync();
  }
  for (auto const& error : errors) {
    ASSERT_FALSE(error);
  }
  for (std::size_t i = 0; i < num_probes; ++i) {
    expect_gather_map(*results[i], expected[i]);
  }
}

TEST_F(AsofJoinTest, NullOrderedKeys)
{
  fixed_width_column_wrapper<int32_t> right_on({0, 1, 3}, cudf::test::iterators::nulls_at({0}));
  fixed_width_column_wrapper<int32_t> left_on({0, 1, 2, 3}, cudf::test::iterators::nulls_at({3}));

  asof_join join(cudf::table_view{}, right_on);
  auto result = join.join(
    cudf::table_view{}, left_on, asof_join_strategy::BACKWARD, asof_join_exact_matches::YES);

  expect_gather_map(*result, {cudf::JoinNoMatch, 1, 1, cudf::JoinNoMatch});
}

TEST_F(AsofJoinTest, NullGroupingKeysDoNotMatch)
{
  strings_column_wrapper right_by({"", "", "a"}, cudf::test::iterators::nulls_at({0, 1}));
  fixed_width_column_wrapper<int32_t> right_on{1, 2, 1};
  strings_column_wrapper left_by({"", "a"}, cudf::test::iterators::nulls_at({0}));
  fixed_width_column_wrapper<int32_t> left_on{2, 1};

  asof_join join(cudf::table_view{{right_by}}, right_on);
  auto result = join.join(cudf::table_view{{left_by}},
                          left_on,
                          asof_join_strategy::BACKWARD,
                          asof_join_exact_matches::YES);

  expect_gather_map(*result, {cudf::JoinNoMatch, 2});
}

TEST_F(AsofJoinTest, EmptyInputs)
{
  fixed_width_column_wrapper<int32_t> empty{};
  fixed_width_column_wrapper<int32_t> left_on{1, 2};

  asof_join empty_right_join(cudf::table_view{}, empty);
  auto no_matches = empty_right_join.join(
    cudf::table_view{}, left_on, asof_join_strategy::BACKWARD, asof_join_exact_matches::YES);
  expect_gather_map(*no_matches, {cudf::JoinNoMatch, cudf::JoinNoMatch});

  asof_join join(cudf::table_view{}, left_on);
  auto empty_result = join.join(
    cudf::table_view{}, empty, asof_join_strategy::BACKWARD, asof_join_exact_matches::YES);
  expect_gather_map(*empty_result, {});
}

TEST_F(AsofJoinTest, TimestampOrderedKey)
{
  fixed_width_column_wrapper<cudf::timestamp_ms, int64_t> right_on{1, 3, 5};
  fixed_width_column_wrapper<cudf::timestamp_ms, int64_t> left_on{2, 5};

  asof_join join(cudf::table_view{}, right_on);
  auto result = join.join(
    cudf::table_view{}, left_on, asof_join_strategy::BACKWARD, asof_join_exact_matches::YES);

  expect_gather_map(*result, {0, 2});
}

TEST_F(AsofJoinTest, UnsupportedOptions)
{
  fixed_width_column_wrapper<int32_t> right_on{1};
  fixed_width_column_wrapper<int32_t> left_on{1};
  asof_join join(cudf::table_view{}, right_on);

  EXPECT_THROW(
    (void)join.join(
      cudf::table_view{}, left_on, asof_join_strategy::BACKWARD, asof_join_exact_matches::NO),
    cudf::logic_error);
}

TEST_F(AsofJoinTest, MismatchedOrderedKeyTypes)
{
  fixed_width_column_wrapper<int32_t> right_on{1};
  fixed_width_column_wrapper<int64_t> left_on{1};
  asof_join join(cudf::table_view{}, right_on);

  EXPECT_THROW(
    (void)join.join(
      cudf::table_view{}, left_on, asof_join_strategy::BACKWARD, asof_join_exact_matches::YES),
    cudf::logic_error);
}

TEST_F(AsofJoinTest, NullGroupBeforeNonNullGroup)
{
  strings_column_wrapper right_by({"", "a", "a"}, cudf::test::iterators::nulls_at({0}));
  fixed_width_column_wrapper<int32_t> right_on{1, 1, 2};
  strings_column_wrapper left_by({"", "a"}, cudf::test::iterators::nulls_at({0}));
  fixed_width_column_wrapper<int32_t> left_on{5, 1};

  asof_join join(cudf::table_view{{right_by}}, right_on);
  auto result = join.join(cudf::table_view{{left_by}},
                          left_on,
                          asof_join_strategy::BACKWARD,
                          asof_join_exact_matches::YES);

  expect_gather_map(*result, {cudf::JoinNoMatch, 1});
}

TEST_F(AsofJoinTest, LeftKeyBetweenGroups)
{
  strings_column_wrapper right_by{"a", "c"};
  fixed_width_column_wrapper<int32_t> right_on{1, 1};
  strings_column_wrapper left_by{"b", "0", "d"};
  fixed_width_column_wrapper<int32_t> left_on{1, 1, 1};

  asof_join join(cudf::table_view{{right_by}}, right_on);
  auto result = join.join(cudf::table_view{{left_by}},
                          left_on,
                          asof_join_strategy::BACKWARD,
                          asof_join_exact_matches::YES);

  expect_gather_map(*result, {cudf::JoinNoMatch, cudf::JoinNoMatch, cudf::JoinNoMatch});
}

TEST_F(AsofJoinTest, NullOrderedKeyAtGroupFront)
{
  strings_column_wrapper right_by{"a", "a", "b"};
  fixed_width_column_wrapper<int32_t> right_on({0, 5, 3}, cudf::test::iterators::nulls_at({0}));
  strings_column_wrapper left_by{"a", "a"};
  fixed_width_column_wrapper<int32_t> left_on{4, 5};

  asof_join join(cudf::table_view{{right_by}}, right_on);
  auto result = join.join(cudf::table_view{{left_by}},
                          left_on,
                          asof_join_strategy::BACKWARD,
                          asof_join_exact_matches::YES);

  expect_gather_map(*result, {cudf::JoinNoMatch, 1});
}

TEST_F(AsofJoinTest, SlicedColumns)
{
  fixed_width_column_wrapper<int32_t> right_on_base{9, 1, 3, 3, 7, 9};
  fixed_width_column_wrapper<int32_t> left_on_base{9, 0, 1, 2, 3, 8};
  auto const right_on = cudf::slice(right_on_base, {1, 5})[0];
  auto const left_on  = cudf::slice(left_on_base, {1, 5})[0];

  asof_join join(cudf::table_view{}, right_on);
  auto result = join.join(
    cudf::table_view{}, left_on, asof_join_strategy::BACKWARD, asof_join_exact_matches::YES);

  expect_gather_map(*result, {cudf::JoinNoMatch, 0, 0, 2});
}

TEST_F(AsofJoinTest, SlicedGroupedColumns)
{
  strings_column_wrapper right_by_base{"z", "a", "a", "b", "b", "z"};
  fixed_width_column_wrapper<int32_t> right_on_base{9, 1, 3, 2, 4, 9};
  auto const right_by = cudf::slice(right_by_base, {1, 5})[0];
  auto const right_on = cudf::slice(right_on_base, {1, 5})[0];
  strings_column_wrapper left_by_base{"z", "b", "a", "z"};
  fixed_width_column_wrapper<int32_t> left_on_base{9, 3, 3, 9};
  auto const left_by = cudf::slice(left_by_base, {1, 3})[0];
  auto const left_on = cudf::slice(left_on_base, {1, 3})[0];

  asof_join join(cudf::table_view{{right_by}}, right_on);
  auto result = join.join(cudf::table_view{{left_by}},
                          left_on,
                          asof_join_strategy::BACKWARD,
                          asof_join_exact_matches::YES);

  expect_gather_map(*result, {2, 1});
}

TEST_F(AsofJoinTest, ValidationErrors)
{
  strings_column_wrapper right_by{"a", "b"};
  fixed_width_column_wrapper<int32_t> right_on{1, 2};
  asof_join join(cudf::table_view{{right_by}}, right_on);

  // Left/right number of grouping keys must match
  fixed_width_column_wrapper<int32_t> left_on{1};
  EXPECT_THROW(
    (void)join.join(
      cudf::table_view{}, left_on, asof_join_strategy::BACKWARD, asof_join_exact_matches::YES),
    cudf::logic_error);

  // Grouping and ordered keys must have the same number of rows
  strings_column_wrapper short_by{"a"};
  EXPECT_THROW((void)asof_join(cudf::table_view{{short_by}}, right_on), cudf::logic_error);
  strings_column_wrapper left_by{"a", "b"};
  fixed_width_column_wrapper<int32_t> long_left_on{1, 2, 3};
  EXPECT_THROW((void)join.join(cudf::table_view{{left_by}},
                               long_left_on,
                               asof_join_strategy::BACKWARD,
                               asof_join_exact_matches::YES),
               cudf::logic_error);

  // Unsupported ordered-key type
  strings_column_wrapper strings_on{"a", "b"};
  EXPECT_THROW((void)asof_join(cudf::table_view{}, strings_on), cudf::logic_error);

  // Unsupported grouping-key type
  auto lists_by = cudf::test::lists_column_wrapper<int32_t>{{1, 2}, {3}};
  EXPECT_THROW((void)asof_join(cudf::table_view{{lists_by}}, right_on), cudf::logic_error);
}

TEST_F(AsofJoinTest, LargeMultiBlock)
{
  // 2000 contiguous groups of 1000 rows each, ordered keys ascending within each group.
  // Exercises comparator lifetimes and dispatch across many thread blocks.
  auto constexpr num_rows   = 2'000'000;
  auto constexpr group_size = 1000;
  std::vector<int32_t> right_by_host(num_rows);
  std::vector<int32_t> right_on_host(num_rows);
  for (int32_t i = 0; i < num_rows; ++i) {
    right_by_host[i] = i / group_size;
    right_on_host[i] = i % group_size;
  }
  fixed_width_column_wrapper<int32_t> right_by(right_by_host.begin(), right_by_host.end());
  fixed_width_column_wrapper<int32_t> right_on(right_on_host.begin(), right_on_host.end());

  auto constexpr num_groups = num_rows / group_size;
  std::vector<int32_t> left_by_host(num_groups);
  std::vector<int32_t> left_on_host(num_groups, group_size / 2);
  for (int32_t g = 0; g < num_groups; ++g) {
    left_by_host[g] = g;
  }
  fixed_width_column_wrapper<int32_t> left_by(left_by_host.begin(), left_by_host.end());
  fixed_width_column_wrapper<int32_t> left_on(left_on_host.begin(), left_on_host.end());

  asof_join join(cudf::table_view{{right_by}}, right_on);
  auto result = join.join(cudf::table_view{{left_by}},
                          left_on,
                          asof_join_strategy::BACKWARD,
                          asof_join_exact_matches::YES);

  std::vector<cudf::size_type> expected(num_groups);
  for (int32_t g = 0; g < num_groups; ++g) {
    expected[g] = g * group_size + group_size / 2;
  }
  expect_gather_map(*result, expected);
}

}  // namespace
