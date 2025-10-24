/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf_test/column_wrapper.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/utilities/export.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <thrust/logical.h>
#include <thrust/sequence.h>

namespace CUDF_EXPORT cudf {
namespace test {
using time_point_ms =
  cuda::std::chrono::time_point<cuda::std::chrono::system_clock, cuda::std::chrono::milliseconds>;

/**
 * @brief Creates a `fixed_width_column_wrapper` with ascending timestamps in the
 * range `[start, stop)`.
 *
 * The period is inferred from `count` and difference between `start`
 * and `stop`.
 *
 * @tparam Rep The arithmetic type representing the number of ticks
 * @tparam Period A cuda::std::ratio representing the tick period (i.e. the
 *number of seconds per tick)
 * @param count The number of timestamps to create
 * @param start The first timestamp as a cuda::std::chrono::time_point
 * @param stop The last timestamp as a cuda::std::chrono::time_point
 */
template <typename T, bool nullable = false>
inline cudf::test::fixed_width_column_wrapper<T, int64_t> generate_timestamps(int32_t count,
                                                                              time_point_ms start,
                                                                              time_point_ms stop)
{
  using Rep        = typename T::rep;
  using Period     = typename T::period;
  using ToDuration = cuda::std::chrono::duration<Rep, Period>;

  auto lhs = start.time_since_epoch().count();
  auto rhs = stop.time_since_epoch().count();

  auto const min   = std::min(lhs, rhs);
  auto const max   = std::max(lhs, rhs);
  auto const range = max - min;
  auto iter        = cudf::detail::make_counting_transform_iterator(0, [=](auto i) {
    return cuda::std::chrono::floor<ToDuration>(
             cuda::std::chrono::milliseconds(min + (range / count) * i))
      .count();
  });

  if (nullable) {
    auto mask =
      cudf::detail::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0; });
    return cudf::test::fixed_width_column_wrapper<T, int64_t>(iter, iter + count, mask);
  } else {
    // This needs to be in an else to quash `statement_not_reachable` warnings
    return cudf::test::fixed_width_column_wrapper<T, int64_t>(iter, iter + count);
  }
}

}  // namespace test
}  // namespace CUDF_EXPORT cudf
