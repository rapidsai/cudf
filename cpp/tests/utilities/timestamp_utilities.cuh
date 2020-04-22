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

#include <cudf/wrappers/timestamps.hpp>
#include <tests/utilities/column_wrapper.hpp>

#include <thrust/device_vector.h>
#include <thrust/logical.h>
#include <thrust/sequence.h>

#include <rmm/thrust_rmm_allocator.h>

namespace cudf {
namespace test {

using time_point_ms =
    simt::std::chrono::time_point<simt::std::chrono::system_clock,
                                  simt::std::chrono::milliseconds>;

/**---------------------------------------------------------------------------*
 * @brief Creates a `thrust::device_vector` with ascending timestamps in the
 * range `[start, stop)`.
 *
 * The period is inferred from `count` and difference between `start`
 * and `stop`.
 *
 * @tparam Rep The arithmetic type representing the number of ticks
 * @tparam Period A simt::std::ratio representing the tick period (i.e. the
 *number of seconds per tick)
 * @param count The number of timestamps to create
 * @param start The first timestamp as a simt::std::chrono::time_point
 * @param stop The last timestamp as a simt::std::chrono::time_point
 *---------------------------------------------------------------------------**/
template <typename T, bool nullable = false>
inline cudf::test::fixed_width_column_wrapper<T>
generate_timestamps(int32_t count, time_point_ms start, time_point_ms stop) {
  using Rep = typename T::rep;
  using Period = typename T::period;
  using ToDuration = simt::std::chrono::duration<Rep, Period>;

  auto lhs = start.time_since_epoch().count();
  auto rhs = stop.time_since_epoch().count();

  // When C++17, auto [min, max] = std::minmax(lhs, rhs)
  auto min = std::min(lhs, rhs);
  auto max = std::max(lhs, rhs);
  auto range = max - min;
  auto iter = cudf::test::make_counting_transform_iterator(0, [=](auto i) {
    return simt::std::chrono::floor<ToDuration>(
      simt::std::chrono::milliseconds(min + (range / count) * i)).count();
  });

  if (nullable) {
    auto mask = cudf::test::make_counting_transform_iterator(0, [](auto i) { return i % 2 == 0; });
    return cudf::test::fixed_width_column_wrapper<T>(iter, iter + count, mask);
  }
  else {
    // This needs to be in an else to quash `statement_not_reachable` warnings
    return cudf::test::fixed_width_column_wrapper<T>(iter, iter + count);
  }
}

}  // namespace test
}  // namespace cudf
