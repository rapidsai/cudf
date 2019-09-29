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

#include <simt/chrono>

#include <thrust/logical.h>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>

namespace cudf {
namespace test {

using time_point_ms = simt::std::chrono::time_point<
  simt::std::chrono::system_clock,
  simt::std::chrono::milliseconds
>;

template <typename Rep>
struct generate_each_timestamp {
    const Rep base;
    const Rep step;
    generate_each_timestamp(Rep _base, Rep _step) : base(_base), step(_step) {}
    __host__ __device__ Rep operator()(const int32_t& i) const { return base + (step * i); }
};

/**---------------------------------------------------------------------------*
 * @brief Creates a ```thrust::device_vector``` with ascending timesamps in the
 * range ```[start, stop)```.
 * 
 * The period is inferred from ```count``` and difference between ```start```
 * and ```stop```.
 *
 * @param count The number of timestamps to create
 * @param start The first timestamp as a simt::std::chrono::time_point
 * @param stop The last timestamp as a simt::std::chrono::time_point
 *---------------------------------------------------------------------------**/
template <typename Rep, typename Period>
inline thrust::device_vector<Rep> generate_timestamps(int32_t count, time_point_ms start, time_point_ms stop) {

  using ToDuration = simt::std::chrono::duration<Rep, Period>;
  auto lhs = simt::std::chrono::duration_cast<ToDuration>(start.time_since_epoch());
  auto rhs = simt::std::chrono::duration_cast<ToDuration>(stop.time_since_epoch());

  auto min = std::min(lhs.count(), rhs.count());
  auto max = std::max(lhs.count(), rhs.count());
  auto range = static_cast<Rep>(std::abs(max - min));
  auto gen_ts = generate_each_timestamp<Rep>{min, range / count};

  thrust::device_vector<Rep> timestamps(count);
  thrust::tabulate(timestamps.begin(), timestamps.end(), gen_ts);

  return timestamps;
}

}  // namespace test
}  // namespace cudf
 