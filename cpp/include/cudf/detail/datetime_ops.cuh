/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <cuda/std/chrono>

namespace cudf {
namespace datetime {
namespace detail {
using namespace cuda::std::chrono;

template <typename Timestamp>
__device__ Timestamp add_calendrical_months_with_scale_back(Timestamp time_val, months months_val)
{
  auto const days_since_epoch = floor<days>(time_val);

  auto const date = [&]() {
    auto const ymd = year_month_day{days_since_epoch} + months_val;
    return ymd.ok() ? ymd : ymd.year() / ymd.month() / last;
  }();

  auto const time = (time_val - days_since_epoch);

  return sys_days{date} + time;
}

}  // namespace detail
}  // namespace datetime
}  // namespace cudf
