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

template <typename Timestamp, typename MonthType>
__device__ Timestamp add_calendrical_months_with_scale_back(Timestamp time_val,
                                                            MonthType months_val)
{
  using namespace cuda::std::chrono;

  // Get the days component from the input
  auto const days_since_epoch = floor<days>(time_val);

  // Add the number of months
  year_month_day ymd{days_since_epoch};
  ymd += duration<int32_t, months::period>{months_val};

  // If the new date isn't valid, scale it back to the last day of the
  // month.
  if (!ymd.ok()) ymd = ymd.year() / ymd.month() / last;

  // Put back the time component to the date
  return sys_days{ymd} + (time_val - days_since_epoch);
}

}  // namespace detail
}  // namespace datetime
}  // namespace cudf
