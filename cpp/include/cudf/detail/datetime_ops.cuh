/*
 * SPDX-FileCopyrightText: Copyright (c) 2021, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
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
