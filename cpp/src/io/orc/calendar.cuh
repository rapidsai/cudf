/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/traits.hpp>

#include <cstdint>

namespace cudf::io::orc::detail {

// Epoch days of 1582-10-15. Earlier hybrid dates use the Julian calendar.
constexpr int32_t calendar_cutover = -141427;

CUDF_HOST_DEVICE constexpr int64_t floor_div(int64_t value, int64_t divisor)
{
  return value / divisor - (value % divisor < 0);
}

/**
 * @brief Convert epoch days while preserving the civil year, month, and day.
 *
 * March-based 400-year Gregorian and 4-year Julian cycles keep intermediate values positive
 * within each cycle, including for astronomical years zero and below. All arithmetic is 64 bit
 * so that the complete int32 DATE domain can be checked before narrowing the result.
 */
template <bool to_hybrid>
CUDF_HOST_DEVICE constexpr int64_t rebase_days(int64_t days)
{
  if (days >= calendar_cutover) { return days; }

  auto const z    = days + (to_hybrid ? 719468 : 719470);
  auto const era  = floor_div(z, to_hybrid ? 146097 : 1461);
  auto const doe  = z - era * (to_hybrid ? 146097 : 1461);
  auto const yoe  = to_hybrid ? (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365
                              : (doe == 1460 ? 3 : doe / 365);
  auto const year = yoe + era * (to_hybrid ? 400 : 4);
  auto const doy  = doe - (365 * yoe + (to_hybrid ? yoe / 4 - yoe / 100 : 0));
  auto const mp   = (5 * doy + 2) / 153;
  auto const day  = doy - (153 * mp + 2) / 5 + 1;

  if constexpr (to_hybrid) {
    return 365 * year + floor_div(year, 4) + (153 * mp + 2) / 5 + day - 1 - 719470;
  } else {
    return 365 * year + floor_div(year, 4) - floor_div(year, 100) + floor_div(year, 400) +
           (153 * mp + 2) / 5 + day - 1 - 719468;
  }
}

template <bool to_hybrid>
CUDF_HOST_DEVICE constexpr bool is_representable_date(int64_t original, int64_t rebased)
{
  // The hybrid cutover gap and Julian-only leap days have no lossless counterpart.
  return rebased >= INT32_MIN and rebased <= INT32_MAX and
         rebase_days<!to_hybrid>(rebased) == original;
}

}  // namespace cudf::io::orc::detail
