/*
 * SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/table/table_device_view.cuh>
#include <cudf/timezone.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

namespace cudf::detail {

/**
 * @brief Returns the UT offset for a given date and given timezone table.
 *
 * @param transition_times Transition times; trailing `solar_cycle_entry_count` entries are used for
 * all times beyond the one covered by the TZif file
 * @param offsets Time offsets in specific intervals; trailing `solar_cycle_entry_count` entries are
 * used for all times beyond the one covered by the TZif file
 * @param ts ORC timestamp
 *
 * @return offset from UT, in seconds
 */
inline __device__ duration_s get_ut_offset(table_device_view tz_table, timestamp_s ts)
{
  if (tz_table.num_rows() == 0) { return duration_s{0}; }

  cudf::device_span<timestamp_s const> transition_times(tz_table.column(0).head<timestamp_s>(),
                                                        static_cast<size_t>(tz_table.num_rows()));

  auto const ts_ttime_it = [&]() {
    auto last_less_equal = [](auto begin, auto end, auto value) {
      auto const first_larger = thrust::upper_bound(thrust::seq, begin, end, value);
      // Return start of the range if all elements are larger than the value
      if (first_larger == begin) return begin;
      // Element before the first larger element is the last one less or equal
      return first_larger - 1;
    };

    auto const file_entry_end =
      transition_times.begin() + (transition_times.size() - solar_cycle_entry_count);

    if (ts <= *(file_entry_end - 1)) {
      // Search the file entries if the timestamp is in range
      return last_less_equal(transition_times.begin(), file_entry_end, ts);
    } else {
      auto project_to_cycle = [](timestamp_s ts) {
        // Years divisible by four are leap years
        // Exceptions are years divisible by 100, but not divisible by 400
        static constexpr int32_t num_leap_years_in_cycle =
          solar_cycle_years / 4 - (solar_cycle_years / 100 - solar_cycle_years / 400);
        static constexpr duration_s cycle_s = cuda::std::chrono::duration_cast<duration_s>(
          duration_D{365 * solar_cycle_years + num_leap_years_in_cycle});
        return timestamp_s{(ts.time_since_epoch() + cycle_s) % cycle_s};
      };
      // Search the 400-year cycle if outside of the file entries range
      return last_less_equal(file_entry_end, transition_times.end(), project_to_cycle(ts));
    }
  }();

  return tz_table.column(1).element<duration_s>(ts_ttime_it - transition_times.begin());
}

}  // namespace cudf::detail
