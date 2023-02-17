/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>
#include <io/utilities/time_utils.cuh>

#include <rmm/device_uvector.hpp>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

#include <cstdint>
#include <string>
#include <vector>

namespace cudf {
namespace io {

// Cycle in which the time offsets repeat
static constexpr int32_t cycle_years = 400;
// Two entries per year, over the length of the cycle
static constexpr uint32_t cycle_entry_cnt = 2 * cycle_years;

inline __device__ auto project_to_cycle(timestamp_s ts)
{
  static constexpr duration_s cycle_s =
    cuda::std::chrono::duration_cast<duration_s>(duration_D{365 * cycle_years + (100 - 3)});
  return timestamp_s{(ts.time_since_epoch() + cycle_s) % cycle_s};
}

/**
 * @brief Returns the GMT offset for a given date and given timezone table.
 *
 * @param ttimes Transition times; trailing `cycle_entry_cnt` entries are used for all times
 * beyond the one covered by the TZif file
 * @param offsets Time offsets in specific intervals; trailing `cycle_entry_cnt` entries are used
 * for all times beyond the one covered by the TZif file
 * @param ts ORC timestamp
 *
 * @return GMT offset
 */
inline __device__ duration_s::rep get_gmt_offset(table_device_view tz_table, timestamp_s ts)
{
  if (tz_table.num_rows() == 0) { return 0; }

  cudf::device_span<timestamp_s const> ttimes(tz_table.column(0).head<timestamp_s>(),
                                              static_cast<size_t>(tz_table.num_rows()));
  cudf::device_span<duration_s::rep const> offsets(tz_table.column(1).head<duration_s::rep>(),
                                                   static_cast<size_t>(tz_table.num_rows()));

  auto const ts_ttime_it = [&]() {
    auto last_less_equal = [](auto begin, auto end, auto value) {
      auto const first_larger = thrust::upper_bound(thrust::seq, begin, end, value);
      // Return start of the range if all elements are larger than the value
      if (first_larger == begin) return begin;
      // Element before the first larger element is the last one less or equal
      return first_larger - 1;
    };

    auto const file_entry_end = ttimes.begin() + (ttimes.size() - cycle_entry_cnt);

    if (ts <= *(file_entry_end - 1)) {
      // Search the file entries if the timestamp is in range
      return last_less_equal(ttimes.begin(), file_entry_end, ts);
    } else {
      // Search the 400-year cycle if outside of the file entries range
      return last_less_equal(file_entry_end, ttimes.end(), project_to_cycle(ts));
    }
  }();

  return offsets[ts_ttime_it - ttimes.begin()];
}

/**
 * @brief Creates a transition table to convert ORC timestamps to UTC.
 *
 * Uses system's TZif files. Assumes little-endian platform when parsing these files.
 *
 * @param timezone_name standard timezone name (for example, "US/Pacific")
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return The transition table for the given timezone
 */
std::unique_ptr<table> build_timezone_transition_table(std::string const& timezone_name,
                                                       rmm::cuda_stream_view stream);

}  // namespace io
}  // namespace cudf
