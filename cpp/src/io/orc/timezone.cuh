/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <stdint.h>
#include <string>
#include <vector>

namespace cudf {
namespace io {

struct timezone_table_view {
  int32_t gmt_offset = 0;
  cudf::detail::device_span<int64_t const> ttimes;
  cudf::detail::device_span<int32_t const> offsets;
};

static constexpr int64_t day_seconds = 24 * 60 * 60;
// Cycle in which the time offsets repeat
static constexpr uint32_t cycle_years = 400;
// Number of seconds in 400 years
static constexpr int64_t cycle_seconds = (365 * 400 + (100 - 3)) * day_seconds;
// Two entries per year, over the length of the cycle
static constexpr uint32_t cycle_entry_cnt = 2 * cycle_years;

/**
 * @brief Returns the GMT offset for a given date and given timezone table.
 *
 * @param ttimes Transition times; trailing @ref `cycle_entry_cnt` entires are used for all times
 * beyond the one covered by the TZif file
 * @param offsets Time offsets in specific intervals; trailing `cycle_entry_cnt` entires are used
 * for all times beyond the one covered by the TZif file
 * @param count Number of elements in @ref ttimes and @ref offsets
 * @param ts ORC timestamp
 *
 * @return GMT offset
 */
CUDA_HOST_DEVICE_CALLABLE int32_t get_gmt_offset_impl(int64_t const *ttimes,
                                                      int32_t const *offsets,
                                                      size_t count,
                                                      int64_t ts)
{
  // Returns start of the range if all elements are larger than the input timestamp
  auto last_less_equal_ttime_idx = [&](long begin_idx, long end_idx, int64_t ts) {
    auto const first_larger_ttime =
      thrust::upper_bound(thrust::seq, ttimes + begin_idx, ttimes + end_idx, ts);
    // Element before the first larger element is the last one less of equal
    return std::max(first_larger_ttime - ttimes - 1, begin_idx);
  };

  auto const file_entry_cnt = count - cycle_entry_cnt;
  // Search in the file entries if the timestamp is in range
  if (ts <= ttimes[file_entry_cnt - 1]) {
    return offsets[last_less_equal_ttime_idx(0, file_entry_cnt, ts)];
  } else {
    // Search in the 400-year cycle if outside of the file entries range
    return offsets[last_less_equal_ttime_idx(
      file_entry_cnt, count, (ts + cycle_seconds) % cycle_seconds)];
  }
}

/**
 * @brief Host `get_gmt_offset` interface.
 *
 * Implemented in `get_gmt_offset_impl`.
 */
inline __host__ int32_t get_gmt_offset(cudf::detail::host_span<int64_t const> ttimes,
                                       cudf::detail::host_span<int32_t const> offsets,
                                       int64_t ts)
{
  CUDF_EXPECTS(ttimes.size() == offsets.size(),
               "transition times and offsets must have the same length");
  return get_gmt_offset_impl(ttimes.begin(), offsets.begin(), ttimes.size(), ts);
}

/**
 * @brief Device `get_gmt_offset` interface.
 *
 * Implemented in `get_gmt_offset_impl`.
 */
inline __device__ int32_t get_gmt_offset(cudf::detail::device_span<int64_t const> ttimes,
                                         cudf::detail::device_span<int32_t const> offsets,
                                         int64_t ts)
{
  return get_gmt_offset_impl(ttimes.begin(), offsets.begin(), ttimes.size(), ts);
}

struct timezone_table {
  int32_t gmt_offset = 0;
  rmm::device_vector<int64_t> ttimes;
  rmm::device_vector<int32_t> offsets;
  timezone_table_view view() const { return {gmt_offset, ttimes, offsets}; }
};

/**
 * @brief Creates a transition table to convert ORC timestamps to UTC.
 *
 * Uses system's TZif files. Assumes little-endian platform when parsing these files.
 *
 * @param timezone_name standard timezone name (for example, "US/Pacific")
 *
 * @return The transition table for the given timezone
 */
timezone_table build_timezone_transition_table(std::string const &timezone_name);

}  // namespace io
}  // namespace cudf
