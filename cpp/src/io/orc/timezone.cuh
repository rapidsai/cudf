/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

struct timezone_table_view {
  int32_t gmt_offset = 0;
  cudf::device_span<int64_t const> ttimes;
  cudf::device_span<int32_t const> offsets;
};

// Cycle in which the time offsets repeat
static constexpr int32_t cycle_years = 400;
// Number of seconds in 400 years
static constexpr int64_t cycle_seconds =
  cuda::std::chrono::duration_cast<duration_s>(duration_D{365 * cycle_years + (100 - 3)}).count();
// Two entries per year, over the length of the cycle
static constexpr uint32_t cycle_entry_cnt = 2 * cycle_years;

/**
 * @brief Returns the GMT offset for a given date and given timezone table.
 *
 * @param ttimes Transition times; trailing `cycle_entry_cnt` entries are used for all times
 * beyond the one covered by the TZif file
 * @param offsets Time offsets in specific intervals; trailing `cycle_entry_cnt` entries are used
 * for all times beyond the one covered by the TZif file
 * @param count Number of elements in @p ttimes and @p offsets
 * @param ts ORC timestamp
 *
 * @return GMT offset
 */
CUDF_HOST_DEVICE inline int32_t get_gmt_offset_impl(int64_t const* ttimes,
                                                    int32_t const* offsets,
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
inline __host__ int32_t get_gmt_offset(cudf::host_span<int64_t const> ttimes,
                                       cudf::host_span<int32_t const> offsets,
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
inline __device__ int32_t get_gmt_offset(cudf::device_span<int64_t const> ttimes,
                                         cudf::device_span<int32_t const> offsets,
                                         int64_t ts)
{
  return get_gmt_offset_impl(ttimes.begin(), offsets.begin(), ttimes.size(), ts);
}

class timezone_table {
  int32_t gmt_offset = 0;
  rmm::device_uvector<int64_t> ttimes;
  rmm::device_uvector<int32_t> offsets;

 public:
  // Safe to use the default stream, device_uvectors will not change after they are created empty
  timezone_table() : ttimes{0, cudf::get_default_stream()}, offsets{0, cudf::get_default_stream()}
  {
  }
  timezone_table(int32_t gmt_offset,
                 rmm::device_uvector<int64_t>&& ttimes,
                 rmm::device_uvector<int32_t>&& offsets)
    : gmt_offset{gmt_offset}, ttimes{std::move(ttimes)}, offsets{std::move(offsets)}
  {
  }
  [[nodiscard]] timezone_table_view view() const { return {gmt_offset, ttimes, offsets}; }
};

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
timezone_table build_timezone_transition_table(std::string const& timezone_name,
                                               rmm::cuda_stream_view stream);

}  // namespace io
}  // namespace cudf
