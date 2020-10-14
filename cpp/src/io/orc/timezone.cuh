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

#include <cudf/utilities/span.hpp>

#include <thrust/device_vector.h>

#include <stdint.h>
#include <string>
#include <vector>

namespace cudf {
namespace io {

struct timezone_table_view {
  int32_t gmt_offset;
  cudf::detail::device_span<int64_t const> ttimes;
  cudf::detail::device_span<int32_t const> offsets;
};

struct timezone_table {
  int32_t gmt_offset = 0;
  rmm::device_vector<int64_t> ttimes;
  rmm::device_vector<int32_t> offsets;
  timezone_table_view view() const { return {gmt_offset, ttimes, offsets}; }
};

/**
 * @brief Creates a transition table to convert ORC timestamps to UTC.
 *
 * @param[in] timezone_name standard timezone name (for example, "US/Pacific")
 *
 * @return The transition table (1st entry = gmtOffset, 2 int64_t per transition, last 800
 * transitions repeat forever with 400 year cycle)
 */
timezone_table BuildTimezoneTransitionTable(std::string const& timezone_name);

}  // namespace io
}  // namespace cudf
