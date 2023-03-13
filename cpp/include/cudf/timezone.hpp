/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#include <cudf/table/table.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <optional>
#include <string>

namespace cudf {

// Cycle in which the time offsets repeat
static constexpr int32_t solar_cycle_years = 400;
// Number of future entries in the timezone transition table:
// Two entries per year, over the length of the Gregorian calendar's solar cycle
static constexpr uint32_t solar_cycle_entry_count = 2 * solar_cycle_years;

/**
 * @brief Creates a transition table to convert ORC timestamps to UTC.
 *
 * Uses system's TZif files. Assumes little-endian platform when parsing these files.
 *
 * @param tzif_dir The directory where the TZif files are located
 * @param timezone_name standard timezone name (for example, "US/Pacific")
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return The transition table for the given timezone
 */
std::unique_ptr<table> make_timezone_transition_table(std::optional<std::string_view> tzif_dir,
                                                      std::string_view timezone_name,
                                                      rmm::cuda_stream_view stream);

}  // namespace cudf
