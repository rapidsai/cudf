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

#include <cudf/types.hpp>

#include <memory>
#include <string>

namespace cudf {
namespace io {

/**
 * @brief Creates a transition table to convert ORC timestamps to UTC.
 *
 * Uses system's TZif files. Assumes little-endian platform when parsing these files.
 *
 * @param tzif_dir TODO
 * @param timezone_name standard timezone name (for example, "US/Pacific")
 *
 * @return The transition table for the given timezone
 */
std::unique_ptr<table> build_timezone_transition_table(std::optional<std::string> const& tzif_dir,
                                                       std::string const& timezone_name);

}  // namespace io
}  // namespace cudf
