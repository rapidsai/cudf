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
#include <stdint.h>
#include <vector>

/**
* @brief Creates a transition table to convert ORC timestanps to UTC
*
* @param[out] table output table (2 int64_t per transition, last 800 transitions repeat forever with 400 year cycle)
* @param[in] timezone_name standard timezone name (for example, "US/Pacific")
*
* @return true if successful, false if failed to find/parse the timezone information
**/
bool BuildTimezoneTransitionTable(std::vector<int64_t> &table,
                                  const std::string &timezone_name);
