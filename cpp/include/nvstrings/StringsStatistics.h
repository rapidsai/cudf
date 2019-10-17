/*
* Copyright (c) 2018-2019, NVIDIA CORPORATION.  All rights reserved.
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
#include <cstddef>
#include <vector>

struct StringsStatistics
{
    // lengths
    size_t total_bytes, total_chars;
    size_t bytes_avg, bytes_max, bytes_min, bytes_95;
    size_t chars_avg, chars_max, chars_min, chars_95;
    // memory
    size_t total_memory; // device memory only
    size_t mem_avg, mem_max, mem_min, mem_95; // per string
    // strings
    size_t total_strings;
    size_t total_nulls;
    size_t total_empty;
    size_t unique_strings;
    // characters
    size_t whitespace_count;
    size_t digits_count;
    size_t uppercase_count, lowercase_count;
    // histogram of characters
    std::vector<std::pair<unsigned int,unsigned int> > char_counts;
};