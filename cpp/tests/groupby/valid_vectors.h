/*
* Copyright (c) 2018, NVIDIA CORPORATION.
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
#include <memory>
#include <functional>
#include <vector>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <cudf.h>
#include <utilities/cudf_utils.h>
#include <utilities/bit_util.cuh>

// host_valid_pointer is a wrapper for gdf_valid_type* with custom deleter
using host_valid_pointer = std::basic_string<uint8_t>;

// Create a valid pointer and init randomly the last half column
static inline host_valid_pointer create_and_init_valid(size_t length)
{
    auto n_bytes = PaddedLength ( gdf_get_num_chars_bitmask(length) );
    uint8_t *ptr= new uint8_t[n_bytes];
    for (size_t i = 0; i < length; ++i) {
        // if (i < length / 2 || std::rand() % 2 == 1) {
        //     gdf::util::turn_bit_on(ptr, i);
        // } else {
        //     gdf::util::turn_bit_off(ptr, i);
        // }
    }
    return host_valid_pointer(n_bytes, 'C');
}


// Create a valid pointer and init randomly the last half column
static inline
void init_valid(std::uint8_t *      valid_bits,
                const std::int64_t  start_offset,
                size_t              length)
{
    gdf_valid_type current_byte;
    size_t byte_offset = start_offset / 8;
    size_t bit_offset = start_offset % 8;

    if (length > 0) {
        current_byte = valid_bits[byte_offset];
    }
    for (size_t i = 0; i < length; ++i) {
        if (i < length / 2 || std::rand() % 2 == 1) {
            current_byte |= gdf::util::byte_bitmask(bit_offset);
        } else {
            current_byte &= gdf::util::flipped_bitmask(bit_offset);
        }
        ++bit_offset;
        valid_bits[byte_offset] = current_byte;
        if (bit_offset == 8) {
            bit_offset = 0;
            ++byte_offset;
            current_byte = valid_bits[byte_offset];
        }
    }
}

// Initialize valids
static inline void initialize_valids(std::vector<host_valid_pointer>& valids, size_t size, size_t length)
{
    for (size_t i = 0; i < size; ++i) {
        valids.push_back(create_and_init_valid(length));
    }
}

