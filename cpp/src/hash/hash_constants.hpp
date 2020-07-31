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

namespace cudf {
namespace detail {

struct md5_intermediate_data {
  uint64_t message_length = 0;
  uint32_t buffer_length  = 0;
  uint32_t hash_value[4]  = {0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476};
  uint8_t buffer[64];
};

// Type for the shift constants table.
using md5_shift_constants_type = uint32_t;

/**
 * @brief Returns pointer to device memory that contains the static
 * md5 shift constants table. On first call, this will copy the table into
 * device memory and is guaranteed to be thread-safe.
 *
 * This table is used in the MD5 hash to lookup the number of bits
 * to rotate left during each hash iteration.
 *
 * @return Device memory pointer to the MD5 shift constants table.
 */
const md5_shift_constants_type* get_md5_shift_constants();

// Type for the hash constants table.
using md5_hash_constants_type = uint32_t;

/**
 * @brief Returns pointer to device memory that contains the static
 * md5 hash constants table. On first call, this will copy the table into
 * device memory and is guaranteed to be thread-safe.
 *
 * This table is used in the MD5 hash to lookup values added to
 * the hash during each hash iteration.
 *
 * @return Device memory pointer to the MD5 hash constants table.
 */
const md5_hash_constants_type* get_md5_hash_constants();

}  // namespace detail
}  // namespace cudf
