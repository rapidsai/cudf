/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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
#include <cudf/wrappers/timestamps.hpp>

namespace cudf {
namespace io {

/**
 * @brief Lookup table to compute power of ten
 */
static const __device__ __constant__ int32_t powers_of_ten[10] = {
  1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000};

/**
 * @brief Function that translates cuDF time unit to clock frequency
 */
constexpr int32_t to_clockrate(type_id timestamp_type_id)
{
  switch (timestamp_type_id) {
    case type_id::DURATION_SECONDS:
    case type_id::TIMESTAMP_SECONDS: return timestamp_s::period::den;
    case type_id::DURATION_MILLISECONDS:
    case type_id::TIMESTAMP_MILLISECONDS: return timestamp_ms::period::den;
    case type_id::DURATION_MICROSECONDS:
    case type_id::TIMESTAMP_MICROSECONDS: return timestamp_us::period::den;
    case type_id::DURATION_NANOSECONDS:
    case type_id::TIMESTAMP_NANOSECONDS: return timestamp_ns::period::den;
    default: return 0;
  }
}

}  // namespace io
}  // namespace cudf
