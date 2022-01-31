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

#include <cudf/detail/utilities/integer_utils.hpp>

#include <limits>
#include <vector>

namespace cudf {
namespace io {
namespace text {

struct byte_range_info {
 public:
  int64_t offset;
  int64_t size;

  byte_range_info() : offset(0), size(0) {}
  byte_range_info(int64_t offset, int64_t size) : offset(offset), size(size) {}

  static std::vector<byte_range_info> create_consecutive(int64_t total_bytes, int64_t range_count)
  {
    auto range_size = util::div_rounding_up_safe(total_bytes, range_count);

    std::vector<byte_range_info> ranges;

    for (int64_t i = 0; i < range_count; i++) {
      auto offset = i * range_size;
      auto size   = std::min(range_size, total_bytes - offset);
      ranges.push_back(byte_range_info{offset, size});
    }

    return ranges;
  }

  static byte_range_info whole_source() { return {0, std::numeric_limits<int64_t>::max()}; }
};

}  // namespace text
}  // namespace io
}  // namespace cudf
