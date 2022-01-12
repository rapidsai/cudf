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

#include <limits>
#include <vector>

namespace cudf {
namespace io {
namespace text {

struct byte_range_info {
  size_t offset;
  size_t size;
  bool is_first;
  bool is_last;

  static std::vector<byte_range_info> create_consecutive(size_t total_bytes, size_t range_count)
  {
    auto range_size = total_bytes / range_count;

    std::vector<byte_range_info> ranges;

    for (size_t i = 0; i < range_count; i++) {
      ranges.push_back(  //
        byte_range_info{i * range_size,
                        std::min(range_size, total_bytes - i * range_size),
                        i == 0,
                        i == range_count - 1});
    }

    return ranges;
  }

  static byte_range_info whole_source()
  {
    return {0, std::numeric_limits<size_t>::max(), true, true};
  }
};

}  // namespace text
}  // namespace io
}  // namespace cudf
