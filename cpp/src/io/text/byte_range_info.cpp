/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/text/byte_range_info.hpp>

#include <limits>

namespace cudf {
namespace io {
namespace text {

byte_range_info create_byte_range_info_max() { return {0, std::numeric_limits<int64_t>::max()}; }

std::vector<byte_range_info> create_byte_range_infos_consecutive(int64_t total_bytes,
                                                                 int64_t range_count)
{
  auto range_size = util::div_rounding_up_safe(total_bytes, range_count);
  auto ranges     = std::vector<byte_range_info>();

  ranges.reserve(range_size);

  for (int64_t i = 0; i < range_count; i++) {
    auto offset = i * range_size;
    auto size   = std::min(range_size, total_bytes - offset);
    ranges.emplace_back(offset, size);
  }

  return ranges;
}

}  // namespace text
}  // namespace io
}  // namespace cudf
