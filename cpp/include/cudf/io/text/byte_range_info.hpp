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

#pragma once

#include <cudf/detail/utilities/integer_utils.hpp>

#include <vector>

namespace cudf {
namespace io {
namespace text {

/**
 * @brief stores offset and size used to indicate a byte range
 */
struct byte_range_info {
 public:
  int64_t offset;
  int64_t size;

  byte_range_info() : offset(0), size(0) {}
  byte_range_info(int64_t offset, int64_t size) : offset(offset), size(size) {}

  static byte_range_info whole_source();
  /**
   * @brief Create a collection of consecutive ranges between [0, total_bytes).
   *
   * Each range wil be the same size except if `total_bytes` is not evenly divisible by
   * `range_count`, in which cas the last range size will be the remainder.
   *
   * @param total_bytes total number of bytes in all ranges
   * @param range_count total number of ranges in which to divide bytes
   * @return
   */
  static std::vector<byte_range_info> create_consecutive(int64_t total_bytes, int64_t range_count);
};

}  // namespace text
}  // namespace io
}  // namespace cudf
