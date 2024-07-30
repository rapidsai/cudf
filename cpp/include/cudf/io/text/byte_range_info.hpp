/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#include <cudf/utilities/error.hpp>
#include <cudf/utilities/export.hpp>

#include <cstdint>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace io {
namespace text {
/**
 * @addtogroup io_readers
 * @{
 * @file
 */

/**
 * @brief stores offset and size used to indicate a byte range
 */
class byte_range_info {
 private:
  int64_t _offset{};  ///< offset in bytes
  int64_t _size{};    ///< size in bytes

 public:
  constexpr byte_range_info() = default;
  /**
   * @brief Constructs a byte_range_info object
   *
   * @param offset offset in bytes
   * @param size size in bytes
   */
  constexpr byte_range_info(int64_t offset, int64_t size) : _offset(offset), _size(size)
  {
    CUDF_EXPECTS(offset >= 0, "offset must be non-negative");
    CUDF_EXPECTS(size >= 0, "size must be non-negative");
  }

  /**
   * @brief Copy constructor
   *
   * @param other byte_range_info object to copy
   */
  constexpr byte_range_info(byte_range_info const& other) noexcept = default;
  /**
   * @brief  Copy assignment operator
   *
   * @param other byte_range_info object to copy
   * @return this object after copying
   */
  constexpr byte_range_info& operator=(byte_range_info const& other) noexcept = default;

  /**
   * @brief Get the offset in bytes
   *
   * @return Offset in bytes
   */
  [[nodiscard]] constexpr int64_t offset() { return _offset; }

  /**
   * @brief Get the size in bytes
   *
   * @return Size in bytes
   */
  [[nodiscard]] constexpr int64_t size() { return _size; }

  /**
   * @brief Returns whether the span is empty.
   *
   * @return true iff the span is empty, i.e. `size() == 0`
   */
  [[nodiscard]] constexpr bool empty() { return size() == 0; }
};

/**
 * @brief Create a collection of consecutive ranges between [0, total_bytes).
 *
 * Each range wil be the same size except if `total_bytes` is not evenly divisible by
 * `range_count`, in which case the last range size will be the remainder.
 *
 * @param total_bytes total number of bytes in all ranges
 * @param range_count total number of ranges in which to divide bytes
 * @return Vector of range objects
 */
std::vector<byte_range_info> create_byte_range_infos_consecutive(int64_t total_bytes,
                                                                 int64_t range_count);

/**
 * @brief Create a byte_range_info which represents as much of a file as possible. Specifically,
 * ``[0, numeric_limits<int64_t>:\:max())``.
 *
 * @return Byte range info of size ``[0, numeric_limits<int64_t>:\:max())``
 */
byte_range_info create_byte_range_info_max();

/** @} */  // end of group

}  // namespace text
}  // namespace io
}  // namespace CUDF_EXPORT cudf
