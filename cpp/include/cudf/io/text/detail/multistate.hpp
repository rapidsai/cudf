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

#include <cstdint>

namespace cudf {
namespace io {
namespace text {
namespace detail {

/**
 * @brief Represents up to 7 segments
 */
struct multistate {
 private:
  /**
   * @brief represents a (head, tail] segment, stored as a single 8 bit value
   */
  struct multistate_segment {
   public:
    /**
     * @brief Creates a segment which represents (0, 0]
     */

    constexpr multistate_segment() = default;
    /**
     * @brief Creates a segment which represents (head, tail]
     *
     * @param head the (head, ____] value. Undefined behavior for values >= 16
     * @param tail the (____, tail] value. Undefined behavior for values >= 16
     */

    constexpr multistate_segment(uint8_t head, uint8_t tail) : _data((head & 0b1111) | (tail << 4))
    {
    }

    /**
     * @brief Get's the (head, ____] value from the segment.
     */
    [[nodiscard]] constexpr uint8_t get_head() const { return _data & 0b1111; }

    /**
     * @brief Get's the (____, tail] value from the segment.
     */
    [[nodiscard]] constexpr uint8_t get_tail() const { return _data >> 4; }

   private:
    uint8_t _data{0};
  };

 public:
  /**
   * @brief The maximum state (head or tail) this multistate can represent
   */

  static auto constexpr max_segment_value = 15;
  /**
   * @brief The maximum number of segments this multistate can represent
   */
  static auto constexpr max_segment_count = 7;

  /**
   * @brief Enqueues a (head, tail] segment to this multistate
   *
   * @note: The behavior of this function is undefined if size() => max_segment_count
   */
  constexpr void enqueue(uint8_t head, uint8_t tail)
  {
    _segments[_size++] = multistate_segment(head, tail);
  }

  /**
   * @brief get's the number of segments this multistate represents
   */
  [[nodiscard]] constexpr uint8_t size() const { return _size; }

  /**
   * @brief get's the highest (____, tail] value this multistate represents
   */
  [[nodiscard]] constexpr uint8_t max_tail() const
  {
    uint8_t maximum = 0;

    for (uint8_t i = 0; i < _size; i++) {
      maximum = std::max(maximum, get_tail(i));
    }

    return maximum;
  }

  /**
   * @brief get's the Nth (head, ____] value state this multistate represents
   */
  [[nodiscard]] constexpr uint8_t get_head(uint8_t idx) const { return _segments[idx].get_head(); }

  /**
   * @brief get's the Nth (____, tail] value state this multistate represents
   */
  [[nodiscard]] constexpr uint8_t get_tail(uint8_t idx) const { return _segments[idx].get_tail(); }

 private:
  uint8_t _size = 0;
  multistate_segment _segments[max_segment_count];
};

/**
 * @brief associatively inner-joins transition histories.
 *
 * Examples:
 *           <(0, 5]> + <(5, 9]>         = <(0, 9]>
 *           <(0, 5]> + <(6, 9]>         = <>
 *   <(0, 1], (0, 2]> + <(2, 3], (1, 4]> = <(0, 4], (0, 3]>
 *   <(0, 1], (0, 2]> + <(1, 3]>         = <(0, 3]>
 *
 * Head and tail value are limited to [0, 1, ..., 16]
 *
 * @param lhs past segments
 * @param rhs future segments
 * @return full join of past and future segments
 */
constexpr multistate operator+(multistate const& lhs, multistate const& rhs)
{
  // combine two multistates together by full-joining LHS tails to RHS heads,
  // and taking the corresponding LHS heads and RHS tails.

  multistate result;
  for (uint8_t lhs_idx = 0; lhs_idx < lhs.size(); lhs_idx++) {
    auto tail = lhs.get_tail(lhs_idx);
    for (uint8_t rhs_idx = 0; rhs_idx < rhs.size(); rhs_idx++) {
      auto head = rhs.get_head(rhs_idx);
      if (tail == head) { result.enqueue(lhs.get_head(lhs_idx), rhs.get_tail(rhs_idx)); }
    }
  }
  return result;
}

}  // namespace detail
}  // namespace text
}  // namespace io
}  // namespace cudf
