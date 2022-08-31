/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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
    _heads |= (head & 0xFu) << (_size * 4);
    _tails |= (tail & 0xFu) << (_size * 4);
    _size++;
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
  [[nodiscard]] constexpr uint8_t get_head(uint8_t idx) const
  {
    return (_heads >> (idx * 4)) & 0xFu;
  }

  /**
   * @brief get's the Nth (____, tail] value state this multistate represents
   */
  [[nodiscard]] constexpr uint8_t get_tail(uint8_t idx) const
  {
    return (_tails >> (idx * 4)) & 0xFu;
  }

 private:
  uint8_t _size = 0;
  uint32_t _heads{};
  uint32_t _tails{};
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
