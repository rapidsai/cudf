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

struct multistate_segment {
 public:
  inline constexpr multistate_segment() : _data(0) {}
  inline constexpr multistate_segment(uint8_t head, uint8_t tail)
    : _data((head & 0b1111) | (tail << 4))
  {
  }

  inline constexpr uint8_t get_head() const { return _data & 0b1111; }
  inline constexpr uint8_t get_tail() const { return _data >> 4; }

 private:
  uint8_t _data;
};

struct multistate {
 public:
  inline constexpr void enqueue(uint8_t head, uint8_t tail)
  {
    _segments[_size++] = multistate_segment(head, tail);
  }

  inline constexpr uint8_t size() const { return _size; }

  inline constexpr uint8_t max_tail() const
  {
    uint8_t maximum = 0;

    for (uint8_t i = 0; i < _size; i++) {
      maximum = std::max(maximum, get_tail(i));
    }

    return maximum;
  }

  inline constexpr uint8_t get_head(uint8_t idx) const { return _segments[idx].get_head(); }
  inline constexpr uint8_t get_tail(uint8_t idx) const { return _segments[idx].get_tail(); }

 private:
  static auto constexpr N = 7;
  uint8_t _size           = 0;
  multistate_segment _segments[N];
};

// lhs contains only zero?

inline constexpr multistate operator+(multistate const& lhs, multistate const& rhs)
{
  // combine two multistates together by full-joining LHS tails to RHS heads,
  // and taking the corosponding LHS heads and RHS tails.

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

}  // namespace text
}  // namespace io
}  // namespace cudf
