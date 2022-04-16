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

#include <cudf/types.hpp>

namespace cudf {

namespace detail {

enum class index_type_name {
  LEFT,
  RIGHT,
};

template <index_type_name IndexName>
struct strong_index {
 public:
  constexpr explicit strong_index(size_type index) : _index(index) {}

  constexpr explicit operator size_type() const { return _index; }

  constexpr size_type value() const { return _index; }

  constexpr strong_index operator+(size_type const& v) const { return strong_index(_index + v); }
  constexpr strong_index operator-(size_type const& v) const { return strong_index(_index - v); }
  constexpr strong_index operator*(size_type const& v) const { return strong_index(_index * v); }

  constexpr bool operator==(size_type v) const { return _index == v; }
  constexpr bool operator!=(size_type v) const { return _index != v; }
  constexpr bool operator<=(size_type v) const { return _index <= v; }
  constexpr bool operator>=(size_type v) const { return _index >= v; }
  constexpr bool operator<(size_type v) const { return _index < v; }
  constexpr bool operator>(size_type v) const { return _index > v; }

  constexpr strong_index& operator=(strong_index const& s)
  {
    _index = s._index;
    return *this;
  }
  constexpr strong_index& operator=(size_type const& i)
  {
    _index = i;
    return *this;
  }
  constexpr strong_index& operator+=(size_type const& v)
  {
    _index += v;
    return *this;
  }
  constexpr strong_index& operator-=(size_type const& v)
  {
    _index -= v;
    return *this;
  }
  constexpr strong_index& operator*=(size_type const& v)
  {
    _index *= v;
    return *this;
  }

  constexpr strong_index& operator++()
  {
    ++_index;
    return *this;
  }
  constexpr strong_index operator++(int)
  {
    strong_index tmp(*this);
    ++_index;
    return tmp;
  }
  constexpr strong_index& operator--()
  {
    --_index;
    return *this;
  }
  constexpr strong_index operator--(int)
  {
    strong_index tmp(*this);
    --_index;
    return tmp;
  }

  friend std::ostream& operator<<(std::ostream& os, strong_index<IndexName> s)
  {
    return os << s._index;
  }
  friend std::istream& operator>>(std::istream& is, strong_index<IndexName>& s)
  {
    return is >> s._index;
  }

 private:
  size_type _index;
};

}  // namespace detail

using lhs_index_type = detail::strong_index<detail::index_type_name::LEFT>;
using rhs_index_type = detail::strong_index<detail::index_type_name::RIGHT>;

}  // namespace cudf
