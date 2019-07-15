/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <vector>

namespace cudf {
struct column_view {
  column_view() = default;
  ~column_view() = default;
  column_view(column_view const&) = default;
  column_view(column_view&&) = default;
  column_view& operator=(column_view const&) = default;
  column_view& operator=(column_view&&) = default;

  column_view(data_type type, size_type size, void const* data,
              bitmask_type const* null_mask, size_type null_count,
              std::vector<column_view> const& children = {});

  template <typename T = void>
  T const* data() const noexcept {
    return static_cast<T const*>(_data);
  }

  size_type size() const noexcept { return _size; }

  data_type type() const noexcept { return _type; }

  bool nullable() const noexcept { return nullptr != _null_mask; }

  size_type null_count() const noexcept { return _null_count; }

  bool has_nulls() const noexcept { return _null_count > 0; }

  bitmask_type const* null_mask() const noexcept { return _null_mask; }

  column_view child(size_type child_index) const noexcept {
    return _children[child_index];
  }

 private:
  data_type _type{INVALID};
  cudf::size_type _size{0};
  void const* _data{nullptr};
  bitmask_type const* _null_mask{nullptr};
  size_type _null_count{0};
  std::vector<column_view> _children{};
};
}  // namespace cudf