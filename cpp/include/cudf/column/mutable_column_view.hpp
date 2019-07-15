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

namespace cudf {
struct mutable_column_view {
  mutable_column_view() = default;
  ~mutable_column_view() = default;
  mutable_column_view(mutable_column_view&&) = default;
  mutable_column_view& operator=(mutable_column_view const&) = default;
  mutable_column_view& operator=(mutable_column_view&&) = default;

  mutable_column_view(void const* data, data_type type, size_type size,
              std::unique_ptr<mutable_column_view> null_mask, size_type null_count,
              std::vector<mutable_column_view> const& children);

  mutable_column_view(mutable_column_view const& other)
      : _data{other._data},
        _type{other._type},
        _size{other._size},
        _null_count{other._null_count},
        _children{other._children} {
    if (nullptr != other._null_mask.get()) {
      _null_mask = std::make_unique<mutable_column_view>(*(other._null_mask));
    }
  }

  template <typename T = void>
  T const* data() const noexcept {
    return static_cast<T const*>(_data);
  }

  size_type size() noexcept { return _size; }

  data_type type() noexcept { return _type; }

  bool nullable() const noexcept { return nullptr != _null_mask.get(); }

  size_type null_count() const noexcept { return _null_count; }

  bool has_nulls() const noexcept { return _null_count > 0; }

  mutable_column_view* null_mask() const noexcept { return _null_mask.get(); }

  mutable_column_view child(size_type child_index) const noexcept {
    return _children[child_index];
  }

 private:
  void const* _data{nullptr};
  data_type _type{INVALID};
  cudf::size_type _size{0};
  std::unique_ptr<mutable_column_view> _null_mask{nullptr};
  size_type _null_count{0};
  std::vector<mutable_column_view> _children{};
};
}  // namespace cudf