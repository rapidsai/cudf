/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

namespace cudf {

class lists_column_view : private column_view {
 public:
  lists_column_view(column_view lists_column);
  lists_column_view(lists_column_view&& strings_view)      = default;
  lists_column_view(const lists_column_view& strings_view) = default;
  ~lists_column_view()                                     = default;
  lists_column_view& operator=(lists_column_view const&) = default;
  lists_column_view& operator=(lists_column_view&&) = default;

  static constexpr size_type offsets_column_index{0};

  using column_view::has_nulls;
  using column_view::null_count;
  using column_view::null_mask;
  using column_view::offset;
  using column_view::size;

  column_view parent() const;

  column_view offsets() const;
  // child == column_view.child(1) since offsets occupies
  column_view child() const;
};

}  // namespace cudf
