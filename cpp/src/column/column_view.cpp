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

#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <utilities/error_utils.hpp>

namespace cudf {

column_view::column_view(column_view const& other)
    : _data{other._data},
      _type{other._type},
      _size{other._size},
      _null_count{other._null_count},
      _children{other._children} {
  if (nullptr != other._null_mask.get()) {
    _null_mask = std::make_unique<column_view>(*(other._null_mask));
  }
}

column_view::column_view(data_type type, size_type size, void const* data,
                         std::unique_ptr<column_view> null_mask,
                         size_type null_count,
                         std::vector<column_view> const& children)
    : _data{data},
      _type{type},
      _size{size},
      _null_mask{std::move(null_mask)},
      _null_count{null_count},
      _children{children} {
  CUDF_EXPECTS(size >= 0, "Column size cannot be negative.");
  if (size > 0) {
    CUDF_EXPECTS(nullptr != data, "Null data pointer.");
    CUDF_EXPECTS(INVALID != type, "Invalid element type.");
  }

  if (null_count > 0) {
    CUDF_EXPECTS(nullptr != null_mask,
                 "Invalid null mask for non-zero null count.");
    CUDF_EXPECTS(BOOL1 == null_mask->type(), "Invalid type for null mask.");
  }
};
}  // namespace cudf