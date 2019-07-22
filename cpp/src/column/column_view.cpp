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

#include <exception>
#include <vector>

namespace cudf {

column_view::column_view(data_type type, size_type size, void const* data,
                         bitmask_type const* null_mask, size_type null_count,
                         size_type offset,
                         std::vector<column_view> const& children)
    : _type{type},
      _size{size},
      _data{data},
      _null_mask{null_mask},
      _null_count{null_count},
      _offset{offset},
      _children{children} {
  CUDF_EXPECTS(size >= 0, "Column size cannot be negative.");

  if (type.id() == EMPTY) {
    _null_count = size;
    CUDF_EXPECTS(nullptr == data, "EMPTY column should have no data.");
    CUDF_EXPECTS(nullptr == null_mask,
                 "EMPTY column should have no null mask.");
    CUDF_EXPECTS(children.size() == 0, "EMPTY column cannot have children.");
  } else if (size > 0) {
    CUDF_EXPECTS(nullptr != data, "Null data pointer.");
  }

  CUDF_EXPECTS(offset >= 0, "Invalid offset.");

  if ((null_count > 0) and (type.id() != EMPTY)) {
    CUDF_EXPECTS(nullptr != null_mask,
                 "Invalid null mask for non-zero null count.");
  }
}

// If null count is known, returns it. Else, computes it and returns
size_type column_view::null_count() const noexcept {
  if (_null_count != cudf::UNKNOWN_NULL_COUNT) {
    return _null_count;
  } else {
    CUDF_FAIL("On-demand null count computation not yet implemented.");
  }
}

//mutable_column_view::operator column_view() {
//  // Convert children to immutable views
//  std::vector<column_view> child_views(num_children());
//  std::copy(cbegin(_children), cend(_children), begin(child_views));
//  return column_view{_type,
//                     _size,
//                     _data,
//                     _null_mask,
//                     _null_count,
//                     _offset,
//                     std::move(child_views)};
//}

}  // namespace cudf