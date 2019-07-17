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

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>

#include <rmm/device_buffer.hpp>

#include <algorithm>
#include <vector>

namespace cudf {

column_view const column::view() const {
  std::vector<column_view> child_views(_children.size());
  std::copy(begin(_children), end(_children), begin(child_views));

  return column_view{_type,
                     _size,
                     const_cast<void *>(_data.data()),
                     const_cast<bitmask_type *>(
                         static_cast<bitmask_type const *>(_null_mask.data())),
                     _null_count,
                     0,
                     std::move(child_views)};
}

column_view column::view() {
  std::vector<column_view> child_views(_children.size());
  std::copy(begin(_children), end(_children), begin(child_views));

  return column_view{_type,
                     _size,
                     _data.data(),
                     static_cast<bitmask_type *>(_null_mask.data()),
                     _null_count,
                     0,
                     std::move(child_views)};
}

}  // namespace cudf