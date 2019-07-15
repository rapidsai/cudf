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

namespace cudf {

column::operator column_view() const {
  std::vector<column_view> child_views(_children.size());
  std::copy(begin(_children), end(_children), begin(child_views));

  std::unique_ptr<column_view> null_mask_view{nullptr};
  if (nullptr != _null_mask.get()) {
    null_mask_view = std::make_unique<column_view>(_null_mask->view());
  }

  return column_view{_type,        _size,
                     _data.data(), std::move(null_mask_view),
                     _null_count,  std::move(child_views)};
}

}  // namespace cudf