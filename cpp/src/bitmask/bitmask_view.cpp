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

#include <cudf/bitmask_view.hpp>
#include <cudf/types.hpp>
#include <utilities/error_utils.hpp>

namespace cudf {

mutable_bitmask_view::mutable_bitmask_view(bitmask_type* mask, size_type size)
    : _mask{mask}, _size{size} {
  if (nullptr == mask) {
    CUDF_EXPECTS(0 == size, "Null mask for non-empty bitmask.");
  }
}

bitmask_view::bitmask_view(bitmask_type const* mask, size_type size)
    : mutable_view{const_cast<bitmask_type*>(mask), size} {}

bitmask_view::bitmask_view(mutable_bitmask_view m_view)
    : mutable_view{m_view} {}

}  // namespace cudf