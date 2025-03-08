

/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>

namespace cudf {
namespace jit {

/// @brief This is a minified version of `cudf::column_device_view` for use in JIT kernels.
struct column_device_view {
  /// @copydoc cudf::column_device_view::_data
  void* __restrict__ const data = nullptr;

  /// @copydoc cudf::column_device_view::d_children
  void* __restrict__ const children = nullptr;

  /// @copydoc cudf::column_device_view::_null_mask
  cudf::bitmask_type const* const __restrict__ nullmask = nullptr;

  /// @copydoc cudf::column_device_view::_type
  cudf::data_type const type = {};

  /// @copydoc cudf::column_device_view::_size
  cudf::size_type const size = 0;
};

}  // namespace jit
}  // namespace cudf
