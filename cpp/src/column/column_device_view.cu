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
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <utilities/error_utils.hpp>

#include <rmm/rmm_api.h>
#include <rmm/thrust_rmm_allocator.h>

namespace cudf {

// Trivially copy all members but the children
column_device_view::column_device_view(column_view source)
    : detail::column_device_view_base{source.type(),       source.size(),
                                      source.head(),       source.null_mask(),
                                      source.null_count(), source.offset()},
      _num_children{source.num_children()} {}

// Free device memory allocated for children
void column_device_view::destroy() {
  delete this;
  // TODO Implement once support for children is added
}

// Construct a unique_ptr that invokes `destroy()` as it's deleter
std::unique_ptr<column_device_view,
                       std::function<void(column_device_view*)>>
column_device_view::create(column_view source_view, cudaStream_t stream) {

  size_type num_descendants{count_descendants(source_view)};
  if (num_descendants > 0) {
    CUDF_FAIL("Columns with children are not currently supported.");
  }

  auto deleter = [](column_device_view* v) { v->destroy(); };

  std::unique_ptr<column_device_view, decltype(deleter)> p{
      new column_device_view(source_view), deleter};

  return p;
}

}  // namespace cudf