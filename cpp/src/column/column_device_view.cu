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
#include <cudf/types.hpp>
#include <utilities/error_utils.hpp>

#include <rmm/rmm_api.h>
#include <rmm/thrust_rmm_allocator.h>

namespace cudf {
namespace {
/**---------------------------------------------------------------------------*
 * @brief Counts the number of descendants of the specified parent.
 *
 * @param parent The parent whose descendants will be counted
 * @return size_type The number of descendants of the parent
 *---------------------------------------------------------------------------**/
size_type count_descendants(column_view parent) {
  size_type count{parent.num_children()};
  for (size_type i = 0; i < parent.num_children(); ++i) {
    count += count_descendants(parent.child(i));
  }
  return count;
}
}  // namespace

// Trivially copy all members but the children
column_device_view::column_device_view(column_view source)
    : detail::column_device_view_base{source.type(),       source.size(),
                                      source.head(),       source.null_mask(),
                                      source.null_count(), source.offset()},
      _num_children{source.num_children()} {}


// Free device memory allocated for children
void column_device_view::destroy() {
  // TODO Implement once support for children is added
}

// Construct a unique_ptr that invokes `destroy()` as it's deleter
auto column_device_view::create(column_view source, cudaStream_t stream) {
  size_type num_descendants{count_descendants(source)};
  if (num_descendants > 0) {
    CUDF_FAIL("Columns with children are not currently supported.");
  }

  auto deleter = [](column_device_view* v) { v->destroy(); };

  std::unique_ptr<column_device_view, decltype(deleter)> p{
      new column_device_view(source), deleter};

  return p;
}

}  // namespace cudf