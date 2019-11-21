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

#include <cudf/types.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/detail/copy_range.cuh>

namespace cudf {
namespace strings {
namespace detail {

// TODO: this may no longer be necessary
std::unique_ptr<column> copy_range(strings_column_view const& source,
                                   strings_column_view const& target,
                                   // TODO: first & last are used for the fill
                                   // function for strings (instead of begin &
                                   // end)
                                   size_type source_begin, size_type source_end,
                                   size_type target_begin,
                                   rmm::mr::device_memory_resource* mr,
                                   cudaStream_t stream) {
  CUDF_EXPECTS((source_begin >= 0) &&
                 (source_begin <= source_end) &&
                 (source_begin < source.size()) &&
                 (source_end <= source.size()) &&
                 (target_begin >= 0) &&
                 (target_begin < target.size()) &&
                 (target_begin + (source_end - source_begin) <=
                   target.size()) &&
                 // overflow
                 (target_begin + (source_end - source_begin) >= target_begin),
               "Range is out of bounds.");

  auto target_end = target_begin + (source_end - source_begin);
  auto d_source = *column_device_view::create(source.parent(), stream);
  if (source.parent().nullable()) {
    return copy_range(
      d_source.begin<string_view>() + source_begin,
      cudf::experimental::detail::make_validity_iterator(d_source) +
        source_begin,
      target, target_begin, target_end,
      mr, stream);
  }
  else {
    return copy_range(
      d_source.begin<string_view>() + source_begin,
      thrust::make_constant_iterator(true),
      target, target_begin, target_end,
      mr, stream);
  }
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
