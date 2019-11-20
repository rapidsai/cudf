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

#include "../utilities.cuh"
#include "../utilities.hpp"

#include <cudf/types.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/detail/copy_range.hpp>

#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
//#include <thrust/scatter.h>
//#include <thrust/transform_scan.h>

namespace cudf {
namespace strings {
namespace detail {

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

  if (source_end != source_begin) {
    auto target_end = target_begin + (source_end - source_begin);
    auto&& d_source = *column_device_view::create(source.parent(), stream);
    auto&& d_target = *column_device_view::create(target.parent(), stream);

    rmm::device_buffer null_mask{};
    size_type null_count = 0;
    if (target.parent().nullable() ||
        // TODO: shouldn't this be
        // source.parent().has_nulls(source_begin, source_end)?
        source.parent().has_nulls()) {
      // create resulting null mask

      auto valid_mask = cudf::experimental::detail::valid_if(
        thrust::make_counting_iterator<size_type>(0),
        thrust::make_counting_iterator<size_type>(target.size()),
        [d_source, d_target, source_begin, target_begin, target_end] __device__
            (size_type idx) {
          return (idx >= target_begin && idx < target_end) ?
            d_source.is_valid(idx + (source_begin - target_begin)) :
            d_target.is_valid(idx);
        },
        stream, mr);
      null_mask = std::move(valid_mask.first);
      null_count = valid_mask.second;
    }

    // build offsets column

    auto string_size_begin =
      thrust::make_transform_iterator(
        thrust::make_counting_iterator(0),
        [d_source, d_target, source_begin, target_begin, target_end] __device__
            (size_type idx) {
          if (idx >= target_begin && idx < target_end) {
            auto source_idx = idx + (source_begin - target_begin);
            return d_source.is_valid(source_idx) ?
              d_source.element<string_view>(source_idx).size_bytes() : 0;
          }
          else {
            return d_target.is_valid(idx) ?
              d_target.element<string_view>(idx).size_bytes() : 0;
          }
      });
    auto p_offsets_column =
        detail::make_offsets_child_column(
          string_size_begin, string_size_begin + target.size(), mr, stream);

    // create the chars column

    auto p_offsets =
      thrust::device_pointer_cast(p_offsets_column->view().data<size_type>());
    auto chars_bytes = p_offsets[target.size()];

    auto p_chars_column =
      strings::detail::create_chars_child_column(
        target.size(), null_count, chars_bytes, mr, stream);

    // copy to the chars column

    auto p_chars = p_chars_column->mutable_view().data<char>();
    thrust::for_each(
      rmm::exec_policy(stream)->on(stream),
      thrust::make_counting_iterator(0),
      thrust::make_counting_iterator(target.size()),
      [d_source, d_target, p_chars, p_offsets, source_begin, target_begin,
          target_end] __device__ (size_type idx) {
        auto source_idx = idx + (source_begin - target_begin);
        const auto source =
          (idx >= target_begin && idx < target_end) ?
            d_source.element<string_view>(source_idx) :
            d_target.element<string_view>(idx);
        memcpy(p_chars + p_offsets[idx], source.data(), source.size_bytes());
    });

    return make_strings_column(
      target.size(), std::move(p_offsets_column), std::move(p_chars_column),
      null_count, std::move(null_mask), stream, mr);
  }
  else {
    return std::make_unique<column>(target.parent(), stream, mr);
  }
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
