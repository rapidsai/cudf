/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

namespace {
template <bool source_has_nulls,
          bool target_has_nulls,
          typename SourceValueIterator,
          typename SourceValidityIterator>
struct compute_element_size {
  SourceValueIterator source_value_begin;
  SourceValidityIterator source_validity_begin;
  cudf::column_device_view d_target;
  cudf::size_type target_begin;
  cudf::size_type target_end;

  __device__ cudf::size_type operator()(cudf::size_type idx)
  {
    if (idx >= target_begin && idx < target_end) {
      if (source_has_nulls) {
        return *(source_validity_begin + (idx - target_begin))
                 ? (*(source_value_begin + (idx - target_begin))).size_bytes()
                 : 0;
      } else {
        return (*(source_value_begin + (idx - target_begin))).size_bytes();
      }
    } else {
      if (target_has_nulls) {
        return d_target.is_valid_nocheck(idx)
                 ? d_target.element<cudf::string_view>(idx).size_bytes()
                 : 0;
      } else {
        return d_target.element<cudf::string_view>(idx).size_bytes();
      }
    }
  }
};

}  // namespace

namespace cudf {
namespace strings {
namespace detail {
/**
 * @brief Internal API to copy a range of string elements out-of-place from
 * source iterators to a target column.
 *
 * Creates a new column as if an in-place copy was performed into @p target.
 * The elements indicated by the indices [@p target_begin, @p target_end) were
 * replaced with the elements retrieved from source iterators;
 * *(@p source_value_begin + idx) if *(@p source_validity_begin + idx) is true,
 * invalidate otherwise (where idx = [0, @p target_end - @p target_begin)).
 * Elements outside the range are copied from @p target into the new target
 * column to return.
 *
 * @throws cudf::logic_error for invalid range (if @p target_begin < 0,
 * target_begin >= @p target.size(), or @p target_end > @p target.size()).
 *
 * @tparam SourceValueIterator Iterator for retrieving source values
 * @tparam SourceValidityIterator Iterator for retrieving source validities
 * @param source_value_begin Start of source value iterator
 * @param source_validity_begin Start of source validity iterator
 * @param target The strings column to copy from outside the range.
 * @param target_begin The starting index of the target range (inclusive)
 * @param target_end The index of the last element in the target range
 * (exclusive)
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return std::unique_ptr<column> The result target column
 */
template <typename SourceValueIterator, typename SourceValidityIterator>
std::unique_ptr<column> copy_range(SourceValueIterator source_value_begin,
                                   SourceValidityIterator source_validity_begin,
                                   strings_column_view const& target,
                                   size_type target_begin,
                                   size_type target_end,
                                   rmm::cuda_stream_view stream,
                                   rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(
    (target_begin >= 0) && (target_begin < target.size()) && (target_end <= target.size()),
    "Range is out of bounds.");

  if (target_end == target_begin) {
    return std::make_unique<column>(target.parent(), stream, mr);
  } else {
    auto p_target_device_view = column_device_view::create(target.parent(), stream);
    auto d_target             = *p_target_device_view;

    // create resulting null mask

    std::pair<rmm::device_buffer, size_type> valid_mask{};
    if (target.has_nulls()) {  // check validities for both source & target
      valid_mask = cudf::detail::valid_if(
        thrust::make_counting_iterator<size_type>(0),
        thrust::make_counting_iterator<size_type>(target.size()),
        [source_validity_begin, d_target, target_begin, target_end] __device__(size_type idx) {
          return (idx >= target_begin && idx < target_end)
                   ? *(source_validity_begin + (idx - target_begin))
                   : d_target.is_valid_nocheck(idx);
        },
        stream,
        mr);
    } else {  // check validities for source only
      valid_mask = cudf::detail::valid_if(
        thrust::make_counting_iterator<size_type>(0),
        thrust::make_counting_iterator<size_type>(target.size()),
        [source_validity_begin, d_target, target_begin, target_end] __device__(size_type idx) {
          return (idx >= target_begin && idx < target_end)
                   ? *(source_validity_begin + (idx - target_begin))
                   : true;
        },
        stream,
        mr);
    }

    auto null_count = valid_mask.second;
    rmm::device_buffer null_mask{0, stream, mr};
    if (target.parent().nullable() || null_count > 0) { null_mask = std::move(valid_mask.first); }

    // build offsets column

    std::unique_ptr<column> p_offsets_column{nullptr};
    size_type chars_bytes = 0;
    if (target.has_nulls()) {  // check validities for both source & target
      auto string_size_begin = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0),
        compute_element_size<true, true, SourceValueIterator, SourceValidityIterator>{
          source_value_begin, source_validity_begin, d_target, target_begin, target_end});

      std::tie(p_offsets_column, chars_bytes) = cudf::detail::make_offsets_child_column(
        string_size_begin, string_size_begin + target.size(), stream, mr);
    } else if (null_count > 0) {  // check validities for source only
      auto string_size_begin = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0),
        compute_element_size<true, false, SourceValueIterator, SourceValidityIterator>{
          source_value_begin, source_validity_begin, d_target, target_begin, target_end});

      std::tie(p_offsets_column, chars_bytes) = cudf::detail::make_offsets_child_column(
        string_size_begin, string_size_begin + target.size(), stream, mr);
    } else {  // no need to check validities
      auto string_size_begin = thrust::make_transform_iterator(
        thrust::make_counting_iterator(0),
        compute_element_size<false, false, SourceValueIterator, SourceValidityIterator>{
          source_value_begin, source_validity_begin, d_target, target_begin, target_end});

      std::tie(p_offsets_column, chars_bytes) = cudf::detail::make_offsets_child_column(
        string_size_begin, string_size_begin + target.size(), stream, mr);
    }

    // create the chars column

    auto p_offsets =
      thrust::device_pointer_cast(p_offsets_column->view().template data<size_type>());
    auto p_chars_column = strings::detail::create_chars_child_column(chars_bytes, stream, mr);

    // copy to the chars column

    auto p_chars = (p_chars_column->mutable_view()).template data<char>();
    thrust::for_each(rmm::exec_policy(stream),
                     thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(target.size()),
                     [source_value_begin,
                      source_validity_begin,
                      d_target,
                      target_begin,
                      target_end,
                      p_offsets,
                      p_chars] __device__(size_type idx) {
                       if (p_offsets[idx + 1] - p_offsets[idx] > 0) {
                         const auto source = (idx >= target_begin && idx < target_end)
                                               ? *(source_value_begin + (idx - target_begin))
                                               : d_target.element<string_view>(idx);
                         memcpy(p_chars + p_offsets[idx], source.data(), source.size_bytes());
                       }
                     });

    return make_strings_column(target.size(),
                               std::move(p_offsets_column),
                               std::move(p_chars_column),
                               null_count,
                               std::move(null_mask));
  }
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
