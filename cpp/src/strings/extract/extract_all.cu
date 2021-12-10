/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

#include <strings/regex/regex.cuh>
#include <strings/utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/extract.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace strings {
namespace detail {

namespace {

using string_index_pair = thrust::pair<const char*, size_type>;

template <int stack_size>
struct extract_count_fn {
  reprog_device prog;
  column_device_view const d_strings;

  __device__ size_type operator()(size_type idx)
  {
    size_type count = 0;
    if (d_strings.is_null(idx)) { return count; }
    auto const d_str = d_strings.element<string_view>(idx);

    // find and count all matching patterns
    int32_t begin = 0;
    int32_t end   = d_str.length();
    while ((begin < end) && prog.find<stack_size>(idx, d_str, begin, end) > 0) {
      count += prog.group_counts();
      begin = end;
      end   = d_str.length();
    }
    return count;
  };
};

template <int stack_size>
struct extract_fn {
  reprog_device prog;
  column_device_view const d_strings;
  offset_type const* d_offsets;
  // cudf::detail::device_2dspan<string_index_pair> d_indices;
  string_index_pair* d_indices;

  __device__ void operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) { return; }

    auto const groups    = prog.group_counts();
    auto d_output        = d_indices + d_offsets[idx];
    size_type output_idx = 0;

    auto const d_str = d_strings.element<string_view>(idx);
    int32_t begin    = 0;
    int32_t end      = d_str.length();
    while ((begin < end) && prog.find<stack_size>(idx, d_str, begin, end) > 0) {
      for (auto col_idx = 0; col_idx < groups; ++col_idx) {
        auto const extracted = prog.extract<stack_size>(idx, d_str, begin, end, col_idx);

        d_output[col_idx + output_idx] = [&] {
          if (!extracted) return string_index_pair{nullptr, 0};
          auto const offset = d_str.byte_offset((*extracted).first);
          return string_index_pair{d_str.data() + offset,
                                   d_str.byte_offset((*extracted).second) - offset};
        }();
      }
      begin = end;
      end   = d_str.length();
      output_idx += groups;
    }
  }
};
}  // namespace

//
std::unique_ptr<column> extract_all(
  strings_column_view const& strings,
  std::string const& pattern,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto const strings_count  = strings.size();
  auto const strings_column = column_device_view::create(strings.parent(), stream);
  auto const d_strings      = *strings_column;

  // compile regex into device object
  auto prog   = reprog_device::create(pattern, get_character_flags_table(), strings_count, stream);
  auto d_prog = *prog;
  // an extract pattern should always include groups
  auto const groups = d_prog.group_counts();
  CUDF_EXPECTS(groups > 0, "Group indicators not found in regex pattern");

  // Create lists offsets column
  auto offsets = make_numeric_column(
    data_type{type_id::INT32}, strings_count + 1, mask_state::UNALLOCATED, stream, mr);
  auto d_offsets = offsets->mutable_view().data<offset_type>();

  // Compute the number of extracted groups per string
  auto const regex_insts = d_prog.insts_counts();
  if (regex_insts <= RX_SMALL_INSTS) {
    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(strings_count),
                      d_offsets,
                      extract_count_fn<RX_STACK_SMALL>{d_prog, d_strings});
  } else if (regex_insts <= RX_MEDIUM_INSTS) {
    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(strings_count),
                      d_offsets,
                      extract_count_fn<RX_STACK_MEDIUM>{d_prog, d_strings});
    thrust::for_each_n(rmm::exec_policy(stream),
                       thrust::make_counting_iterator<size_type>(0),
                       strings_count,
                       extract_count_fn<RX_STACK_MEDIUM>{d_prog, d_strings});
  } else if (regex_insts <= RX_LARGE_INSTS) {
    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(strings_count),
                      d_offsets,
                      extract_count_fn<RX_STACK_LARGE>{d_prog, d_strings});
  } else {
    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(strings_count),
                      d_offsets,
                      extract_count_fn<RX_STACK_ANY>{d_prog, d_strings});
  }

  // Compute null output rows
  auto [null_mask, null_count] = cudf::detail::valid_if(
    d_offsets, d_offsets + strings_count, [] __device__(auto v) { return v > 0; }, stream, mr);

  auto const valid_count = strings_count - null_count;

  // Return an empty lists column if there are no valid rows
  if (valid_count == 0) {
    return make_lists_column(0,
                             make_empty_column(type_to_id<offset_type>()),
                             make_empty_column(type_id::STRING),
                             0,
                             rmm::device_buffer{},
                             stream,
                             mr);
  }

  // Convert sizes into offsets
  thrust::exclusive_scan(
    rmm::exec_policy(stream), d_offsets, d_offsets + strings_count + 1, d_offsets);

  // Create indices vector with the total number of groups that will be extracted
  auto total_groups = cudf::detail::get_value<size_type>(offsets->view(), strings_count, stream);

  rmm::device_uvector<string_index_pair> indices(total_groups, stream);
  auto d_indices = indices.data();

  // Call the extract functor to fill in the indices vector
  if (regex_insts <= RX_SMALL_INSTS) {
    thrust::for_each_n(rmm::exec_policy(stream),
                       thrust::make_counting_iterator<size_type>(0),
                       strings_count,
                       extract_fn<RX_STACK_SMALL>{d_prog, d_strings, d_offsets, d_indices});
  } else if (regex_insts <= RX_MEDIUM_INSTS) {
    thrust::for_each_n(rmm::exec_policy(stream),
                       thrust::make_counting_iterator<size_type>(0),
                       strings_count,
                       extract_fn<RX_STACK_MEDIUM>{d_prog, d_strings, d_offsets, d_indices});
  } else if (regex_insts <= RX_LARGE_INSTS) {
    thrust::for_each_n(rmm::exec_policy(stream),
                       thrust::make_counting_iterator<size_type>(0),
                       strings_count,
                       extract_fn<RX_STACK_LARGE>{d_prog, d_strings, d_offsets, d_indices});
  } else {
    thrust::for_each_n(rmm::exec_policy(stream),
                       thrust::make_counting_iterator<size_type>(0),
                       strings_count,
                       extract_fn<RX_STACK_ANY>{d_prog, d_strings, d_offsets, d_indices});
  }

  // Build the child strings column from the resulting indices
  auto strings_output = make_strings_column(indices.begin(), indices.end(), stream, mr);

  // Build the lists column from the offsets and the strings
  return make_lists_column(strings_count,
                           std::move(offsets),
                           std::move(strings_output),
                           null_count,
                           std::move(null_mask),
                           stream,
                           mr);
}

}  // namespace detail

// external API

std::unique_ptr<column> extract_all(strings_column_view const& strings,
                                    std::string const& pattern,
                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::extract_all(strings, pattern, rmm::cuda_stream_default, mr);
}

}  // namespace strings
}  // namespace cudf
