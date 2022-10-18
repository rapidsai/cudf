/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <strings/count_matches.hpp>
#include <strings/regex/utilities.cuh>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/extract.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/functional.h>
#include <thrust/transform_scan.h>

namespace cudf {
namespace strings {
namespace detail {

namespace {

/**
 * @brief Functor extracts matched string pointers for each input string.
 *
 * For regex match within a string, the specified groups are extracted into
 * the `d_indices` output vector.
 * The `d_offsets` are pre-computed to identify the location of where each
 * string's output groups are to be written.
 */
struct extract_fn {
  column_device_view const d_strings;
  offset_type const* d_offsets;
  string_index_pair* d_indices;

  __device__ void operator()(size_type const idx,
                             reprog_device const d_prog,
                             int32_t const prog_idx)
  {
    if (d_strings.is_null(idx)) { return; }

    auto const groups    = d_prog.group_counts();
    auto d_output        = d_indices + d_offsets[idx];
    size_type output_idx = 0;

    auto const d_str  = d_strings.element<string_view>(idx);
    auto const nchars = d_str.length();

    size_type begin = 0;
    size_type end   = nchars;
    // match the regex
    while ((begin < end) && d_prog.find(prog_idx, d_str, begin, end) > 0) {
      // extract each group into the output
      for (auto group_idx = 0; group_idx < groups; ++group_idx) {
        // result is an optional containing the bounds of the extracted string at group_idx
        auto const extracted = d_prog.extract(prog_idx, d_str, begin, end, group_idx);

        d_output[group_idx + output_idx] = [&] {
          if (!extracted) { return string_index_pair{nullptr, 0}; }
          auto const start_offset = d_str.byte_offset(extracted->first);
          auto const end_offset   = d_str.byte_offset(extracted->second);
          return string_index_pair{d_str.data() + start_offset, end_offset - start_offset};
        }();
      }
      // continue to next match
      begin = end;
      end   = nchars;
      output_idx += groups;
    }
  }
};

}  // namespace

/**
 * @copydoc cudf::strings::extract_all_record
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> extract_all_record(
  strings_column_view const& input,
  std::string_view pattern,
  regex_flags const flags,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto const strings_count = input.size();
  auto const d_strings     = column_device_view::create(input.parent(), stream);

  // Compile regex into device object.
  auto d_prog = reprog_device::create(pattern, flags, capture_groups::EXTRACT, stream);
  // The extract pattern should always include groups.
  auto const groups = d_prog->group_counts();
  CUDF_EXPECTS(groups > 0, "extract_all requires group indicators in the regex pattern.");

  // Get the match counts for each string.
  // This column will become the output lists child offsets column.
  auto offsets   = count_matches(*d_strings, *d_prog, strings_count + 1, stream, mr);
  auto d_offsets = offsets->mutable_view().data<offset_type>();

  // Compute null output rows
  auto [null_mask, null_count] = cudf::detail::valid_if(
    d_offsets, d_offsets + strings_count, [] __device__(auto v) { return v > 0; }, stream, mr);

  // Return an empty lists column if there are no valid rows
  if (strings_count == null_count) {
    return make_lists_column(0,
                             make_empty_column(type_to_id<offset_type>()),
                             make_empty_column(type_id::STRING),
                             0,
                             rmm::device_buffer{},
                             stream,
                             mr);
  }

  // Convert counts into offsets.
  // Multiply each count by the number of groups.
  thrust::transform_exclusive_scan(
    rmm::exec_policy(stream),
    d_offsets,
    d_offsets + strings_count + 1,
    d_offsets,
    [groups] __device__(auto v) { return v * groups; },
    offset_type{0},
    thrust::plus{});
  auto const total_groups =
    cudf::detail::get_value<offset_type>(offsets->view(), strings_count, stream);

  rmm::device_uvector<string_index_pair> indices(total_groups, stream);

  launch_for_each_kernel(
    extract_fn{*d_strings, d_offsets, indices.data()}, *d_prog, strings_count, stream);

  auto strings_output = make_strings_column(indices.begin(), indices.end(), stream, mr);

  // Build the lists column from the offsets and the strings.
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

std::unique_ptr<column> extract_all_record(strings_column_view const& strings,
                                           std::string_view pattern,
                                           regex_flags const flags,
                                           rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::extract_all_record(strings, pattern, flags, cudf::default_stream_value, mr);
}

}  // namespace strings
}  // namespace cudf
