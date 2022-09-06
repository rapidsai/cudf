/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/findall.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/pair.h>
#include <thrust/scan.h>

namespace cudf {
namespace strings {
namespace detail {

using string_index_pair = thrust::pair<const char*, size_type>;

namespace {

/**
 * @brief This functor handles extracting matched strings by applying the compiled regex pattern
 * and creating string_index_pairs for all the substrings.
 */
struct findall_fn {
  column_device_view const d_strings;
  offset_type const* d_offsets;
  string_index_pair* d_indices;

  __device__ void operator()(size_type const idx, reprog_device const prog, int32_t const prog_idx)
  {
    if (d_strings.is_null(idx)) { return; }
    auto const d_str  = d_strings.element<string_view>(idx);
    auto const nchars = d_str.length();

    auto d_output        = d_indices + d_offsets[idx];
    size_type output_idx = 0;

    size_type begin = 0;
    size_type end   = nchars;
    while ((begin < end) && (prog.find(prog_idx, d_str, begin, end) > 0)) {
      auto const spos = d_str.byte_offset(begin);  // convert
      auto const epos = d_str.byte_offset(end);    // to bytes

      d_output[output_idx++] = string_index_pair{d_str.data() + spos, (epos - spos)};

      begin = end + (begin == end);
      end   = nchars;
    }
  }
};

std::unique_ptr<column> findall_util(column_device_view const& d_strings,
                                     reprog_device& d_prog,
                                     size_type total_matches,
                                     offset_type const* d_offsets,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
{
  rmm::device_uvector<string_index_pair> indices(total_matches, stream);

  launch_for_each_kernel(
    findall_fn{d_strings, d_offsets, indices.data()}, d_prog, d_strings.size(), stream);

  return make_strings_column(indices.begin(), indices.end(), stream, mr);
}

}  // namespace

//
std::unique_ptr<column> findall(
  strings_column_view const& input,
  std::string_view pattern,
  regex_flags const flags,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto const strings_count = input.size();
  auto const d_strings     = column_device_view::create(input.parent(), stream);

  // compile regex into device object
  auto const d_prog = reprog_device::create(pattern, flags, stream);

  // Create lists offsets column
  auto offsets   = count_matches(*d_strings, *d_prog, strings_count + 1, stream, mr);
  auto d_offsets = offsets->mutable_view().data<offset_type>();

  // Convert counts into offsets
  thrust::exclusive_scan(
    rmm::exec_policy(stream), d_offsets, d_offsets + strings_count + 1, d_offsets);

  // Create indices vector with the total number of groups that will be extracted
  auto const total_matches =
    cudf::detail::get_value<size_type>(offsets->view(), strings_count, stream);

  auto strings_output = findall_util(*d_strings, *d_prog, total_matches, d_offsets, stream, mr);

  // Build the lists column from the offsets and the strings
  return make_lists_column(strings_count,
                           std::move(offsets),
                           std::move(strings_output),
                           input.null_count(),
                           cudf::detail::copy_bitmask(input.parent(), stream, mr),
                           stream,
                           mr);
}

}  // namespace detail

// external API

std::unique_ptr<column> findall(strings_column_view const& input,
                                std::string_view pattern,
                                regex_flags const flags,
                                rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::findall(input, pattern, flags, cudf::default_stream_value, mr);
}

}  // namespace strings
}  // namespace cudf
