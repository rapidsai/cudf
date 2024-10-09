/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include "strings/count_matches.hpp"
#include "strings/regex/regex_program_impl.h"
#include "strings/regex/utilities.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/lists/detail/lists_column_factories.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/findall.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/pair.h>

namespace cudf {
namespace strings {
namespace detail {

using string_index_pair = thrust::pair<char const*, size_type>;

namespace {

/**
 * @brief This functor handles extracting matched strings by applying the compiled regex pattern
 * and creating string_index_pairs for all the substrings.
 */
struct findall_fn {
  column_device_view const d_strings;
  size_type const* d_offsets;
  string_index_pair* d_indices;

  __device__ void operator()(size_type const idx, reprog_device const prog, int32_t const prog_idx)
  {
    if (d_strings.is_null(idx)) { return; }
    auto const d_str  = d_strings.element<string_view>(idx);
    auto const nchars = d_str.length();

    auto d_output        = d_indices + d_offsets[idx];
    size_type output_idx = 0;

    auto itr = d_str.begin();
    while (itr.position() < nchars) {
      auto const match = prog.find(prog_idx, d_str, itr);
      if (!match) { break; }

      auto const d_result    = string_from_match(*match, d_str, itr);
      d_output[output_idx++] = string_index_pair{d_result.data(), d_result.size_bytes()};

      itr += (match->second - itr.position());
    }
  }
};

std::unique_ptr<column> findall_util(column_device_view const& d_strings,
                                     reprog_device& d_prog,
                                     int64_t total_matches,
                                     size_type const* d_offsets,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  rmm::device_uvector<string_index_pair> indices(total_matches, stream);

  launch_for_each_kernel(
    findall_fn{d_strings, d_offsets, indices.data()}, d_prog, d_strings.size(), stream);

  return make_strings_column(indices.begin(), indices.end(), stream, mr);
}

}  // namespace

//
std::unique_ptr<column> findall(strings_column_view const& input,
                                regex_program const& prog,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) {
    return cudf::lists::detail::make_empty_lists_column(input.parent().type(), stream, mr);
  }

  auto const d_strings = column_device_view::create(input.parent(), stream);

  // create device object from regex_program
  auto d_prog = regex_device_builder::create_prog_device(prog, stream);

  // Create lists offsets column
  auto const sizes              = count_matches(*d_strings, *d_prog, stream, mr);
  auto [offsets, total_matches] = cudf::detail::make_offsets_child_column(
    sizes->view().begin<size_type>(), sizes->view().end<size_type>(), stream, mr);
  auto const d_offsets = offsets->view().data<size_type>();

  // Build strings column of the matches
  auto strings_output = findall_util(*d_strings, *d_prog, total_matches, d_offsets, stream, mr);

  // Build the lists column from the offsets and the strings
  return make_lists_column(input.size(),
                           std::move(offsets),
                           std::move(strings_output),
                           input.null_count(),
                           cudf::detail::copy_bitmask(input.parent(), stream, mr),
                           stream,
                           mr);
}

namespace {
struct find_re_fn {
  column_device_view d_strings;

  __device__ size_type operator()(size_type const idx,
                                  reprog_device const prog,
                                  int32_t const thread_idx) const
  {
    if (d_strings.is_null(idx)) { return 0; }
    auto const d_str = d_strings.element<string_view>(idx);

    auto const result = prog.find(thread_idx, d_str, d_str.begin());
    return result.has_value() ? result.value().first : -1;
  }
};
}  // namespace

std::unique_ptr<column> find_re(strings_column_view const& input,
                                regex_program const& prog,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  auto results = make_numeric_column(data_type{type_to_id<size_type>()},
                                     input.size(),
                                     cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                     input.null_count(),
                                     stream,
                                     mr);
  if (input.is_empty()) { return results; }

  auto d_results       = results->mutable_view().data<size_type>();
  auto d_prog          = regex_device_builder::create_prog_device(prog, stream);
  auto const d_strings = column_device_view::create(input.parent(), stream);
  launch_transform_kernel(find_re_fn{*d_strings}, *d_prog, d_results, input.size(), stream);

  return results;
}
}  // namespace detail

// external API

std::unique_ptr<column> findall(strings_column_view const& input,
                                regex_program const& prog,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::findall(input, prog, stream, mr);
}

std::unique_ptr<column> find_re(strings_column_view const& input,
                                regex_program const& prog,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::find_re(input, prog, stream, mr);
}

}  // namespace strings
}  // namespace cudf
