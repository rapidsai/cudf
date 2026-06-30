/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "strings/count_matches.hpp"
#include "strings/regex/regex_program_impl.h"
#include "strings/regex/utilities.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/memory_resource.hpp>

namespace cudf {
namespace strings {
namespace detail {

namespace {
/**
 * @brief Kernel counts the total matches for the given regex in each string.
 */
template <positional P>
struct count_fn {
  column_device_view const d_strings;

  template <typename ProgDevice>
  __device__ int32_t operator()(size_type const idx,
                                ProgDevice const prog,
                                int32_t const thread_idx)
  {
    if (d_strings.is_null(idx)) return 0;
    auto const d_str  = d_strings.element<string_view>(idx);
    auto const nchars = d_str.length();
    int32_t count     = 0;

    auto itr = d_str.begin();
    while (itr.position() <= nchars) {
      auto result = prog.template find<P>(thread_idx, d_str, itr);
      if (!result) { break; }
      ++count;
      // increment the iterator is faster than creating a new one
      // +1 if the match was on a virtual position (e.g. word boundary)
      itr += (result->second - itr.position()) + (result->first == result->second);
    }
    return count;
  }
};

}  // namespace

std::unique_ptr<column> count_matches(column_device_view const& d_strings,
                                      regex_program const& prog,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  auto results = make_numeric_column(
    data_type{type_to_id<size_type>()}, d_strings.size(), mask_state::UNALLOCATED, stream, mr);

  if (d_strings.size() == 0) { return results; }

  auto d_results = results->mutable_view().data<cudf::size_type>();

  if (regex_device_builder::glushkov_fast_path_supported(prog)) {
    auto d_prog = regex_device_builder::create_gkprog_device(prog, stream);
    launch_transform_kernel(
      count_fn<positional::BEGIN_END>{d_strings}, *d_prog, d_results, d_strings.size(), stream);
  } else {
    auto d_prog = regex_device_builder::create_prog_device(prog, stream);
    if (d_prog->is_empty_match_possible()) {
      launch_transform_kernel(
        count_fn<positional::BEGIN_END>{d_strings}, *d_prog, d_results, d_strings.size(), stream);
    } else {
      launch_transform_kernel(
        count_fn<positional::END_ONLY>{d_strings}, *d_prog, d_results, d_strings.size(), stream);
    }
  }

  return results;
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
