/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "strings/count_matches.hpp"
#include "strings/regex/utilities.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/memory_resource.hpp>

namespace cudf {
namespace strings {
namespace detail {

namespace {
/**
 * @brief Kernel counts the total matches for the given regex in each string.
 */
struct count_fn {
  column_device_view const d_strings;

  __device__ int32_t operator()(size_type const idx,
                                reprog_device const prog,
                                int32_t const thread_idx)
  {
    if (d_strings.is_null(idx)) return 0;
    auto const d_str  = d_strings.element<string_view>(idx);
    auto const nchars = d_str.length();
    int32_t count     = 0;

    auto itr = d_str.begin();
    while (itr.position() <= nchars) {
      auto result = prog.find(thread_idx, d_str, itr);
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
                                      reprog_device& d_prog,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  auto results = make_numeric_column(
    data_type{type_to_id<size_type>()}, d_strings.size(), mask_state::UNALLOCATED, stream, mr);

  if (d_strings.size() == 0) { return results; }

  auto d_results = results->mutable_view().data<cudf::size_type>();

  launch_transform_kernel(count_fn{d_strings}, d_prog, d_results, d_strings.size(), stream);

  return results;
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
