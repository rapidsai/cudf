/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "strings/count_matches.hpp"
#include "strings/regex/regex_program_impl.h"
#include "strings/regex/utilities.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace strings {
namespace detail {

namespace {
/**
 * @brief This functor handles both contains_re and match_re to regex-match a pattern
 * to each string in a column.
 */
struct contains_fn {
  column_device_view const d_strings;
  bool const beginning_only;

  __device__ bool operator()(size_type const idx,
                             reprog_device const prog,
                             int32_t const thread_idx)
  {
    if (d_strings.is_null(idx)) return false;
    auto const d_str = d_strings.element<string_view>(idx);

    size_type end = beginning_only ? 1    // match only the beginning of the string;
                                   : -1;  // match anywhere in the string
    return prog.find<positional::END_ONLY>(thread_idx, d_str, d_str.begin(), end).has_value();
  }
};

std::unique_ptr<column> contains_impl(strings_column_view const& input,
                                      regex_program const& prog,
                                      bool const beginning_only,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  auto results = make_numeric_column(data_type{type_id::BOOL8},
                                     input.size(),
                                     cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                     input.null_count(),
                                     stream,
                                     mr);
  if (input.is_empty()) { return results; }

  auto d_prog = regex_device_builder::create_prog_device(prog, stream);

  auto d_results       = results->mutable_view().data<bool>();
  auto const d_strings = column_device_view::create(input.parent(), stream);

  launch_transform_kernel(
    contains_fn{*d_strings, beginning_only}, *d_prog, d_results, input.size(), stream);

  results->set_null_count(input.null_count());

  return results;
}

}  // namespace

std::unique_ptr<column> contains_re(strings_column_view const& input,
                                    regex_program const& prog,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  return contains_impl(input, prog, false, stream, mr);
}

std::unique_ptr<column> matches_re(strings_column_view const& input,
                                   regex_program const& prog,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  return contains_impl(input, prog, true, stream, mr);
}

std::unique_ptr<column> count_re(strings_column_view const& input,
                                 regex_program const& prog,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  // create device object from regex_program
  auto d_prog = regex_device_builder::create_prog_device(prog, stream);

  auto const d_strings = column_device_view::create(input.parent(), stream);

  auto result = count_matches(*d_strings, *d_prog, stream, mr);
  if (input.has_nulls()) {
    result->set_null_mask(cudf::detail::copy_bitmask(input.parent(), stream, mr),
                          input.null_count());
  }
  return result;
}

}  // namespace detail

// external APIs

std::unique_ptr<column> contains_re(strings_column_view const& input,
                                    regex_program const& prog,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::contains_re(input, prog, stream, mr);
}

std::unique_ptr<column> matches_re(strings_column_view const& input,
                                   regex_program const& prog,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::matches_re(input, prog, stream, mr);
}

std::unique_ptr<column> count_re(strings_column_view const& input,
                                 regex_program const& prog,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::count_re(input, prog, stream, mr);
}

}  // namespace strings
}  // namespace cudf
