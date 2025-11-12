/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "split.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/lists/detail/lists_column_factories.hpp>
#include <cudf/strings/detail/split_utils.cuh>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/split/split.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/functional>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {

namespace {

template <typename Tokenizer, typename DelimiterFn>
std::unique_ptr<column> split_record_fn(strings_column_view const& input,
                                        Tokenizer tokenizer,
                                        DelimiterFn delimiter_fn,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) {
    return cudf::lists::detail::make_empty_lists_column(data_type{type_id::STRING}, stream, mr);
  }
  if (input.size() == input.null_count()) {
    auto offsets = std::make_unique<column>(input.offsets(), stream, mr);
    auto results = make_empty_column(type_id::STRING);
    return make_lists_column(input.size(),
                             std::move(offsets),
                             std::move(results),
                             input.null_count(),
                             cudf::detail::copy_bitmask(input.parent(), stream, mr),
                             stream,
                             mr);
  }

  // builds the offsets and the vector of all tokens
  auto [offsets, tokens] = split_helper(input, tokenizer, delimiter_fn, stream, mr);
  CUDF_EXPECTS(tokens.size() < static_cast<std::size_t>(std::numeric_limits<size_type>::max()),
               "Size of output exceeds the column size limit",
               std::overflow_error);

  // build a strings column from the tokens
  auto strings_child = make_strings_column(tokens.begin(), tokens.end(), stream, mr);

  return make_lists_column(input.size(),
                           std::move(offsets),
                           std::move(strings_child),
                           input.null_count(),
                           cudf::detail::copy_bitmask(input.parent(), stream, mr),
                           stream,
                           mr);
}

}  // namespace

std::unique_ptr<column> split_record(strings_column_view const& input,
                                     string_scalar const& delimiter,
                                     size_type maxsplit,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(delimiter.is_valid(stream), "Parameter delimiter must be valid");

  // makes consistent with Pandas
  size_type max_tokens = maxsplit > 0 ? maxsplit + 1 : std::numeric_limits<size_type>::max();

  auto d_strings = column_device_view::create(input.parent(), stream);
  if (delimiter.size() == 0) {
    auto tokenizer    = split_ws_tokenizer_fn{*d_strings, max_tokens};
    auto delimiter_fn = whitespace_delimiter_fn{};
    return split_record_fn(input, tokenizer, delimiter_fn, stream, mr);
  }

  auto tokenizer    = split_tokenizer_fn{*d_strings, delimiter.size(), max_tokens};
  auto delimiter_fn = string_delimiter_fn{delimiter.value(stream)};
  return split_record_fn(input, tokenizer, delimiter_fn, stream, mr);
}

std::unique_ptr<column> rsplit_record(strings_column_view const& input,
                                      string_scalar const& delimiter,
                                      size_type maxsplit,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(delimiter.is_valid(stream), "Parameter delimiter must be valid");

  // makes consistent with Pandas
  size_type max_tokens = maxsplit > 0 ? maxsplit + 1 : std::numeric_limits<size_type>::max();

  auto d_strings = column_device_view::create(input.parent(), stream);
  if (delimiter.size() == 0) {
    auto tokenizer    = rsplit_ws_tokenizer_fn{*d_strings, max_tokens};
    auto delimiter_fn = whitespace_delimiter_fn{};
    return split_record_fn(input, tokenizer, delimiter_fn, stream, mr);
  }

  auto tokenizer    = rsplit_tokenizer_fn{*d_strings, delimiter.size(), max_tokens};
  auto delimiter_fn = string_delimiter_fn{delimiter.value(stream)};
  return split_record_fn(input, tokenizer, delimiter_fn, stream, mr);
}

}  // namespace detail

// external APIs

std::unique_ptr<column> split_record(strings_column_view const& input,
                                     string_scalar const& delimiter,
                                     size_type maxsplit,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::split_record(input, delimiter, maxsplit, stream, mr);
}

std::unique_ptr<column> rsplit_record(strings_column_view const& input,
                                      string_scalar const& delimiter,
                                      size_type maxsplit,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::rsplit_record(input, delimiter, maxsplit, stream, mr);
}

}  // namespace strings
}  // namespace cudf
