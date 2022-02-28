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

#include <strings/regex/dispatcher.hpp>
#include <strings/regex/regex.cuh>
#include <strings/utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {

namespace {
/**
 * @brief This functor handles both contains_re and match_re to minimize the number
 * of regex calls to find() to be inlined greatly reducing compile time.
 */
template <int stack_size>
struct contains_fn {
  reprog_device prog;
  column_device_view const d_strings;
  bool const bmatch;  // do not make this a template parameter to keep compile times down

  __device__ bool operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) return false;
    auto const d_str = d_strings.element<string_view>(idx);
    int32_t begin    = 0;
    int32_t end      = bmatch ? 1    // match only the beginning of the string;
                              : -1;  // this handles empty strings too
    return static_cast<bool>(prog.find<stack_size>(idx, d_str, begin, end));
  }
};

struct contains_dispatch_fn {
  reprog_device d_prog;
  bool const beginning_only{false};

  template <int stack_size>
  std::unique_ptr<column> operator()(strings_column_view const& input,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    auto results = make_numeric_column(data_type{type_id::BOOL8},
                                       input.size(),
                                       cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                       input.null_count(),
                                       stream,
                                       mr);

    auto const d_strings = column_device_view::create(input.parent(), stream);
    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(input.size()),
                      results->mutable_view().data<bool>(),
                      contains_fn<stack_size>{d_prog, *d_strings, beginning_only});
    return results;
  }
};

}  // namespace

std::unique_ptr<column> contains_re(
  strings_column_view const& input,
  std::string const& pattern,
  regex_flags const flags,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto d_prog =
    reprog_device::create(pattern, flags, get_character_flags_table(), input.size(), stream);

  return regex_dispatcher(*d_prog, contains_dispatch_fn{*d_prog, false}, input, stream, mr);
}

std::unique_ptr<column> matches_re(
  strings_column_view const& input,
  std::string const& pattern,
  regex_flags const flags,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto d_prog =
    reprog_device::create(pattern, flags, get_character_flags_table(), input.size(), stream);

  return regex_dispatcher(*d_prog, contains_dispatch_fn{*d_prog, true}, input, stream, mr);
}

}  // namespace detail

// external APIs

std::unique_ptr<column> contains_re(strings_column_view const& strings,
                                    std::string const& pattern,
                                    regex_flags const flags,
                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::contains_re(strings, pattern, flags, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> matches_re(strings_column_view const& strings,
                                   std::string const& pattern,
                                   regex_flags const flags,
                                   rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::matches_re(strings, pattern, flags, rmm::cuda_stream_default, mr);
}

namespace detail {
namespace {
/**
 * @brief This counts the number of times the regex pattern matches in each string.
 */
template <int stack_size>
struct count_fn {
  reprog_device prog;
  column_device_view const d_strings;

  __device__ int32_t operator()(unsigned int idx)
  {
    if (d_strings.is_null(idx)) return 0;
    auto const d_str   = d_strings.element<string_view>(idx);
    auto const nchars  = d_str.length();
    int32_t find_count = 0;
    int32_t begin      = 0;
    while (begin < nchars) {
      auto end = static_cast<int32_t>(nchars);
      if (prog.find<stack_size>(idx, d_str, begin, end) <= 0) break;
      ++find_count;
      begin = end > begin ? end : begin + 1;
    }
    return find_count;
  }
};

struct count_dispatch_fn {
  reprog_device d_prog;

  template <int stack_size>
  std::unique_ptr<column> operator()(strings_column_view const& input,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    auto results = make_numeric_column(data_type{type_id::INT32},
                                       input.size(),
                                       cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                       input.null_count(),
                                       stream,
                                       mr);

    auto const d_strings = column_device_view::create(input.parent(), stream);
    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(input.size()),
                      results->mutable_view().data<int32_t>(),
                      count_fn<stack_size>{d_prog, *d_strings});
    return results;
  }
};

}  // namespace

std::unique_ptr<column> count_re(
  strings_column_view const& input,
  std::string const& pattern,
  regex_flags const flags,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  // compile regex into device object
  auto d_prog =
    reprog_device::create(pattern, flags, get_character_flags_table(), input.size(), stream);

  return regex_dispatcher(*d_prog, count_dispatch_fn{*d_prog}, input, stream, mr);
}

}  // namespace detail

// external API

std::unique_ptr<column> count_re(strings_column_view const& strings,
                                 std::string const& pattern,
                                 regex_flags const flags,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::count_re(strings, pattern, flags, rmm::cuda_stream_default, mr);
}

}  // namespace strings
}  // namespace cudf
