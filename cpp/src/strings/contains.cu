/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <strings/regex/regex.cuh>
#include <strings/utilities.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

namespace cudf {
namespace strings {
namespace detail {
namespace {
/**
 * @brief This functor handles both contains_re and match_re to minimize the number
 * of regex calls to find() to be inlined greatly reducing compile time.
 *
 * The stack is used to keep progress on evaluating the regex instructions on each string.
 * So the size of the stack is in proportion to the number of instructions in the given regex
 * pattern.
 *
 * There are three call types based on the number of regex instructions in the given pattern.
 * Small to medium instruction lengths can use the stack effectively though smaller executes faster.
 * Longer patterns require global memory.
 *
 */
template <size_t stack_size>
struct contains_fn {
  reprog_device prog;
  column_device_view d_strings;
  bool bmatch{false};  // do not make this a template parameter to keep compile times down

  __device__ bool operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) return 0;
    u_char data1[stack_size], data2[stack_size];
    prog.set_stack_mem(data1, data2);
    string_view d_str = d_strings.element<string_view>(idx);
    int32_t begin     = 0;
    int32_t end       = bmatch ? 1  // match only the beginning of the string;
                         : -1;      // this handles empty strings too
    return static_cast<bool>(prog.find(idx, d_str, begin, end));
  }
};

//
std::unique_ptr<column> contains_util(
  strings_column_view const& strings,
  std::string const& pattern,
  bool beginning_only                 = false,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto strings_count  = strings.size();
  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_column       = *strings_column;

  // compile regex into device object
  auto prog   = reprog_device::create(pattern, get_character_flags_table(), strings_count, stream);
  auto d_prog = *prog;

  // create the output column
  auto results   = make_numeric_column(data_type{type_id::BOOL8},
                                     strings_count,
                                     cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                                     strings.null_count(),
                                     stream,
                                     mr);
  auto d_results = results->mutable_view().data<bool>();

  // fill the output column
  int regex_insts = d_prog.insts_counts();
  if ((regex_insts > MAX_STACK_INSTS) || (regex_insts <= RX_SMALL_INSTS))
    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(strings_count),
                      d_results,
                      contains_fn<RX_STACK_SMALL>{d_prog, d_column, beginning_only});
  else if (regex_insts <= RX_MEDIUM_INSTS)
    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(strings_count),
                      d_results,
                      contains_fn<RX_STACK_MEDIUM>{d_prog, d_column, beginning_only});
  else
    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(strings_count),
                      d_results,
                      contains_fn<RX_STACK_LARGE>{d_prog, d_column, beginning_only});

  results->set_null_count(strings.null_count());
  return results;
}

}  // namespace

std::unique_ptr<column> contains_re(
  strings_column_view const& strings,
  std::string const& pattern,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  return contains_util(strings, pattern, false, stream, mr);
}

std::unique_ptr<column> matches_re(
  strings_column_view const& strings,
  std::string const& pattern,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  return contains_util(strings, pattern, true, stream, mr);
}

}  // namespace detail

// external APIs

std::unique_ptr<column> contains_re(strings_column_view const& strings,
                                    std::string const& pattern,
                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::contains_re(strings, pattern, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> matches_re(strings_column_view const& strings,
                                   std::string const& pattern,
                                   rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::matches_re(strings, pattern, rmm::cuda_stream_default, mr);
}

namespace detail {
namespace {
/**
 * @brief This counts the number of times the regex pattern matches in each string.
 *
 */
template <size_t stack_size>
struct count_fn {
  reprog_device prog;
  column_device_view d_strings;

  __device__ int32_t operator()(unsigned int idx)
  {
    u_char data1[stack_size], data2[stack_size];
    prog.set_stack_mem(data1, data2);
    if (d_strings.is_null(idx)) return 0;
    string_view d_str  = d_strings.element<string_view>(idx);
    int32_t find_count = 0;
    size_type nchars   = d_str.length();
    size_type begin    = 0;
    while (begin <= nchars) {
      auto end = nchars;
      if (prog.find(idx, d_str, begin, end) <= 0) break;
      ++find_count;
      begin = end > begin ? end : begin + 1;
    }
    return find_count;
  }
};

}  // namespace

std::unique_ptr<column> count_re(
  strings_column_view const& strings,
  std::string const& pattern,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto strings_count  = strings.size();
  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_column       = *strings_column;

  // compile regex into device object
  auto prog   = reprog_device::create(pattern, get_character_flags_table(), strings_count, stream);
  auto d_prog = *prog;

  // create the output column
  auto results   = make_numeric_column(data_type{type_id::INT32},
                                     strings_count,
                                     cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                                     strings.null_count(),
                                     stream,
                                     mr);
  auto d_results = results->mutable_view().data<int32_t>();

  // fill the output column
  int regex_insts = d_prog.insts_counts();
  if ((regex_insts > MAX_STACK_INSTS) || (regex_insts <= RX_SMALL_INSTS))
    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(strings_count),
                      d_results,
                      count_fn<RX_STACK_SMALL>{d_prog, d_column});
  else if (regex_insts <= RX_MEDIUM_INSTS)
    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(strings_count),
                      d_results,
                      count_fn<RX_STACK_MEDIUM>{d_prog, d_column});
  else
    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(strings_count),
                      d_results,
                      count_fn<RX_STACK_LARGE>{d_prog, d_column});

  results->set_null_count(strings.null_count());
  return results;
}

}  // namespace detail

// external API

std::unique_ptr<column> count_re(strings_column_view const& strings,
                                 std::string const& pattern,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::count_re(strings, pattern, rmm::cuda_stream_default, mr);
}

}  // namespace strings
}  // namespace cudf
