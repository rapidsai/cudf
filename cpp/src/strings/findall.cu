/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/findall.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>

#include <strings/regex/regex.cuh>
#include <strings/utilities.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/extrema.h>

namespace cudf {
namespace strings {
namespace detail {
using string_index_pair = thrust::pair<const char*, size_type>;
using findall_result    = thrust::pair<size_type, string_index_pair>;

namespace {
/**
 * @brief This functor handles extracting matched strings by applying the compiled regex pattern
 * and creating string_index_pairs for all the substrings.
 */
template <int stack_size>
struct findall_fn {
  column_device_view const d_strings;
  reprog_device prog;
  size_type column_index;
  size_type const* d_counts;

  findall_fn(column_device_view const& d_strings,
             reprog_device& prog,
             size_type column_index    = -1,
             size_type const* d_counts = nullptr)
    : d_strings(d_strings), prog(prog), column_index(column_index), d_counts(d_counts)
  {
  }

  // this will count columns as well as locate a specific string for a column
  __device__ findall_result findall(size_type idx)
  {
    string_index_pair result{nullptr, 0};
    if (d_strings.is_null(idx) || (d_counts && (column_index >= d_counts[idx])))
      return findall_result{0, result};
    string_view d_str      = d_strings.element<string_view>(idx);
    auto const nchars      = d_str.length();
    int32_t spos           = 0;
    int32_t epos           = static_cast<int32_t>(nchars);
    size_type column_count = 0;
    while (spos <= nchars) {
      if (prog.find<stack_size>(idx, d_str, spos, epos) <= 0) break;  // no more matches found
      if (column_count == column_index) break;                        // found our column
      spos = epos > spos ? epos : spos + 1;
      epos = static_cast<int32_t>(nchars);
      ++column_count;
    }
    if (spos <= epos) {
      spos   = d_str.byte_offset(spos);  // convert
      epos   = d_str.byte_offset(epos);  // to bytes
      result = string_index_pair{d_str.data() + spos, (epos - spos)};
    }
    // return the strings location and the column count
    return findall_result{column_count, result};
  }

  __device__ string_index_pair operator()(size_type idx)
  {
    // this one only cares about the string
    return findall(idx).second;
  }
};

template <size_t stack_size>
struct findall_count_fn : public findall_fn<stack_size> {
  findall_count_fn(column_device_view const& strings, reprog_device& prog)
    : findall_fn<stack_size>{strings, prog}
  {
  }

  __device__ size_type operator()(size_type idx)
  {
    // this one only cares about the column count
    return findall_fn<stack_size>::findall(idx).first;
  }
};

}  // namespace

//
std::unique_ptr<table> findall_re(
  strings_column_view const& strings,
  std::string const& pattern,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource(),
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default)
{
  auto const strings_count = strings.size();
  auto const d_strings     = column_device_view::create(strings.parent(), stream);

  auto const d_flags = detail::get_character_flags_table();
  // compile regex into device object
  auto const d_prog      = reprog_device::create(pattern, d_flags, strings_count, stream);
  auto const regex_insts = d_prog->insts_counts();

  rmm::device_uvector<size_type> find_counts(strings_count, stream);
  auto d_find_counts = find_counts.data();

  if (regex_insts <= RX_SMALL_INSTS)
    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(strings_count),
                      d_find_counts,
                      findall_count_fn<RX_STACK_SMALL>{*d_strings, *d_prog});
  else if (regex_insts <= RX_MEDIUM_INSTS)
    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(strings_count),
                      d_find_counts,
                      findall_count_fn<RX_STACK_MEDIUM>{*d_strings, *d_prog});
  else if (regex_insts <= RX_LARGE_INSTS)
    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(strings_count),
                      d_find_counts,
                      findall_count_fn<RX_STACK_LARGE>{*d_strings, *d_prog});
  else
    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(strings_count),
                      d_find_counts,
                      findall_count_fn<RX_STACK_ANY>{*d_strings, *d_prog});

  std::vector<std::unique_ptr<column>> results;

  size_type const columns = thrust::reduce(
    rmm::exec_policy(stream), find_counts.begin(), find_counts.end(), 0, thrust::maximum{});
  // boundary case: if no columns, return all nulls column (issue #119)
  if (columns == 0)
    results.emplace_back(std::make_unique<column>(
      data_type{type_id::STRING},
      strings_count,
      rmm::device_buffer{0, stream, mr},  // no data
      cudf::detail::create_null_mask(strings_count, mask_state::ALL_NULL, stream, mr),
      strings_count));

  for (int32_t column_index = 0; column_index < columns; ++column_index) {
    rmm::device_uvector<string_index_pair> indices(strings_count, stream);

    if (regex_insts <= RX_SMALL_INSTS)
      thrust::transform(
        rmm::exec_policy(stream),
        thrust::make_counting_iterator<size_type>(0),
        thrust::make_counting_iterator<size_type>(strings_count),
        indices.begin(),
        findall_fn<RX_STACK_SMALL>{*d_strings, *d_prog, column_index, d_find_counts});
    else if (regex_insts <= RX_MEDIUM_INSTS)
      thrust::transform(
        rmm::exec_policy(stream),
        thrust::make_counting_iterator<size_type>(0),
        thrust::make_counting_iterator<size_type>(strings_count),
        indices.begin(),
        findall_fn<RX_STACK_MEDIUM>{*d_strings, *d_prog, column_index, d_find_counts});
    else if (regex_insts <= RX_LARGE_INSTS)
      thrust::transform(
        rmm::exec_policy(stream),
        thrust::make_counting_iterator<size_type>(0),
        thrust::make_counting_iterator<size_type>(strings_count),
        indices.begin(),
        findall_fn<RX_STACK_LARGE>{*d_strings, *d_prog, column_index, d_find_counts});
    else
      thrust::transform(rmm::exec_policy(stream),
                        thrust::make_counting_iterator<size_type>(0),
                        thrust::make_counting_iterator<size_type>(strings_count),
                        indices.begin(),
                        findall_fn<RX_STACK_ANY>{*d_strings, *d_prog, column_index, d_find_counts});

    //
    results.emplace_back(make_strings_column(indices.begin(), indices.end(), stream, mr));
  }
  return std::make_unique<table>(std::move(results));
}

}  // namespace detail

// external API

std::unique_ptr<table> findall_re(strings_column_view const& strings,
                                  std::string const& pattern,
                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::findall_re(strings, pattern, mr);
}

}  // namespace strings
}  // namespace cudf
