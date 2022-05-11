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
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/findall.hpp>
#include <cudf/strings/string_view.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/pair.h>
#include <thrust/reduce.h>

namespace cudf {
namespace strings {
namespace detail {
using string_index_pair = thrust::pair<const char*, size_type>;
using indices_span      = cudf::detail::device_2dspan<string_index_pair>;

namespace {
/**
 * @brief This functor calls regex find on each string and creates
 * string_index_pairs for all matching substrings.
 *
 * The number of output columns is dependent on the string with the most matches.
 * For strings with fewer matches, null entries are appended into `d_indices`
 * up to the maximum column count.
 */
struct findall_fn {
  column_device_view const d_strings;
  size_type const* d_counts;  ///< match counts for each string
  indices_span d_indices;     ///< 2D-span: output matches added here

  __device__ void operator()(size_type const idx, reprog_device const prog, int32_t const prog_idx)
  {
    auto const match_count = d_counts[idx];

    auto d_output = d_indices[idx];

    if (d_strings.is_valid(idx)) {
      auto const d_str  = d_strings.element<string_view>(idx);
      auto const nchars = d_str.length();

      int32_t begin = 0;
      int32_t end   = -1;
      for (auto col_idx = 0; col_idx < match_count; ++col_idx) {
        if (prog.find(prog_idx, d_str, begin, end) > 0) {
          auto const begin_offset = d_str.byte_offset(begin);
          auto const end_offset   = d_str.byte_offset(end);
          d_output[col_idx] =
            string_index_pair{d_str.data() + begin_offset, end_offset - begin_offset};
        }
        begin = end + (begin == end);
        end   = nchars;
      }
    }
    // fill the remaining entries for this row with nulls
    thrust::fill(
      thrust::seq, d_output.begin() + match_count, d_output.end(), string_index_pair{nullptr, 0});
  }
};

}  // namespace

std::unique_ptr<table> findall(strings_column_view const& input,
                               std::string const& pattern,
                               regex_flags const flags,
                               rmm::cuda_stream_view stream,
                               rmm::mr::device_memory_resource* mr)
{
  auto const strings_count = input.size();

  // compile regex into device object
  auto const d_prog = reprog_device::create(pattern, flags, stream);

  auto const d_strings = column_device_view::create(input.parent(), stream);
  auto find_counts     = count_matches(*d_strings, *d_prog, strings_count, stream);
  auto d_find_counts   = find_counts->view().data<size_type>();

  size_type const columns_count = thrust::reduce(
    rmm::exec_policy(stream), d_find_counts, d_find_counts + strings_count, 0, thrust::maximum{});

  auto indices = rmm::device_uvector<string_index_pair>(strings_count * columns_count, stream);

  std::vector<std::unique_ptr<column>> results;
  // boundary case: if no columns, return all nulls column (issue #119)
  if (columns_count == 0) {
    results.emplace_back(std::make_unique<column>(
      data_type{type_id::STRING},
      strings_count,
      rmm::device_buffer{0, stream, mr},  // no data
      cudf::detail::create_null_mask(strings_count, mask_state::ALL_NULL, stream, mr),
      strings_count));
  } else {
    // place all matching strings into the indices vector
    auto d_indices = indices_span(indices.data(), strings_count, columns_count);
    launch_for_each_kernel(
      findall_fn{*d_strings, d_find_counts, d_indices}, *d_prog, strings_count, stream);
    results.resize(columns_count);
  }

  // build the output column using the strings in the indices vector
  auto make_strings_lambda = [&](size_type const column_index) {
    // this iterator transposes the strided results into column order
    auto indices_itr = thrust::make_permutation_iterator(
      indices.begin(),
      cudf::detail::make_counting_transform_iterator(
        0, [column_index, columns_count] __device__(size_type const idx) {
          return (idx * columns_count) + column_index;
        }));
    return make_strings_column(indices_itr, indices_itr + strings_count, stream, mr);
  };

  std::transform(thrust::make_counting_iterator<size_type>(0),
                 thrust::make_counting_iterator<size_type>(columns_count),
                 results.begin(),
                 make_strings_lambda);

  return std::make_unique<table>(std::move(results));
}

}  // namespace detail

// external API

std::unique_ptr<table> findall(strings_column_view const& input,
                               std::string const& pattern,
                               regex_flags const flags,
                               rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::findall(input, pattern, flags, rmm::cuda_stream_default, mr);
}

}  // namespace strings
}  // namespace cudf
