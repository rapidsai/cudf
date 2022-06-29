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

#include <strings/regex/utilities.cuh>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/extract.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/pair.h>

namespace cudf {
namespace strings {
namespace detail {

namespace {

using string_index_pair = thrust::pair<const char*, size_type>;

/**
 * @brief This functor handles extracting strings by applying the compiled regex pattern
 * and creating string_index_pairs for all the substrings.
 */
struct extract_fn {
  column_device_view const d_strings;
  cudf::detail::device_2dspan<string_index_pair> d_indices;

  __device__ void operator()(size_type const idx,
                             reprog_device const d_prog,
                             int32_t const prog_idx)
  {
    auto const groups = d_prog.group_counts();
    auto d_output     = d_indices[idx];

    if (d_strings.is_valid(idx)) {
      auto const d_str = d_strings.element<string_view>(idx);

      size_type begin = 0;
      size_type end   = -1;  // handles empty strings automatically
      if (d_prog.find(prog_idx, d_str, begin, end) > 0) {
        for (auto col_idx = 0; col_idx < groups; ++col_idx) {
          auto const extracted = d_prog.extract(prog_idx, d_str, begin, end, col_idx);
          d_output[col_idx]    = [&] {
            if (!extracted) return string_index_pair{nullptr, 0};
            auto const offset = d_str.byte_offset((*extracted).first);
            return string_index_pair{d_str.data() + offset,
                                     d_str.byte_offset((*extracted).second) - offset};
          }();
        }
        return;
      }
    }

    // if null row or no match found, fill the output with null entries
    thrust::fill(thrust::seq, d_output.begin(), d_output.end(), string_index_pair{nullptr, 0});
  }
};

}  // namespace

//
std::unique_ptr<table> extract(strings_column_view const& input,
                               std::string_view pattern,
                               regex_flags const flags,
                               rmm::cuda_stream_view stream,
                               rmm::mr::device_memory_resource* mr)
{
  // compile regex into device object
  auto d_prog = reprog_device::create(pattern, flags, stream);

  auto const groups = d_prog->group_counts();
  CUDF_EXPECTS(groups > 0, "Group indicators not found in regex pattern");

  auto indices = rmm::device_uvector<string_index_pair>(input.size() * groups, stream);
  auto d_indices =
    cudf::detail::device_2dspan<string_index_pair>(indices.data(), input.size(), groups);

  auto const d_strings = column_device_view::create(input.parent(), stream);

  launch_for_each_kernel(extract_fn{*d_strings, d_indices}, *d_prog, input.size(), stream);

  // build a result column for each group
  std::vector<std::unique_ptr<column>> results(groups);
  auto make_strings_lambda = [&](size_type column_index) {
    // this iterator transposes the extract results into column order
    auto indices_itr =
      thrust::make_permutation_iterator(indices.begin(),
                                        cudf::detail::make_counting_transform_iterator(
                                          0, [column_index, groups] __device__(size_type idx) {
                                            return (idx * groups) + column_index;
                                          }));
    return make_strings_column(indices_itr, indices_itr + input.size(), stream, mr);
  };

  std::transform(thrust::make_counting_iterator<size_type>(0),
                 thrust::make_counting_iterator<size_type>(groups),
                 results.begin(),
                 make_strings_lambda);

  return std::make_unique<table>(std::move(results));
}

}  // namespace detail

// external API

std::unique_ptr<table> extract(strings_column_view const& strings,
                               std::string_view pattern,
                               regex_flags const flags,
                               rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::extract(strings, pattern, flags, cudf::default_stream_value, mr);
}

}  // namespace strings
}  // namespace cudf
