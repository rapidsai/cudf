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

#include <strings/regex/regex.cuh>
#include <strings/utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/extract.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace strings {
namespace detail {

namespace {

using string_index_pair = thrust::pair<const char*, size_type>;

/**
 * @brief This functor handles extracting strings by applying the compiled regex pattern
 * and creating string_index_pairs for all the substrings.
 *
 * @tparam stack_size Correlates to the regex instructions state to maintain for each string.
 *         Each instruction requires a fixed amount of overhead data.
 */
template <int stack_size>
struct extract_fn {
  reprog_device prog;
  column_device_view d_strings;
  size_type column_index;

  __device__ string_index_pair operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) return string_index_pair{nullptr, 0};
    string_view d_str = d_strings.element<string_view>(idx);
    string_index_pair result{nullptr, 0};
    int32_t begin = 0;
    int32_t end   = -1;  // handles empty strings automatically
    if (prog.find<stack_size>(idx, d_str, begin, end) > 0) {
      auto extracted = prog.extract<stack_size>(idx, d_str, begin, end, column_index);
      if (extracted) {
        auto const offset = d_str.byte_offset(extracted.value().first);
        // build index-pair
        result = string_index_pair{d_str.data() + offset,
                                   d_str.byte_offset(extracted.value().second) - offset};
      }
    }
    return result;
  }
};

}  // namespace

//
std::unique_ptr<table> extract(
  strings_column_view const& strings,
  std::string const& pattern,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto strings_count  = strings.size();
  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;

  // compile regex into device object
  auto prog   = reprog_device::create(pattern, get_character_flags_table(), strings_count, stream);
  auto d_prog = *prog;
  // extract should include groups
  auto const groups = d_prog.group_counts();
  CUDF_EXPECTS(groups > 0, "Group indicators not found in regex pattern");

  // build a result column for each group
  std::vector<std::unique_ptr<column>> results;
  auto regex_insts = d_prog.insts_counts();

  for (int32_t column_index = 0; column_index < groups; ++column_index) {
    rmm::device_uvector<string_index_pair> indices(strings_count, stream);

    if (regex_insts <= RX_SMALL_INSTS) {
      thrust::transform(rmm::exec_policy(stream),
                        thrust::make_counting_iterator<size_type>(0),
                        thrust::make_counting_iterator<size_type>(strings_count),
                        indices.begin(),
                        extract_fn<RX_STACK_SMALL>{d_prog, d_strings, column_index});
    } else if (regex_insts <= RX_MEDIUM_INSTS) {
      thrust::transform(rmm::exec_policy(stream),
                        thrust::make_counting_iterator<size_type>(0),
                        thrust::make_counting_iterator<size_type>(strings_count),
                        indices.begin(),
                        extract_fn<RX_STACK_MEDIUM>{d_prog, d_strings, column_index});
    } else if (regex_insts <= RX_LARGE_INSTS) {
      thrust::transform(rmm::exec_policy(stream),
                        thrust::make_counting_iterator<size_type>(0),
                        thrust::make_counting_iterator<size_type>(strings_count),
                        indices.begin(),
                        extract_fn<RX_STACK_LARGE>{d_prog, d_strings, column_index});
    } else {
      thrust::transform(rmm::exec_policy(stream),
                        thrust::make_counting_iterator<size_type>(0),
                        thrust::make_counting_iterator<size_type>(strings_count),
                        indices.begin(),
                        extract_fn<RX_STACK_ANY>{d_prog, d_strings, column_index});
    }

    results.emplace_back(make_strings_column(indices, stream, mr));
  }
  return std::make_unique<table>(std::move(results));
}

}  // namespace detail

// external API

std::unique_ptr<table> extract(strings_column_view const& strings,
                               std::string const& pattern,
                               rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::extract(strings, pattern, rmm::cuda_stream_default, mr);
}

}  // namespace strings
}  // namespace cudf
