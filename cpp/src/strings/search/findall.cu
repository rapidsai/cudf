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
#include <strings/regex/dispatcher.hpp>
#include <strings/regex/regex.cuh>
#include <strings/utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/findall.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/reduce.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {
using string_index_pair = thrust::pair<const char*, size_type>;

namespace {
/**
 * @brief This functor handles extracting matched strings by applying the compiled regex pattern
 * and creating string_index_pairs for all the substrings.
 */
template <int stack_size>
struct findall_fn {
  column_device_view const d_strings;
  reprog_device prog;
  size_type const column_index;
  size_type const* d_counts;

  __device__ string_index_pair operator()(size_type idx)
  {
    if (d_strings.is_null(idx) || (column_index >= d_counts[idx])) {
      return string_index_pair{nullptr, 0};
    }

    auto const d_str  = d_strings.element<string_view>(idx);
    auto const nchars = d_str.length();
    int32_t spos      = 0;
    auto epos         = static_cast<int32_t>(nchars);

    size_type column_count = 0;
    while (spos <= nchars) {
      if (prog.find<stack_size>(idx, d_str, spos, epos) <= 0) break;  // no more matches found
      if (column_count == column_index) break;                        // found our column
      spos = epos > spos ? epos : spos + 1;
      epos = static_cast<int32_t>(nchars);
      ++column_count;
    }

    auto const result = [&] {
      if (spos > epos) { return string_index_pair{nullptr, 0}; }
      spos = d_str.byte_offset(spos);  // convert
      epos = d_str.byte_offset(epos);  // to bytes
      return string_index_pair{d_str.data() + spos, (epos - spos)};
    }();

    return result;
  }
};

struct findall_dispatch_fn {
  reprog_device d_prog;

  template <int stack_size>
  std::unique_ptr<column> operator()(column_device_view const& d_strings,
                                     size_type column_index,
                                     size_type const* d_find_counts,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    rmm::device_uvector<string_index_pair> indices(d_strings.size(), stream);

    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(d_strings.size()),
                      indices.begin(),
                      findall_fn<stack_size>{d_strings, d_prog, column_index, d_find_counts});

    return make_strings_column(indices.begin(), indices.end(), stream, mr);
  }
};
}  // namespace

//
std::unique_ptr<table> findall(
  strings_column_view const& input,
  std::string const& pattern,
  regex_flags const flags,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto const strings_count = input.size();
  auto const d_strings     = column_device_view::create(input.parent(), stream);

  // compile regex into device object
  auto const d_prog =
    reprog_device::create(pattern, flags, get_character_flags_table(), strings_count, stream);
  auto const regex_insts = d_prog->insts_counts();

  auto find_counts =
    count_matches(*d_strings, *d_prog, stream, rmm::mr::get_current_device_resource());
  auto d_find_counts = find_counts->mutable_view().data<size_type>();

  std::vector<std::unique_ptr<column>> results;

  size_type const columns = thrust::reduce(
    rmm::exec_policy(stream), d_find_counts, d_find_counts + strings_count, 0, thrust::maximum{});

  // boundary case: if no columns, return all nulls column (issue #119)
  if (columns == 0)
    results.emplace_back(std::make_unique<column>(
      data_type{type_id::STRING},
      strings_count,
      rmm::device_buffer{0, stream, mr},  // no data
      cudf::detail::create_null_mask(strings_count, mask_state::ALL_NULL, stream, mr),
      strings_count));

  for (int32_t column_index = 0; column_index < columns; ++column_index) {
    results.emplace_back(regex_dispatcher(
      *d_prog, findall_dispatch_fn{*d_prog}, *d_strings, column_index, d_find_counts, stream, mr));
  }

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
