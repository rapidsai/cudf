/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/strings/string_view.cuh>

#include <rmm/exec_policy.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {

namespace {
/**
 * @brief Functor counts the total matches to the given regex in each string.
 */
template <int stack_size>
struct count_matches_fn {
  column_device_view const d_strings;
  reprog_device prog;

  __device__ size_type operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) { return 0; }
    size_type count   = 0;
    auto const d_str  = d_strings.element<string_view>(idx);
    auto const nchars = d_str.length();

    int32_t begin = 0;
    int32_t end   = nchars;
    while ((begin < end) && (prog.find<stack_size>(idx, d_str, begin, end) > 0)) {
      ++count;
      begin = end + (begin == end);
      end   = nchars;
    }
    return count;
  }
};

struct count_dispatch_fn {
  reprog_device d_prog;

  template <int stack_size>
  std::unique_ptr<column> operator()(column_device_view const& d_strings,
                                     size_type output_size,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr)
  {
    assert(output_size >= d_strings.size() and "Unexpected output size");

    auto results = make_numeric_column(
      data_type{type_id::INT32}, output_size, mask_state::UNALLOCATED, stream, mr);

    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(d_strings.size()),
                      results->mutable_view().data<int32_t>(),
                      count_matches_fn<stack_size>{d_strings, d_prog});
    return results;
  }
};

}  // namespace

/**
 * @copydoc cudf::strings::detail::count_matches
 */
std::unique_ptr<column> count_matches(column_device_view const& d_strings,
                                      reprog_device const& d_prog,
                                      size_type output_size,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{
  return regex_dispatcher(d_prog, count_dispatch_fn{d_prog}, d_strings, output_size, stream, mr);
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf
