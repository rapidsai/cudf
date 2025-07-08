/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {

/**
 * @brief Threshold to decide on using string or warp parallel functions.
 *
 * If the average byte length of a string in a column exceeds this value then
 * a warp-parallel function is used.
 *
 * Note that this value is shared by find, rfind, and contains functions.
 */
constexpr size_type AVG_CHAR_BYTES_THRESHOLD = 64;

/**
 * @brief Find function handles a string per thread
 */
struct find_instance_fn {
  column_device_view const d_strings;
  string_view const d_target;
  size_type const index;

  __device__ size_type operator()(size_type idx) const
  {
    if (d_strings.is_null(idx)) { return -1; }
    auto const d_str = d_strings.element<string_view>(idx);

    auto position = -1;
    for (auto i = 0; i <= index; ++i) {
      position = d_str.find(d_target, position + 1);
    }
    return position;
  }
};

/**
 * @brief String per warp function for find_instance
 */
CUDF_KERNEL void find_instance_warp_parallel_fn(column_device_view const d_strings,
                                                string_view const d_target,
                                                size_type const index,
                                                size_type* d_results)
{
  namespace cg        = cooperative_groups;
  auto const warp     = cg::tiled_partition<cudf::detail::warp_size>(cg::this_thread_block());
  auto const lane_idx = warp.thread_rank();

  auto const tid     = cudf::detail::grid_1d::global_thread_id();
  auto const str_idx = tid / cudf::detail::warp_size;
  if (str_idx >= d_strings.size() or d_strings.is_null(str_idx)) { return; }

  auto const d_str = d_strings.element<string_view>(str_idx);
  auto const begin = d_str.data();
  auto const end   = d_str.data() + d_str.size_bytes();

  // each thread compares the target with the thread's individual starting byte
  size_type char_pos = d_str.size_bytes();
  size_type total    = 0;
  for (auto itr = begin + lane_idx;
       itr + d_target.size_bytes() <= end && char_pos < d_str.size_bytes();
       itr += cudf::detail::warp_size) {
    size_type is_char = !is_utf8_continuation_char(*itr);
    size_type found   = is_char && (d_target.compare(itr, d_target.size_bytes()) == 0);
    auto count        = cg::reduce(warp, found, cg::plus<int>());
    auto found_scan   = cg::inclusive_scan(warp, found);
    auto chars_scan   = cg::inclusive_scan(warp, is_char);
    auto pos_check    = (found_scan + total) == index ? chars_scan : char_pos;
    char_pos          = cg::reduce(warp, pos_check, cg::less<size_type>());
    total += count;
  }

  if (lane_idx == 0) { d_results[str_idx] = char_pos == d_str.size_bytes() ? -1 : char_pos; }
}

void find_utility(strings_column_view const& input,
                  string_view const& target,
                  column& output,
                  size_type index,
                  rmm::cuda_stream_view stream)
{
  auto d_strings = column_device_view::create(input.parent(), stream);
  auto d_results = output.mutable_view().data<size_type>();
  if ((input.chars_size(stream) / (input.size() - input.null_count())) > AVG_CHAR_BYTES_THRESHOLD) {
    // warp-per-string runs faster for longer strings (but not shorter ones)
    constexpr int block_size = 256;
    cudf::detail::grid_1d grid{input.size() * cudf::detail::warp_size, block_size};
    find_instance_warp_parallel_fn<<<grid.num_blocks,
                                     grid.num_threads_per_block,
                                     0,
                                     stream.value()>>>(*d_strings, target, index, d_results);
  } else {
    // string-per-thread function
    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(input.size()),
                      d_results,
                      find_instance_fn{*d_strings, target, index});
  }
}

}  // namespace

std::unique_ptr<column> find_instance(strings_column_view const& input,
                                      string_scalar const& target,
                                      size_type instance,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(
    instance >= 0, "Parameter instance must be positive integer or zero.", std::invalid_argument);
  CUDF_EXPECTS(target.is_valid(stream), "Parameter target must be valid.", std::invalid_argument);

  // create output column
  auto results = make_numeric_column(data_type{type_to_id<size_type>()},
                                     input.size(),
                                     cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                     input.null_count(),
                                     stream,
                                     mr);
  // if input is empty or all-null then we are done
  if (input.size() == input.null_count()) { return results; }

  // call find utility with target iterator
  auto d_target = target.value(stream);
  find_utility(input, d_target, *results, instance, stream);
  return results;
}

}  // namespace detail

std::unique_ptr<column> find_instance(strings_column_view const& input,
                                      string_scalar const& target,
                                      size_type instance,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::find_instance(input, target, instance, stream, mr);
}

}  // namespace strings
}  // namespace cudf
