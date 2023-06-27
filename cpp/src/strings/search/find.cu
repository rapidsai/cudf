/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.
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
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/device_atomics.cuh>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
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
template <bool forward = true>
struct finder_fn {
  column_device_view const d_strings;
  string_view const d_target;
  size_type const start;
  size_type const stop;

  __device__ size_type operator()(size_type idx) const
  {
    if (d_strings.is_null(idx)) { return -1; }
    auto d_str = d_strings.element<string_view>(idx);

    auto const length = d_str.length();
    auto const begin  = (start > length) ? length : start;
    auto const end    = (stop < 0) || (stop > length) ? length : stop;
    return forward ? d_str.find(d_target, begin, end - begin)
                   : d_str.rfind(d_target, begin, end - begin);
  }
};

/**
 * @brief Special logic handles an empty target for find/rfind
 *
 * where length = number of characters in the input string
 * if forward = true:
 *   return start iff (start <= length), otherwise return -1
 * if forward = false:
 *   return stop iff (0 <= stop <= length), otherwise return length
 */
template <bool forward = true>
struct empty_target_fn {
  column_device_view const d_strings;
  size_type const start;
  size_type const stop;

  __device__ size_type operator()(size_type idx) const
  {
    if (d_strings.is_null(idx)) { return -1; }
    auto d_str = d_strings.element<string_view>(idx);

    // common case shortcut
    if (forward && start == 0) { return 0; }

    auto const length = d_str.length();
    if (start > length) { return -1; }
    if constexpr (forward) { return start; }

    return (stop < 0) || (stop > length) ? length : stop;
  }
};

/**
 * @brief String per warp function for find/rfind
 */
template <bool forward = true>
__global__ void finder_warp_parallel_fn(column_device_view const d_strings,
                                        string_view const d_target,
                                        size_type const start,
                                        size_type const stop,
                                        size_type* d_results)
{
  size_type const idx = static_cast<size_type>(threadIdx.x + blockIdx.x * blockDim.x);

  if (idx >= (d_strings.size() * cudf::detail::warp_size)) { return; }

  auto const str_idx  = idx / cudf::detail::warp_size;
  auto const lane_idx = idx % cudf::detail::warp_size;

  if (d_strings.is_null(str_idx)) { return; }

  // initialize the output for the atomicMin/Max
  if (lane_idx == 0) { d_results[str_idx] = forward ? std::numeric_limits<size_type>::max() : -1; }
  __syncwarp();

  auto const d_str = d_strings.element<string_view>(str_idx);

  auto const [begin, left_over] = bytes_to_character_position(d_str, start);
  auto const start_char_pos     = start - left_over;  // keep track of character position

  auto const end = [d_str, start, stop, begin = begin] {
    if (stop < 0) { return d_str.size_bytes(); }
    if (stop <= start) { return begin; }
    // we count from `begin` instead of recounting from the beginning of the string
    return begin + std::get<0>(bytes_to_character_position(
                     string_view(d_str.data() + begin, d_str.size_bytes() - begin), stop - start));
  }();

  // each thread compares the target with the thread's individual starting byte
  size_type position = forward ? std::numeric_limits<size_type>::max() : -1;
  for (auto itr = begin + lane_idx; itr + d_target.size_bytes() <= end;
       itr += cudf::detail::warp_size) {
    if (d_target.compare(d_str.data() + itr, d_target.size_bytes()) == 0) {
      position = itr;
      if (forward) break;
    }
  }

  // find stores the minimum position while rfind stores the maximum position
  // note that this was slightly faster than using cub::WarpReduce
  forward ? atomicMin(d_results + str_idx, position) : atomicMax(d_results + str_idx, position);
  __syncwarp();

  if (lane_idx == 0) {
    // the final result needs to be fixed up convert max() to -1
    // and a byte position to a character position
    auto const result = d_results[str_idx];
    d_results[str_idx] =
      ((result < std::numeric_limits<size_type>::max()) && (result >= begin))
        ? start_char_pos + characters_in_string(d_str.data() + begin, result - begin)
        : -1;
  }
}

template <bool forward = true>
std::unique_ptr<column> find_fn(strings_column_view const& input,
                                string_scalar const& target,
                                size_type start,
                                size_type stop,
                                rmm::cuda_stream_view stream,
                                rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(target.is_valid(stream), "Parameter target must be valid.");
  CUDF_EXPECTS(start >= 0, "Parameter start must be positive integer or zero.");
  if ((stop > 0) && (start > stop)) CUDF_FAIL("Parameter start must be less than stop.");

  auto d_target  = string_view(target.data(), target.size());
  auto d_strings = column_device_view::create(input.parent(), stream);

  // create output column
  auto results = make_numeric_column(data_type{type_to_id<size_type>()},
                                     input.size(),
                                     cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                     input.null_count(),
                                     stream,
                                     mr);
  // if input is empty or all-null then we are done
  if (input.size() == input.null_count()) { return results; }

  auto d_results = results->mutable_view().data<size_type>();

  if (d_target.empty()) {
    // special logic for empty target results
    thrust::transform(rmm::exec_policy(stream),
                      thrust::counting_iterator<size_type>(0),
                      thrust::counting_iterator<size_type>(input.size()),
                      d_results,
                      empty_target_fn<forward>{*d_strings, start, stop});
  } else if ((input.chars_size() / (input.size() - input.null_count())) >
             AVG_CHAR_BYTES_THRESHOLD) {
    // warp-per-string runs faster for longer strings (but not shorter ones)
    constexpr int block_size = 256;
    cudf::detail::grid_1d grid{input.size() * cudf::detail::warp_size, block_size};
    finder_warp_parallel_fn<forward>
      <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
        *d_strings, d_target, start, stop, d_results);
  } else {
    // string-per-thread function
    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(input.size()),
                      d_results,
                      finder_fn<forward>{*d_strings, d_target, start, stop});
  }

  results->set_null_count(input.null_count());
  return results;
}
}  // namespace

std::unique_ptr<column> find(strings_column_view const& input,
                             string_scalar const& target,
                             size_type start,
                             size_type stop,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr)
{
  return find_fn<true>(input, target, start, stop, stream, mr);
}

std::unique_ptr<column> rfind(strings_column_view const& input,
                              string_scalar const& target,
                              size_type start,
                              size_type stop,
                              rmm::cuda_stream_view stream,
                              rmm::mr::device_memory_resource* mr)
{
  return find_fn<false>(input, target, start, stop, stream, mr);
}

}  // namespace detail

// external APIs

std::unique_ptr<column> find(strings_column_view const& strings,
                             string_scalar const& target,
                             size_type start,
                             size_type stop,
                             rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::find(strings, target, start, stop, cudf::get_default_stream(), mr);
}

std::unique_ptr<column> rfind(strings_column_view const& strings,
                              string_scalar const& target,
                              size_type start,
                              size_type stop,
                              rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::rfind(strings, target, start, stop, cudf::get_default_stream(), mr);
}

namespace detail {
namespace {

/**
 * @brief Check if `d_target` appears in a row in `d_strings`.
 *
 * This executes as a warp per string/row and performs well for longer strings.
 * @see AVG_CHAR_BYTES_THRESHOLD
 *
 * @param d_strings Column of input strings
 * @param d_target String to search for in each row of `d_strings`
 * @param d_results Indicates which rows contain `d_target`
 */
__global__ void contains_warp_parallel_fn(column_device_view const d_strings,
                                          string_view const d_target,
                                          bool* d_results)
{
  size_type const idx = static_cast<size_type>(threadIdx.x + blockIdx.x * blockDim.x);
  using warp_reduce   = cub::WarpReduce<bool>;
  __shared__ typename warp_reduce::TempStorage temp_storage;

  if (idx >= (d_strings.size() * cudf::detail::warp_size)) { return; }

  auto const str_idx  = idx / cudf::detail::warp_size;
  auto const lane_idx = idx % cudf::detail::warp_size;
  if (d_strings.is_null(str_idx)) { return; }
  // get the string for this warp
  auto const d_str = d_strings.element<string_view>(str_idx);
  // each thread of the warp will check just part of the string
  auto found = false;
  for (auto i = static_cast<size_type>(idx % cudf::detail::warp_size);
       !found && (i + d_target.size_bytes()) < d_str.size_bytes();
       i += cudf::detail::warp_size) {
    // check the target matches this part of the d_str data
    if (d_target.compare(d_str.data() + i, d_target.size_bytes()) == 0) { found = true; }
  }
  auto const result = warp_reduce(temp_storage).Reduce(found, cub::Max());
  if (lane_idx == 0) { d_results[str_idx] = result; }
}

std::unique_ptr<column> contains_warp_parallel(strings_column_view const& input,
                                               string_scalar const& target,
                                               rmm::cuda_stream_view stream,
                                               rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(target.is_valid(stream), "Parameter target must be valid.");
  auto d_target = string_view(target.data(), target.size());

  // create output column
  auto results = make_numeric_column(data_type{type_id::BOOL8},
                                     input.size(),
                                     cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                     input.null_count(),
                                     stream,
                                     mr);

  // fill the output with `false` unless the `d_target` is empty
  auto results_view = results->mutable_view();
  thrust::fill(rmm::exec_policy(stream),
               results_view.begin<bool>(),
               results_view.end<bool>(),
               d_target.empty());

  if (!d_target.empty()) {
    // launch warp per string
    auto const d_strings     = column_device_view::create(input.parent(), stream);
    constexpr int block_size = 256;
    cudf::detail::grid_1d grid{input.size() * cudf::detail::warp_size, block_size};
    contains_warp_parallel_fn<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
      *d_strings, d_target, results_view.data<bool>());
  }
  results->set_null_count(input.null_count());
  return results;
}

/**
 * @brief Utility to return a bool column indicating the presence of
 * a given target string in a strings column.
 *
 * Null string entries return corresponding null output column entries.
 *
 * @tparam BoolFunction Return bool value given two strings.
 *
 * @param strings Column of strings to check for target.
 * @param target UTF-8 encoded string to check in strings column.
 * @param pfn Returns bool value if target is found in the given string.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New BOOL column.
 */
template <typename BoolFunction>
std::unique_ptr<column> contains_fn(strings_column_view const& strings,
                                    string_scalar const& target,
                                    BoolFunction pfn,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  auto strings_count = strings.size();
  if (strings_count == 0) return make_empty_column(type_id::BOOL8);

  CUDF_EXPECTS(target.is_valid(stream), "Parameter target must be valid.");
  if (target.size() == 0)  // empty target string returns true
  {
    auto const true_scalar = make_fixed_width_scalar<bool>(true, stream);
    auto results           = make_column_from_scalar(*true_scalar, strings.size(), stream, mr);
    results->set_null_mask(cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                           strings.null_count());
    return results;
  }

  auto d_target       = string_view(target.data(), target.size());
  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;
  // create output column
  auto results      = make_numeric_column(data_type{type_id::BOOL8},
                                     strings_count,
                                     cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                                     strings.null_count(),
                                     stream,
                                     mr);
  auto results_view = results->mutable_view();
  auto d_results    = results_view.data<bool>();
  // set the bool values by evaluating the passed function
  thrust::transform(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<size_type>(0),
                    thrust::make_counting_iterator<size_type>(strings_count),
                    d_results,
                    [d_strings, pfn, d_target] __device__(size_type idx) {
                      if (!d_strings.is_null(idx))
                        return bool{pfn(d_strings.element<string_view>(idx), d_target)};
                      return false;
                    });
  results->set_null_count(strings.null_count());
  return results;
}

/**
 * @brief Utility to return a bool column indicating the presence of
 * a string targets[i] in strings[i].
 *
 * Null string entries return corresponding null output column entries.
 *
 * @tparam BoolFunction Return bool value given two strings.
 *
 * @param strings Column of strings to check for `targets[i]`.
 * @param targets Column of strings to be checked in `strings[i]``.
 * @param pfn Returns bool value if target is found in the given string.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New BOOL column.
 */
template <typename BoolFunction>
std::unique_ptr<column> contains_fn(strings_column_view const& strings,
                                    strings_column_view const& targets,
                                    BoolFunction pfn,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  if (strings.is_empty()) return make_empty_column(type_id::BOOL8);

  CUDF_EXPECTS(targets.size() == strings.size(),
               "strings and targets column must be the same size");

  auto targets_column = column_device_view::create(targets.parent(), stream);
  auto d_targets      = *targets_column;
  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;
  // create output column
  auto results      = make_numeric_column(data_type{type_id::BOOL8},
                                     strings.size(),
                                     cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                                     strings.null_count(),
                                     stream,
                                     mr);
  auto results_view = results->mutable_view();
  auto d_results    = results_view.data<bool>();
  // set the bool values by evaluating the passed function
  thrust::transform(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<size_type>(0),
    thrust::make_counting_iterator<size_type>(strings.size()),
    d_results,
    [d_strings, pfn, d_targets] __device__(size_type idx) {
      // empty target string returns true
      if (d_targets.is_valid(idx) && d_targets.element<string_view>(idx).length() == 0) {
        return true;
      } else if (!d_strings.is_null(idx) && !d_targets.is_null(idx)) {
        return bool{pfn(d_strings.element<string_view>(idx), d_targets.element<string_view>(idx))};
      } else {
        return false;
      }
    });
  results->set_null_count(strings.null_count());
  return results;
}

}  // namespace

std::unique_ptr<column> contains(strings_column_view const& input,
                                 string_scalar const& target,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  // use warp parallel when the average string width is greater than the threshold
  if ((input.null_count() < input.size()) &&
      ((input.chars_size() / input.size()) > AVG_CHAR_BYTES_THRESHOLD)) {
    return contains_warp_parallel(input, target, stream, mr);
  }

  // benchmark measurements showed this to be faster for smaller strings
  auto pfn = [] __device__(string_view d_string, string_view d_target) {
    return d_string.find(d_target) != string_view::npos;
  };
  return contains_fn(input, target, pfn, stream, mr);
}

std::unique_ptr<column> contains(strings_column_view const& strings,
                                 strings_column_view const& targets,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource* mr)
{
  auto pfn = [] __device__(string_view d_string, string_view d_target) {
    return d_string.find(d_target) != string_view::npos;
  };
  return contains_fn(strings, targets, pfn, stream, mr);
}

std::unique_ptr<column> starts_with(strings_column_view const& strings,
                                    string_scalar const& target,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  auto pfn = [] __device__(string_view d_string, string_view d_target) {
    return (d_target.size_bytes() <= d_string.size_bytes()) &&
           (d_target.compare(d_string.data(), d_target.size_bytes()) == 0);
  };
  return contains_fn(strings, target, pfn, stream, mr);
}

std::unique_ptr<column> starts_with(strings_column_view const& strings,
                                    strings_column_view const& targets,
                                    rmm::cuda_stream_view stream,
                                    rmm::mr::device_memory_resource* mr)
{
  auto pfn = [] __device__(string_view d_string, string_view d_target) {
    return (d_target.size_bytes() <= d_string.size_bytes()) &&
           (d_target.compare(d_string.data(), d_target.size_bytes()) == 0);
  };
  return contains_fn(strings, targets, pfn, stream, mr);
}

std::unique_ptr<column> ends_with(strings_column_view const& strings,
                                  string_scalar const& target,
                                  rmm::cuda_stream_view stream,
                                  rmm::mr::device_memory_resource* mr)
{
  auto pfn = [] __device__(string_view d_string, string_view d_target) {
    auto const str_size = d_string.size_bytes();
    auto const tgt_size = d_target.size_bytes();
    return (tgt_size <= str_size) &&
           (d_target.compare(d_string.data() + str_size - tgt_size, tgt_size) == 0);
  };

  return contains_fn(strings, target, pfn, stream, mr);
}

std::unique_ptr<column> ends_with(strings_column_view const& strings,
                                  strings_column_view const& targets,
                                  rmm::cuda_stream_view stream,
                                  rmm::mr::device_memory_resource* mr)
{
  auto pfn = [] __device__(string_view d_string, string_view d_target) {
    auto const str_size = d_string.size_bytes();
    auto const tgt_size = d_target.size_bytes();
    return (tgt_size <= str_size) &&
           (d_target.compare(d_string.data() + str_size - tgt_size, tgt_size) == 0);
  };

  return contains_fn(strings, targets, pfn, stream, mr);
}

}  // namespace detail

// external APIs

std::unique_ptr<column> contains(strings_column_view const& strings,
                                 string_scalar const& target,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::contains(strings, target, cudf::get_default_stream(), mr);
}

std::unique_ptr<column> contains(strings_column_view const& strings,
                                 strings_column_view const& targets,
                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::contains(strings, targets, cudf::get_default_stream(), mr);
}

std::unique_ptr<column> starts_with(strings_column_view const& strings,
                                    string_scalar const& target,
                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::starts_with(strings, target, cudf::get_default_stream(), mr);
}

std::unique_ptr<column> starts_with(strings_column_view const& strings,
                                    strings_column_view const& targets,
                                    rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::starts_with(strings, targets, cudf::get_default_stream(), mr);
}

std::unique_ptr<column> ends_with(strings_column_view const& strings,
                                  string_scalar const& target,
                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::ends_with(strings, target, cudf::get_default_stream(), mr);
}

std::unique_ptr<column> ends_with(strings_column_view const& strings,
                                  strings_column_view const& targets,
                                  rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::ends_with(strings, targets, cudf::get_default_stream(), mr);
}

}  // namespace strings
}  // namespace cudf
