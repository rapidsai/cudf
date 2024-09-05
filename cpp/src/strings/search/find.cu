/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/lists/list_device_view.cuh>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/atomic>
#include <thrust/binary_search.h>
#include <thrust/fill.h>
#include <thrust/find.h>
#include <thrust/for_each.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>
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
template <typename TargetIterator, bool forward = true>
struct finder_fn {
  column_device_view const d_strings;
  TargetIterator const d_targets;
  size_type const start;
  size_type const stop;

  __device__ size_type operator()(size_type idx) const
  {
    if (d_strings.is_null(idx)) { return -1; }
    auto const d_str = d_strings.element<string_view>(idx);
    if (d_str.empty() && (start > 0)) { return -1; }
    auto const d_target = d_targets[idx];

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
template <typename TargetIterator, bool forward = true>
CUDF_KERNEL void finder_warp_parallel_fn(column_device_view const d_strings,
                                         TargetIterator const d_targets,
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

  auto const d_str    = d_strings.element<string_view>(str_idx);
  auto const d_target = d_targets[str_idx];

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
  cuda::atomic_ref<size_type, cuda::thread_scope_block> ref{*(d_results + str_idx)};
  forward ? ref.fetch_min(position, cuda::std::memory_order_relaxed)
          : ref.fetch_max(position, cuda::std::memory_order_relaxed);
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

template <typename TargetIterator, bool forward = true>
void find_utility(strings_column_view const& input,
                  TargetIterator const& target_itr,
                  column& output,
                  size_type start,
                  size_type stop,
                  rmm::cuda_stream_view stream)
{
  auto d_strings = column_device_view::create(input.parent(), stream);
  auto d_results = output.mutable_view().data<size_type>();
  if ((input.chars_size(stream) / (input.size() - input.null_count())) > AVG_CHAR_BYTES_THRESHOLD) {
    // warp-per-string runs faster for longer strings (but not shorter ones)
    constexpr int block_size = 256;
    cudf::detail::grid_1d grid{input.size() * cudf::detail::warp_size, block_size};
    finder_warp_parallel_fn<TargetIterator, forward>
      <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
        *d_strings, target_itr, start, stop, d_results);
  } else {
    // string-per-thread function
    thrust::transform(rmm::exec_policy(stream),
                      thrust::make_counting_iterator<size_type>(0),
                      thrust::make_counting_iterator<size_type>(input.size()),
                      d_results,
                      finder_fn<TargetIterator, forward>{*d_strings, target_itr, start, stop});
  }
}

template <bool forward = true>
std::unique_ptr<column> find_fn(strings_column_view const& input,
                                string_scalar const& target,
                                size_type start,
                                size_type stop,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(target.is_valid(stream), "Parameter target must be valid.");
  CUDF_EXPECTS(start >= 0, "Parameter start must be positive integer or zero.");
  if ((stop > 0) && (start > stop)) CUDF_FAIL("Parameter start must be less than stop.");

  // create output column
  auto results = make_numeric_column(data_type{type_to_id<size_type>()},
                                     input.size(),
                                     cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                     input.null_count(),
                                     stream,
                                     mr);
  // if input is empty or all-null then we are done
  if (input.size() == input.null_count()) { return results; }

  auto d_target = string_view(target.data(), target.size());

  // special logic for empty target results
  if (d_target.empty()) {
    auto d_strings = column_device_view::create(input.parent(), stream);
    auto d_results = results->mutable_view().data<size_type>();
    thrust::transform(rmm::exec_policy(stream),
                      thrust::counting_iterator<size_type>(0),
                      thrust::counting_iterator<size_type>(input.size()),
                      d_results,
                      empty_target_fn<forward>{*d_strings, start, stop});
    return results;
  }

  // find-utility function fills in the results column
  auto target_itr      = thrust::make_constant_iterator(d_target);
  using TargetIterator = decltype(target_itr);
  find_utility<TargetIterator, forward>(input, target_itr, *results, start, stop, stream);
  results->set_null_count(input.null_count());
  return results;
}
}  // namespace

std::unique_ptr<column> find(strings_column_view const& input,
                             string_scalar const& target,
                             size_type start,
                             size_type stop,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  return find_fn<true>(input, target, start, stop, stream, mr);
}

std::unique_ptr<column> rfind(strings_column_view const& input,
                              string_scalar const& target,
                              size_type start,
                              size_type stop,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  return find_fn<false>(input, target, start, stop, stream, mr);
}

template <bool forward = true>
std::unique_ptr<column> find(strings_column_view const& input,
                             strings_column_view const& target,
                             size_type start,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(start >= 0, "Parameter start must be positive integer or zero.");
  CUDF_EXPECTS(input.size() == target.size(), "input and target columns must be the same size");

  // create output column
  auto results = make_numeric_column(
    data_type{type_to_id<size_type>()}, input.size(), rmm::device_buffer{}, 0, stream, mr);
  // if input is empty or all-null then we are done
  if (input.size() == input.null_count()) { return results; }

  // call find utility with target iterator
  auto d_targets  = column_device_view::create(target.parent(), stream);
  auto target_itr = cudf::detail::make_null_replacement_iterator<string_view>(
    *d_targets, string_view{}, target.has_nulls());
  find_utility<decltype(target_itr), forward>(input, target_itr, *results, start, -1, stream);

  // AND the bitmasks from input and target
  auto [null_mask, null_count] =
    cudf::detail::bitmask_and(table_view({input.parent(), target.parent()}), stream, mr);
  results->set_null_mask(std::move(null_mask), null_count);
  return results;
}

}  // namespace detail

// external APIs

std::unique_ptr<column> find(strings_column_view const& strings,
                             string_scalar const& target,
                             size_type start,
                             size_type stop,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::find(strings, target, start, stop, stream, mr);
}

std::unique_ptr<column> rfind(strings_column_view const& strings,
                              string_scalar const& target,
                              size_type start,
                              size_type stop,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::rfind(strings, target, start, stop, stream, mr);
}

std::unique_ptr<column> find(strings_column_view const& input,
                             strings_column_view const& target,
                             size_type start,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::find<true>(input, target, start, stream, mr);
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
CUDF_KERNEL void contains_warp_parallel_fn(column_device_view const d_strings,
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
  // each warp processes 4 starting bytes
  auto constexpr bytes_per_warp = 4;
  auto found                    = false;
  for (auto i = lane_idx * bytes_per_warp;
       !found && ((i + d_target.size_bytes()) <= d_str.size_bytes());
       i += cudf::detail::warp_size * bytes_per_warp) {
    // check the target matches this part of the d_str data
    // this is definitely faster for very long strings > 128B
    for (auto j = 0; j < bytes_per_warp; j++) {
      if (((i + j + d_target.size_bytes()) <= d_str.size_bytes()) &&
          d_target.compare(d_str.data() + i + j, d_target.size_bytes()) == 0) {
        found = true;
      }
    }
  }

  auto const result = warp_reduce(temp_storage).Reduce(found, cub::Max());
  if (lane_idx == 0) { d_results[str_idx] = result; }
}

std::unique_ptr<column> contains_warp_parallel(strings_column_view const& input,
                                               string_scalar const& target,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
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
  if (d_target.empty()) {
    thrust::fill(
      rmm::exec_policy_nosync(stream), results_view.begin<bool>(), results_view.end<bool>(), true);
  } else {
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
 * Each string uses a warp(32 threads) to handle all the targets.
 * Each thread uses num_targets bools shared memory to store temp result for each lane.
 */
CUDF_KERNEL void multi_contains_warp_parallel_multi_scalars_fn(
  column_device_view const d_strings,
  column_device_view const d_targets,
  cudf::device_span<char> const d_target_first_bytes,
  column_device_view const d_target_indexes_for_first_bytes,
  cudf::device_span<bool*> d_results)
{
  auto const num_targets = d_targets.size();
  auto const num_rows    = d_strings.size();

  auto const idx = static_cast<size_type>(threadIdx.x + blockIdx.x * blockDim.x);
  if (idx >= (num_rows * cudf::detail::warp_size)) { return; }

  auto const lane_idx = idx % cudf::detail::warp_size;
  auto const str_idx  = idx / cudf::detail::warp_size;
  if (d_strings.is_null(str_idx)) { return; }  // bitmask will set result to null.
  // get the string for this warp
  auto const d_str = d_strings.element<string_view>(str_idx);

  /**
   * size of shared_bools = Min(targets_size * block_size, target_group * block_size)
   * each thread uses targets_size bools
   */
  extern __shared__ bool shared_bools[];

  // initialize temp result:
  // set true if target is empty, set false otherwise
  for (int target_idx = 0; target_idx < num_targets; target_idx++) {
    auto const d_target = d_targets.element<string_view>(target_idx);
    shared_bools[threadIdx.x * num_targets + target_idx] = d_target.size_bytes() == 0;
  }

  for (size_type str_byte_idx = lane_idx; str_byte_idx < d_str.size_bytes();
       str_byte_idx += cudf::detail::warp_size) {
    // 1. check the first chars using binary search on first char set
    char c = *(d_str.data() + str_byte_idx);
    auto first_byte_ptr =
      thrust::lower_bound(thrust::seq, d_target_first_bytes.begin(), d_target_first_bytes.end(), c);
    if (not(first_byte_ptr != d_target_first_bytes.end() && *first_byte_ptr == c)) {
      // first char is not matched for all targets, already set result as found
      continue;
    }

    // 2. check the 2nd chars
    int first_char_index_in_list = first_byte_ptr - d_target_first_bytes.begin();
    // get possible targets
    auto const possible_targets_list =
      cudf::list_device_view{d_target_indexes_for_first_bytes, first_char_index_in_list};
    for (auto list_idx = 0; list_idx < possible_targets_list.size();
         ++list_idx) {  // iterate possible targets
      auto target_idx     = possible_targets_list.element<size_type>(list_idx);
      int temp_result_idx = threadIdx.x * num_targets + target_idx;
      if (!shared_bools[temp_result_idx]) {  // not found before
        auto const d_target = d_targets.element<string_view>(target_idx);
        if (d_str.size_bytes() - str_byte_idx >= d_target.size_bytes()) {
          // first char already checked, only need to check the [2nd, end) chars if has.
          bool found = true;
          for (auto i = 1; i < d_target.size_bytes(); i++) {
            if (*(d_str.data() + str_byte_idx + i) != *(d_target.data() + i)) {
              found = false;
              break;
            }
          }
          if (found) { shared_bools[temp_result_idx] = true; }
        }
      }
    }
  }

  // wait all lanes are done in a warp
  __syncwarp();

  if (lane_idx == 0) {
    for (int target_idx = 0; target_idx < num_targets; target_idx++) {
      bool found = false;
      for (int lane_idx = 0; lane_idx < cudf::detail::warp_size; lane_idx++) {
        bool temp_idx = (str_idx * cudf::detail::warp_size + lane_idx) * num_targets + target_idx;
        if (shared_bools[temp_idx]) {
          found = true;
          break;
        }
      }
      d_results[target_idx][str_idx] = found;
    }
  }
}

CUDF_KERNEL void multi_contains_using_indexes_fn(
  column_device_view const d_strings,
  column_device_view const d_targets,
  cudf::device_span<char> const d_target_first_bytes,
  column_device_view const d_target_indexes_for_first_bytes,
  cudf::device_span<bool*> d_results)
{
  auto const str_idx     = static_cast<size_type>(cudf::detail::grid_1d::global_thread_id());
  auto const num_targets = d_targets.size();
  auto const num_rows    = d_strings.size();
  if (str_idx >= num_rows) { return; }
  if (d_strings.is_null(str_idx)) { return; }  // bitmask will set result to null.
  auto const d_str = d_strings.element<string_view>(str_idx);

  // check empty target, the result of searching empty target is true.
  for (auto target_idx = 0; target_idx < num_targets; ++target_idx) {
    auto const d_target            = d_targets.element<string_view>(target_idx);
    d_results[target_idx][str_idx] = d_target.size_bytes() == 0;
  }

  for (auto str_byte_idx = 0; str_byte_idx < d_str.size_bytes();
       ++str_byte_idx) {  // iterate the start index in the string

    // 1. check the first chars using binary search on first char set
    char c = *(d_str.data() + str_byte_idx);
    auto first_byte_ptr =
      thrust::lower_bound(thrust::seq, d_target_first_bytes.begin(), d_target_first_bytes.end(), c);

    if (not(first_byte_ptr != d_target_first_bytes.end() && *first_byte_ptr == c)) {
      // For non-empty targets: no need to search for `str_byte_idx` position, because first char is
      // unmatched. For empty targets: already set result as found.
      continue;
    }

    int first_char_index_in_list = first_byte_ptr - d_target_first_bytes.begin();
    // get possible targets
    auto const possible_targets_list =
      cudf::list_device_view{d_target_indexes_for_first_bytes, first_char_index_in_list};

    for (auto list_idx = 0; list_idx < possible_targets_list.size();
         ++list_idx) {  // iterate possible targets
      auto target_idx = possible_targets_list.element<size_type>(list_idx);
      if (!d_results[target_idx][str_idx]) {  // not found before
        auto const d_target = d_targets.element<string_view>(target_idx);
        if (d_str.size_bytes() - str_byte_idx >= d_target.size_bytes()) {
          // first char already checked, only need to check the [2nd, end) chars if has.
          bool found = true;
          for (auto i = 1; i < d_target.size_bytes(); i++) {
            if (*(d_str.data() + str_byte_idx + i) != *(d_target.data() + i)) {
              found = false;
              break;
            }
          }
          if (found) { d_results[target_idx][str_idx] = true; }
        }
      }
    }
  }
}

/**
 * Execute multi contains for short strings
 * First index the first char for all targets.
 * Index the first char:
 *   collect first char for all targets and do uniq and sort,
 *   then index the targets for the first char.
 *   e.g.:
 *     targets: xa xb ac ad af
 *     first char set is: (a, x)
 *     index result is:
 *       {
 *         a: [2, 3, 4],   // indexes for: ac ad af
 *         x: [0, 1]       // indexes for: xa xb
 *       }
 * when do searching:
 *   find (binary search) from `first char set` for a char in string:
 *     if char in string is not in ['a', 'x'], fast skip
 *     if char in string is 'x', then only need to try ["xa", "xb"] targets.
 *     if char in string is 'a', then only need to try ["ac", "ad", "af"] targets.
 *
 */
std::vector<std::unique_ptr<column>> multi_contains(bool warp_parallel,
                                                    strings_column_view const& input,
                                                    strings_column_view const& targets,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::mr::device_memory_resource* mr)
{
  auto const num_targets = static_cast<size_type>(targets.size());
  CUDF_EXPECTS(not targets.is_empty(), "Must specify at least one target string.");

  // 1. copy targets from device to host
  auto const h_targets_child = cudf::detail::make_std_vector_sync<char>(
    cudf::device_span<char const>(targets.chars_begin(stream), targets.chars_size(stream)), stream);
  auto const targets_offsets   = targets.offsets();
  auto const h_targets_offsets = cudf::detail::make_std_vector_sync(
    cudf::device_span<int const>{targets_offsets.data<int>(),
                                 static_cast<size_t>(targets_offsets.size())},
    stream);

  // 2. index the first characters in targets
  // 2.1 collect first characters in targets
  thrust::host_vector<char> h_first_bytes = {};
  for (auto i = 0; i < targets.size(); i++) {
    auto target_begin_offset = h_targets_offsets[i];
    auto target_end_offset   = h_targets_offsets[i + 1];
    if (target_end_offset - target_begin_offset > 0) {
      char first_char = h_targets_child[target_begin_offset];
      auto no_exist =
        thrust::find(h_first_bytes.begin(), h_first_bytes.end(), first_char) == h_first_bytes.end();
      if (no_exist) { h_first_bytes.push_back(first_char); }
    }
  }

  // 2.2 sort the first characters
  thrust::sort(h_first_bytes.begin(), h_first_bytes.end());

  // 2.3 generate indexes: map from `first char in target` to `target indexes`
  thrust::host_vector<size_type> h_offsets  = {0};
  thrust::host_vector<size_type> h_elements = {};
  for (size_t i = 0; i < h_first_bytes.size(); i++) {
    auto expected_first_byte = h_first_bytes[i];
    for (auto target_idx = 0; target_idx < targets.size(); target_idx++) {
      auto target_begin_offset = h_targets_offsets[target_idx];
      auto target_end_offset   = h_targets_offsets[target_idx + 1];
      if (target_end_offset - target_begin_offset > 0) {
        char curr_first_byte = h_targets_child[target_begin_offset];
        if (expected_first_byte == curr_first_byte) { h_elements.push_back(target_idx); }
      }
    }
    h_offsets.push_back(h_elements.size());
  }

  // 2.4 copy first char set and first char indexes to device
  auto d_first_bytes  = cudf::detail::make_device_uvector_async(h_first_bytes, stream, mr);
  auto d_offsets      = cudf::detail::make_device_uvector_async(h_offsets, stream, mr);
  auto d_elements     = cudf::detail::make_device_uvector_async(h_elements, stream, mr);
  auto offsets_column = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                       h_offsets.size(),
                                                       d_offsets.release(),
                                                       rmm::device_buffer{},  // null mask
                                                       0                      // null size
  );
  auto element_column = std::make_unique<cudf::column>(cudf::data_type{cudf::type_id::INT32},
                                                       h_elements.size(),
                                                       d_elements.release(),
                                                       rmm::device_buffer{},  // null mask
                                                       0                      // null size
  );
  auto list_column    = cudf::make_lists_column(h_first_bytes.size(),
                                             std::move(offsets_column),
                                             std::move(element_column),
                                             0,                     // null count
                                             rmm::device_buffer{},  // null mask
                                             stream,
                                             mr);
  auto d_list_column  = column_device_view::create(list_column->view(), stream);

  // 3. Create output columns.
  auto const results_iter =
    thrust::make_transform_iterator(thrust::counting_iterator<cudf::size_type>(0), [&](int i) {
      return make_numeric_column(data_type{type_id::BOOL8},
                                 input.size(),
                                 cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                 input.null_count(),
                                 stream,
                                 mr);
    });
  auto results_list =
    std::vector<std::unique_ptr<column>>(results_iter, results_iter + targets.size());
  auto device_results_list = [&] {
    auto host_results_pointer_iter =
      thrust::make_transform_iterator(results_list.begin(), [](auto const& results_column) {
        return results_column->mutable_view().template data<bool>();
      });
    auto host_results_pointers = std::vector<bool*>(
      host_results_pointer_iter, host_results_pointer_iter + results_list.size());
    return cudf::detail::make_device_uvector_async(host_results_pointers, stream, mr);
  }();

  auto const d_strings = column_device_view::create(input.parent(), stream);
  auto const d_targets = column_device_view::create(targets.parent(), stream);

  constexpr int block_size = 256;
  cudf::detail::grid_1d grid{input.size(), block_size};

  if (warp_parallel) {
    int shared_mem_size = block_size * targets.size();
    multi_contains_warp_parallel_multi_scalars_fn<<<grid.num_blocks,
                                                    grid.num_threads_per_block,
                                                    shared_mem_size,
                                                    stream.value()>>>(
      *d_strings, *d_targets, d_first_bytes, *d_list_column, device_results_list);
  } else {
    multi_contains_using_indexes_fn<<<grid.num_blocks,
                                      grid.num_threads_per_block,
                                      0,
                                      stream.value()>>>(
      *d_strings, *d_targets, d_first_bytes, *d_list_column, device_results_list);
  }

  return results_list;
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
                                    rmm::device_async_resource_ref mr)
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
                      return !d_strings.is_null(idx) &&
                             bool{pfn(d_strings.element<string_view>(idx), d_target)};
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
                                    rmm::device_async_resource_ref mr)
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

std::unique_ptr<column> contains_small_strings_impl(strings_column_view const& input,
                                                    string_scalar const& target,
                                                    rmm::cuda_stream_view stream,
                                                    rmm::device_async_resource_ref mr)
{
  auto pfn = [] __device__(string_view d_string, string_view d_target) {
    return d_string.find(d_target) != string_view::npos;
  };
  return contains_fn(input, target, pfn, stream, mr);
}
}  // namespace

std::unique_ptr<column> contains(strings_column_view const& input,
                                 string_scalar const& target,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  // use warp parallel when the average string width is greater than the threshold
  if ((input.null_count() < input.size()) &&
      ((input.chars_size(stream) / input.size()) > AVG_CHAR_BYTES_THRESHOLD)) {
    return contains_warp_parallel(input, target, stream, mr);
  }

  // benchmark measurements showed this to be faster for smaller strings
  return contains_small_strings_impl(input, target, stream, mr);
}

std::unique_ptr<table> multi_contains(strings_column_view const& input,
                                      strings_column_view const& targets,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(not targets.has_nulls(), "Target strings cannot be null");
  auto result_columns = [&] {
    if ((input.null_count() < input.size()) &&
        ((input.chars_size(stream) / input.size()) > AVG_CHAR_BYTES_THRESHOLD)) {
      // Large strings.
      // use warp parallel when the average string width is greater than the threshold
      return multi_contains(/**warp parallel**/ true, input, targets, stream, mr);
    } else {
      // Small strings. Searching for multiple targets in one thread seems to work fastest.
      return multi_contains(/**warp parallel**/ false, input, targets, stream, mr);
    }
  }();
  return std::make_unique<table>(std::move(result_columns));
}

std::unique_ptr<column> contains(strings_column_view const& strings,
                                 strings_column_view const& targets,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  auto pfn = [] __device__(string_view d_string, string_view d_target) {
    return d_string.find(d_target) != string_view::npos;
  };
  return contains_fn(strings, targets, pfn, stream, mr);
}

std::unique_ptr<column> starts_with(strings_column_view const& strings,
                                    string_scalar const& target,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
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
                                    rmm::device_async_resource_ref mr)
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
                                  rmm::device_async_resource_ref mr)
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
                                  rmm::device_async_resource_ref mr)
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
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::contains(strings, target, stream, mr);
}

std::unique_ptr<table> multi_contains(strings_column_view const& strings,
                                      strings_column_view const& targets,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::multi_contains(strings, targets, stream, mr);
}

std::unique_ptr<column> contains(strings_column_view const& strings,
                                 strings_column_view const& targets,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::contains(strings, targets, stream, mr);
}

std::unique_ptr<column> starts_with(strings_column_view const& strings,
                                    string_scalar const& target,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::starts_with(strings, target, stream, mr);
}

std::unique_ptr<column> starts_with(strings_column_view const& strings,
                                    strings_column_view const& targets,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::starts_with(strings, targets, stream, mr);
}

std::unique_ptr<column> ends_with(strings_column_view const& strings,
                                  string_scalar const& target,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::ends_with(strings, target, stream, mr);
}

std::unique_ptr<column> ends_with(strings_column_view const& strings,
                                  strings_column_view const& targets,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::ends_with(strings, targets, stream, mr);
}

}  // namespace strings
}  // namespace cudf
