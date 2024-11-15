/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.
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
#include <cudf/detail/copy.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sequence.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/hashing/detail/murmurhash3_x64_128.cuh>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/lists/list_device_view.cuh>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvtext/minhash.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cooperative_groups.h>
#include <cuda/atomic>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include <limits>

namespace nvtext {
namespace detail {
namespace {

/**
 * @brief Compute the minhash of each string for each seed
 *
 * This is a warp-per-string algorithm where parallel threads within a warp
 * work on substrings of a single string row.
 *
 * @tparam HashFunction hash function to use on each substring
 *
 * @param d_strings Strings column to process
 * @param seeds Seeds for hashing each string
 * @param width Substring window size in characters
 * @param d_hashes Minhash output values for each string
 */
template <
  typename HashFunction,
  typename hash_value_type = std::
    conditional_t<std::is_same_v<typename HashFunction::result_type, uint32_t>, uint32_t, uint64_t>>
CUDF_KERNEL void minhash_kernel(cudf::column_device_view const d_strings,
                                cudf::device_span<hash_value_type const> seeds,
                                cudf::size_type width,
                                hash_value_type* d_hashes)
{
  auto const idx = static_cast<std::size_t>(threadIdx.x + blockIdx.x * blockDim.x);
  if (idx >= (static_cast<std::size_t>(d_strings.size()) *
              static_cast<std::size_t>(cudf::detail::warp_size))) {
    return;
  }

  auto const str_idx  = static_cast<cudf::size_type>(idx / cudf::detail::warp_size);
  auto const lane_idx = static_cast<cudf::size_type>(idx % cudf::detail::warp_size);

  if (d_strings.is_null(str_idx)) { return; }

  auto const d_str    = d_strings.element<cudf::string_view>(str_idx);
  auto const d_output = d_hashes + (str_idx * seeds.size());

  // initialize hashes output for this string
  if (lane_idx == 0) {
    auto const init = d_str.empty() ? 0 : std::numeric_limits<hash_value_type>::max();
    thrust::fill(thrust::seq, d_output, d_output + seeds.size(), init);
  }
  __syncwarp();

  auto const begin = d_str.data() + lane_idx;
  auto const end   = d_str.data() + d_str.size_bytes();

  // each lane hashes 'width' substrings of d_str
  for (auto itr = begin; itr < end; itr += cudf::detail::warp_size) {
    if (cudf::strings::detail::is_utf8_continuation_char(*itr)) { continue; }
    auto const check_str =  // used for counting 'width' characters
      cudf::string_view(itr, static_cast<cudf::size_type>(thrust::distance(itr, end)));
    auto const [bytes, left] = cudf::strings::detail::bytes_to_character_position(check_str, width);
    if ((itr != d_str.data()) && (left > 0)) { continue; }  // true if past the end of the string

    auto const hash_str = cudf::string_view(itr, bytes);
    // hashing with each seed on the same section of the string is 10x faster than
    // computing the substrings for each seed
    for (std::size_t seed_idx = 0; seed_idx < seeds.size(); ++seed_idx) {
      auto const hasher = HashFunction(seeds[seed_idx]);
      // hash substring and store the min value
      if constexpr (std::is_same_v<hash_value_type, uint32_t>) {
        auto const hvalue = hasher(hash_str);
        cuda::atomic_ref<hash_value_type, cuda::thread_scope_block> ref{*(d_output + seed_idx)};
        ref.fetch_min(hvalue, cuda::std::memory_order_relaxed);
      } else {
        // This code path assumes the use of MurmurHash3_x64_128 which produces 2 uint64 values
        // but only uses the first uint64 value as requested by the LLM team.
        auto const hvalue = thrust::get<0>(hasher(hash_str));
        cuda::atomic_ref<hash_value_type, cuda::thread_scope_block> ref{*(d_output + seed_idx)};
        ref.fetch_min(hvalue, cuda::std::memory_order_relaxed);
      }
    }
  }
}

template <
  typename HashFunction,
  typename hash_value_type = std::
    conditional_t<std::is_same_v<typename HashFunction::result_type, uint32_t>, uint32_t, uint64_t>>
std::unique_ptr<cudf::column> minhash_fn(cudf::strings_column_view const& input,
                                         cudf::device_span<hash_value_type const> seeds,
                                         cudf::size_type width,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(!seeds.empty(), "Parameter seeds cannot be empty", std::invalid_argument);
  CUDF_EXPECTS(width >= 2,
               "Parameter width should be an integer value of 2 or greater",
               std::invalid_argument);
  CUDF_EXPECTS((static_cast<std::size_t>(input.size()) * seeds.size()) <
                 static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()),
               "The number of seeds times the number of input rows exceeds the column size limit",
               std::overflow_error);

  auto const output_type = cudf::data_type{cudf::type_to_id<hash_value_type>()};
  if (input.is_empty()) { return cudf::make_empty_column(output_type); }

  auto const d_strings = cudf::column_device_view::create(input.parent(), stream);

  auto hashes   = cudf::make_numeric_column(output_type,
                                          input.size() * static_cast<cudf::size_type>(seeds.size()),
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          mr);
  auto d_hashes = hashes->mutable_view().data<hash_value_type>();

  constexpr cudf::thread_index_type block_size = 256;
  cudf::detail::grid_1d grid{
    static_cast<cudf::thread_index_type>(input.size()) * cudf::detail::warp_size, block_size};
  minhash_kernel<HashFunction><<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
    *d_strings, seeds, width, d_hashes);

  return hashes;
}

constexpr cudf::thread_index_type block_size = 256;
// for potentially tuning minhash_seed_kernel independently from block_size
constexpr cudf::thread_index_type tile_size = block_size;

// Number of a/b parameter values to process per thread.
// The intermediate values are stored in shared-memory and therefore limits this count.
// This value was found to be an efficient size for both uint32 and uint64
// hash types based on benchmarks.
constexpr cuda::std::size_t params_per_thread = 16;

// Separate kernels are used to process strings above and below this value (in bytes).
constexpr cudf::size_type wide_string_threshold = 1 << 18;  // 256K
// The number of blocks per string for the above-threshold kernel processing.
constexpr cudf::size_type blocks_per_string = 64;
// The above values were determined using the redpajama and books_sample datasets

/**
 * @brief Hashing kernel launched as a thread per tile-size (block or warp)
 *
 * This kernel computes the hashes for each string using the seed and the specified
 * hash function. The width is used to compute rolling substrings to hash over.
 * The hashes are stored in d_hashes to be used in the minhash_permuted_kernel.
 *
 * This kernel also counts the number of strings above the wide_string_threshold
 * and proactively initializes the output values for those strings.
 *
 * @tparam HashFunction The hash function to use for this kernel
 * @tparam hash_value_type Derived from HashFunction result_type
 *
 * @param d_strings The input strings to hash
 * @param seed The seed used for the hash function
 * @param width Width in characters used for determining substrings to hash
 * @param d_hashes The resulting hash values are stored here
 * @param threshold_count Stores the number of strings above wide_string_threshold
 * @param param_count Number of parameters (used for the proactive initialize)
 * @param d_results Final results vector (used for the proactive initialize)
 */
template <typename HashFunction, typename hash_value_type = typename HashFunction::result_type>
CUDF_KERNEL void minhash_seed_kernel(cudf::column_device_view const d_strings,
                                     hash_value_type seed,
                                     cudf::size_type width,
                                     hash_value_type* d_hashes,
                                     cudf::size_type* threshold_count,
                                     cudf::size_type param_count,
                                     hash_value_type* d_results)
{
  auto const tid     = cudf::detail::grid_1d::global_thread_id();
  auto const str_idx = tid / tile_size;
  if (str_idx >= d_strings.size()) { return; }
  if (d_strings.is_null(str_idx)) { return; }

  // retrieve this string's offset to locate the output position in d_hashes
  auto const offsets = d_strings.child(cudf::strings_column_view::offsets_column_index);
  auto const offsets_itr =
    cudf::detail::input_offsetalator(offsets.head(), offsets.type(), d_strings.offset());
  auto const offset     = offsets_itr[str_idx];
  auto const size_bytes = static_cast<cudf::size_type>(offsets_itr[str_idx + 1] - offset);
  if (size_bytes == 0) { return; }

  auto const d_str    = cudf::string_view(d_strings.head<char>() + offset, size_bytes);
  auto const lane_idx = tid % tile_size;

  // hashes for this string/thread are stored here
  auto seed_hashes = d_hashes + offset - offsets_itr[0] + lane_idx;

  auto const begin  = d_str.data() + lane_idx;
  auto const end    = d_str.data() + d_str.size_bytes();
  auto const hasher = HashFunction(seed);

  for (auto itr = begin; itr < end; itr += tile_size, seed_hashes += tile_size) {
    if (cudf::strings::detail::is_utf8_continuation_char(*itr)) {
      *seed_hashes = 0;
      continue;
    }
    auto const check_str =  // used for counting 'width' characters
      cudf::string_view(itr, static_cast<cudf::size_type>(thrust::distance(itr, end)));
    auto const [bytes, left] = cudf::strings::detail::bytes_to_character_position(check_str, width);
    if ((itr != d_str.data()) && (left > 0)) {
      // true itr+width is past the end of the string
      *seed_hashes = 0;
      continue;
    }

    auto const hash_str = cudf::string_view(itr, bytes);
    hash_value_type hv;
    if constexpr (std::is_same_v<hash_value_type, uint32_t>) {
      hv = hasher(hash_str);
    } else {
      hv = thrust::get<0>(hasher(hash_str));
    }
    // disallowing hash to zero case
    *seed_hashes = cuda::std::max(hv, hash_value_type{1});
  }

  // logic appended here so an extra kernel is not required
  if (size_bytes >= wide_string_threshold) {
    if (lane_idx == 0) {
      // count the number of wide strings
      cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> ref{*threshold_count};
      ref.fetch_add(1, cuda::std::memory_order_relaxed);
    }
    // initialize the output -- only needed for wider strings
    auto d_output = d_results + (str_idx * param_count);
    for (auto i = lane_idx; i < param_count; i += tile_size) {
      d_output[i] = std::numeric_limits<hash_value_type>::max();
    }
  }
}

/**
 * @brief Permutation calculation kernel
 *
 * This kernel uses the hashes from the minhash_seed_kernel and the parameter_a and
 * parameter_b values to compute the final output results.
 * The output is the number of input rows (N) by the number of parameter values (M).
 * Each output[i] is the calculated result for parameter_a/b[0:M].
 *
 * This kernel is launched with either blocks per strings of 1 for strings
 * below the wide_strings_threshold or blocks per string = blocks_per_strings
 * for strings above wide_strings_threshold.
 *
 * @tparam hash_value_type Derived from HashFunction result_type
 * @tparam blocks_per_string Number of blocks used to process each string
 *
 * @param d_strings The input strings to hash
 * @param indices The indices of the strings in d_strings to process
 * @param parameter_a 1st set of parameters for the calculation result
 * @param parameter_b 2nd set of parameters for the calculation result
 * @param width Used for calculating the number of available hashes in each string
 * @param d_hashes The hash values computed in minhash_seed_kernel
 * @param d_results Final results vector of calculate values
 */
template <typename hash_value_type, int blocks_per_string>
CUDF_KERNEL void minhash_permuted_kernel(cudf::column_device_view const d_strings,
                                         cudf::device_span<cudf::size_type const> indices,
                                         cudf::device_span<hash_value_type const> parameter_a,
                                         cudf::device_span<hash_value_type const> parameter_b,
                                         cudf::size_type width,
                                         hash_value_type const* d_hashes,
                                         hash_value_type* d_results)
{
  auto const tid = cudf::detail::grid_1d::global_thread_id();
  auto const idx = (tid / blocks_per_string) / block_size;
  if (idx >= indices.size()) { return; }
  auto const str_idx = indices[idx];
  if (d_strings.is_null(str_idx)) { return; }

  auto const block      = cooperative_groups::this_thread_block();
  int const section_idx = block.group_index().x % blocks_per_string;

  auto const offsets = d_strings.child(cudf::strings_column_view::offsets_column_index);
  auto const offsets_itr =
    cudf::detail::input_offsetalator(offsets.head(), offsets.type(), d_strings.offset());
  auto const offset     = offsets_itr[str_idx];
  auto const size_bytes = static_cast<cudf::size_type>(offsets_itr[str_idx + 1] - offset);

  // number of items to process in this block;
  // last block also includes any remainder values from the size_bytes/blocks_per_string truncation
  // example:
  //  each section_size for string with size 588090 and blocks_per_string=64 is 9188
  //  except the last section which is 9188 + (588090 % 64) = 9246
  auto const section_size =
    (size_bytes / blocks_per_string) +
    (section_idx < (blocks_per_string - 1) ? 0 : size_bytes % blocks_per_string);
  auto const section_offset = section_idx * (size_bytes / blocks_per_string);

  // hash values for this block/section
  auto const seed_hashes = d_hashes + offset - offsets_itr[0] + section_offset;
  // width used here as a max value since a string's char-count <= byte-count
  auto const hashes_size =
    section_idx < (blocks_per_string - 1)
      ? section_size
      : cuda::std::max(static_cast<cudf::size_type>(size_bytes > 0), section_size - width + 1);

  auto const init     = size_bytes == 0 ? 0 : std::numeric_limits<hash_value_type>::max();
  auto const lane_idx = block.thread_rank();
  auto const d_output = d_results + (str_idx * parameter_a.size());

  auto const begin = seed_hashes + lane_idx;
  auto const end   = seed_hashes + hashes_size;

  // constants used in the permutation calculations
  constexpr uint64_t mersenne_prime  = (1UL << 61) - 1;
  constexpr hash_value_type hash_max = std::numeric_limits<hash_value_type>::max();

  // found to be an efficient shared memory size for both hash types
  __shared__ hash_value_type block_values[block_size * params_per_thread];

  for (std::size_t i = 0; i < parameter_a.size(); i += params_per_thread) {
    // initialize this block's chunk of shared memory
    // each thread handles params_per_thread of values
    auto const chunk_values = block_values + (lane_idx * params_per_thread);
    thrust::uninitialized_fill(thrust::seq, chunk_values, chunk_values + params_per_thread, init);
    block.sync();

    auto const param_count =
      cuda::std::min(static_cast<cuda::std::size_t>(params_per_thread), parameter_a.size() - i);

    // each lane accumulates min hashes in its shared memory
    for (auto itr = begin; itr < end; itr += block_size) {
      auto const hv = *itr;
      // 0 is used as a skip sentinel for UTF-8 and trailing bytes
      if (hv == 0) { continue; }

      for (std::size_t param_idx = i; param_idx < (i + param_count); ++param_idx) {
        // permutation formula used by datatrove
        hash_value_type const v =
          ((hv * parameter_a[param_idx] + parameter_b[param_idx]) % mersenne_prime) & hash_max;
        auto const block_idx    = ((param_idx % params_per_thread) * block_size) + lane_idx;
        block_values[block_idx] = cuda::std::min(v, block_values[block_idx]);
      }
    }
    block.sync();

    // reduce each parameter values vector to a single min value;
    // assumes that the block_size > params_per_thread;
    // each thread reduces a block_size of parameter values (thread per parameter)
    if (lane_idx < param_count) {
      auto const values = block_values + (lane_idx * block_size);
      // cooperative groups does not have a min function and cub::BlockReduce was slower
      auto const minv =
        thrust::reduce(thrust::seq, values, values + block_size, init, thrust::minimum{});
      if constexpr (blocks_per_string > 1) {
        // accumulates mins for each block into d_output
        cuda::atomic_ref<hash_value_type, cuda::thread_scope_block> ref{d_output[lane_idx + i]};
        ref.fetch_min(minv, cuda::std::memory_order_relaxed);
      } else {
        d_output[lane_idx + i] = minv;
      }
    }
    block.sync();
  }
}

template <typename HashFunction, typename hash_value_type = typename HashFunction::result_type>
std::unique_ptr<cudf::column> minhash_fn(cudf::strings_column_view const& input,
                                         hash_value_type seed,
                                         cudf::device_span<hash_value_type const> parameter_a,
                                         cudf::device_span<hash_value_type const> parameter_b,
                                         cudf::size_type width,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(width >= 2,
               "Parameter width should be an integer value of 2 or greater",
               std::invalid_argument);
  CUDF_EXPECTS(!parameter_a.empty(), "Parameters A and B cannot be empty", std::invalid_argument);
  CUDF_EXPECTS(parameter_a.size() == parameter_b.size(),
               "Parameters A and B should have the same number of elements",
               std::invalid_argument);
  CUDF_EXPECTS(
    (static_cast<std::size_t>(input.size()) * parameter_a.size()) <
      static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()),
    "The number of parameters times the number of input rows exceeds the column size limit",
    std::overflow_error);

  auto const output_type = cudf::data_type{cudf::type_to_id<hash_value_type>()};
  if (input.is_empty()) { return cudf::make_empty_column(output_type); }

  auto const d_strings = cudf::column_device_view::create(input.parent(), stream);

  auto results =
    cudf::make_numeric_column(output_type,
                              input.size() * static_cast<cudf::size_type>(parameter_a.size()),
                              cudf::mask_state::UNALLOCATED,
                              stream,
                              mr);
  auto d_results = results->mutable_view().data<hash_value_type>();

  cudf::detail::grid_1d grid{static_cast<cudf::thread_index_type>(input.size()) * block_size,
                             block_size};
  auto const hashes_size = input.chars_size(stream);
  auto d_hashes          = rmm::device_uvector<hash_value_type>(hashes_size, stream);
  auto d_threshold_count = cudf::detail::device_scalar<cudf::size_type>(0, stream);

  minhash_seed_kernel<HashFunction>
    <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(*d_strings,
                                                                         seed,
                                                                         width,
                                                                         d_hashes.data(),
                                                                         d_threshold_count.data(),
                                                                         parameter_a.size(),
                                                                         d_results);
  auto const threshold_count = d_threshold_count.value(stream);

  auto indices = rmm::device_uvector<cudf::size_type>(input.size(), stream);
  thrust::sequence(rmm::exec_policy(stream), indices.begin(), indices.end());
  cudf::size_type threshold_index = threshold_count < input.size() ? input.size() : 0;

  // if we counted a split of above/below threshold then
  // compute partitions based on the size of each string
  if ((threshold_count > 0) && (threshold_count < input.size())) {
    auto sizes = rmm::device_uvector<cudf::size_type>(input.size(), stream);
    thrust::transform(rmm::exec_policy_nosync(stream),
                      thrust::counting_iterator<cudf::size_type>(0),
                      thrust::counting_iterator<cudf::size_type>(input.size()),
                      sizes.data(),
                      cuda::proclaim_return_type<cudf::size_type>(
                        [d_strings = *d_strings] __device__(auto idx) -> cudf::size_type {
                          if (d_strings.is_null(idx)) { return 0; }
                          return d_strings.element<cudf::string_view>(idx).size_bytes();
                        }));
    thrust::sort_by_key(
      rmm::exec_policy_nosync(stream), sizes.begin(), sizes.end(), indices.begin());
    auto const lb = thrust::lower_bound(
      rmm::exec_policy_nosync(stream), sizes.begin(), sizes.end(), wide_string_threshold);
    threshold_index = static_cast<cudf::size_type>(thrust::distance(sizes.begin(), lb));
  }

  // handle the strings below the threshold width
  if (threshold_index > 0) {
    auto d_indices = cudf::device_span<cudf::size_type const>(indices.data(), threshold_index);
    cudf::detail::grid_1d grid{static_cast<cudf::thread_index_type>(d_indices.size()) * block_size,
                               block_size};
    minhash_permuted_kernel<hash_value_type, 1>
      <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
        *d_strings, d_indices, parameter_a, parameter_b, width, d_hashes.data(), d_results);
  }

  // handle the strings above the threshold width
  if (threshold_index < input.size()) {
    auto const count = static_cast<cudf::thread_index_type>(input.size() - threshold_index);
    auto d_indices =
      cudf::device_span<cudf::size_type const>(indices.data() + threshold_index, count);
    cudf::detail::grid_1d grid{count * block_size * blocks_per_string, block_size};
    minhash_permuted_kernel<hash_value_type, blocks_per_string>
      <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
        *d_strings, d_indices, parameter_a, parameter_b, width, d_hashes.data(), d_results);
  }

  return results;
}

/**
 * @brief Compute the minhash of each list row of strings for each seed
 *
 * This is a warp-per-row algorithm where parallel threads within a warp
 * work on strings in a single list row.
 *
 * @tparam HashFunction hash function to use on each string
 *
 * @param d_input List of strings to process
 * @param seeds Seeds for hashing each string
 * @param d_hashes Minhash output values (one per row)
 */
template <
  typename HashFunction,
  typename hash_value_type = std::
    conditional_t<std::is_same_v<typename HashFunction::result_type, uint32_t>, uint32_t, uint64_t>>
CUDF_KERNEL void minhash_word_kernel(cudf::detail::lists_column_device_view const d_input,
                                     cudf::device_span<hash_value_type const> seeds,
                                     hash_value_type* d_hashes)
{
  auto const idx     = cudf::detail::grid_1d::global_thread_id();
  auto const row_idx = idx / cudf::detail::warp_size;

  if (row_idx >= d_input.size()) { return; }
  if (d_input.is_null(row_idx)) { return; }

  auto const d_row    = cudf::list_device_view(d_input, row_idx);
  auto const d_output = d_hashes + (row_idx * seeds.size());

  // initialize hashes output for this row
  auto const lane_idx = static_cast<cudf::size_type>(idx % cudf::detail::warp_size);
  if (lane_idx == 0) {
    auto const init = d_row.size() == 0 ? 0 : std::numeric_limits<hash_value_type>::max();
    thrust::fill(thrust::seq, d_output, d_output + seeds.size(), init);
  }
  __syncwarp();

  // each lane hashes a string from the input row
  for (auto str_idx = lane_idx; str_idx < d_row.size(); str_idx += cudf::detail::warp_size) {
    auto const hash_str =
      d_row.is_null(str_idx) ? cudf::string_view{} : d_row.element<cudf::string_view>(str_idx);
    for (std::size_t seed_idx = 0; seed_idx < seeds.size(); ++seed_idx) {
      auto const hasher = HashFunction(seeds[seed_idx]);
      // hash string and store the min value
      hash_value_type hv;
      if constexpr (std::is_same_v<hash_value_type, uint32_t>) {
        hv = hasher(hash_str);
      } else {
        // This code path assumes the use of MurmurHash3_x64_128 which produces 2 uint64 values
        // but only uses the first uint64 value as requested by the LLM team.
        hv = thrust::get<0>(hasher(hash_str));
      }
      cuda::atomic_ref<hash_value_type, cuda::thread_scope_block> ref{*(d_output + seed_idx)};
      ref.fetch_min(hv, cuda::std::memory_order_relaxed);
    }
  }
}

template <
  typename HashFunction,
  typename hash_value_type = std::
    conditional_t<std::is_same_v<typename HashFunction::result_type, uint32_t>, uint32_t, uint64_t>>
std::unique_ptr<cudf::column> word_minhash_fn(cudf::lists_column_view const& input,
                                              cudf::device_span<hash_value_type const> seeds,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(!seeds.empty(), "Parameter seeds cannot be empty", std::invalid_argument);
  CUDF_EXPECTS((static_cast<std::size_t>(input.size()) * seeds.size()) <
                 static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()),
               "The number of seeds times the number of input rows exceeds the column size limit",
               std::overflow_error);

  auto const output_type = cudf::data_type{cudf::type_to_id<hash_value_type>()};
  if (input.is_empty()) { return cudf::make_empty_column(output_type); }

  auto const d_input = cudf::column_device_view::create(input.parent(), stream);

  auto hashes   = cudf::make_numeric_column(output_type,
                                          input.size() * static_cast<cudf::size_type>(seeds.size()),
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          mr);
  auto d_hashes = hashes->mutable_view().data<hash_value_type>();
  auto lcdv     = cudf::detail::lists_column_device_view(*d_input);

  constexpr cudf::thread_index_type block_size = 256;
  cudf::detail::grid_1d grid{
    static_cast<cudf::thread_index_type>(input.size()) * cudf::detail::warp_size, block_size};
  minhash_word_kernel<HashFunction>
    <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(lcdv, seeds, d_hashes);

  return hashes;
}

std::unique_ptr<cudf::column> build_list_result(cudf::column_view const& input,
                                                std::unique_ptr<cudf::column>&& hashes,
                                                cudf::size_type seeds_size,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr)
{
  // build the offsets for the output lists column
  auto const zero = cudf::numeric_scalar<cudf::size_type>(0, true, stream);
  auto const size = cudf::numeric_scalar<cudf::size_type>(seeds_size, true, stream);
  auto offsets    = cudf::detail::sequence(input.size() + 1, zero, size, stream, mr);
  hashes->set_null_mask(rmm::device_buffer{}, 0);  // children have no nulls

  // build the lists column from the offsets and the hashes
  auto result = make_lists_column(input.size(),
                                  std::move(offsets),
                                  std::move(hashes),
                                  input.null_count(),
                                  cudf::detail::copy_bitmask(input, stream, mr),
                                  stream,
                                  mr);
  // expect this condition to be very rare
  if (input.null_count() > 0) {
    result = cudf::detail::purge_nonempty_nulls(result->view(), stream, mr);
  }
  return result;
}
}  // namespace

std::unique_ptr<cudf::column> minhash(cudf::strings_column_view const& input,
                                      cudf::numeric_scalar<uint32_t> const& seed,
                                      cudf::size_type width,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  using HashFunction = cudf::hashing::detail::MurmurHash3_x86_32<cudf::string_view>;
  auto const seeds   = cudf::device_span<uint32_t const>{seed.data(), 1};
  auto hashes        = detail::minhash_fn<HashFunction>(input, seeds, width, stream, mr);
  hashes->set_null_mask(cudf::detail::copy_bitmask(input.parent(), stream, mr), input.null_count());
  return hashes;
}

std::unique_ptr<cudf::column> minhash(cudf::strings_column_view const& input,
                                      cudf::device_span<uint32_t const> seeds,
                                      cudf::size_type width,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  using HashFunction = cudf::hashing::detail::MurmurHash3_x86_32<cudf::string_view>;
  auto hashes        = detail::minhash_fn<HashFunction>(input, seeds, width, stream, mr);
  return build_list_result(input.parent(), std::move(hashes), seeds.size(), stream, mr);
}

std::unique_ptr<cudf::column> minhash(cudf::strings_column_view const& input,
                                      uint32_t seed,
                                      cudf::device_span<uint32_t const> parameter_a,
                                      cudf::device_span<uint32_t const> parameter_b,
                                      cudf::size_type width,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  using HashFunction = cudf::hashing::detail::MurmurHash3_x86_32<cudf::string_view>;
  auto hashes =
    detail::minhash_fn<HashFunction>(input, seed, parameter_a, parameter_b, width, stream, mr);
  return build_list_result(input.parent(), std::move(hashes), parameter_a.size(), stream, mr);
}

std::unique_ptr<cudf::column> minhash64(cudf::strings_column_view const& input,
                                        cudf::numeric_scalar<uint64_t> const& seed,
                                        cudf::size_type width,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  using HashFunction = cudf::hashing::detail::MurmurHash3_x64_128<cudf::string_view>;
  auto const seeds   = cudf::device_span<uint64_t const>{seed.data(), 1};
  auto hashes        = detail::minhash_fn<HashFunction>(input, seeds, width, stream, mr);
  hashes->set_null_mask(cudf::detail::copy_bitmask(input.parent(), stream, mr), input.null_count());
  return hashes;
}

std::unique_ptr<cudf::column> minhash64(cudf::strings_column_view const& input,
                                        cudf::device_span<uint64_t const> seeds,
                                        cudf::size_type width,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  using HashFunction = cudf::hashing::detail::MurmurHash3_x64_128<cudf::string_view>;
  auto hashes        = detail::minhash_fn<HashFunction>(input, seeds, width, stream, mr);
  return build_list_result(input.parent(), std::move(hashes), seeds.size(), stream, mr);
}

std::unique_ptr<cudf::column> minhash64(cudf::strings_column_view const& input,
                                        uint64_t seed,
                                        cudf::device_span<uint64_t const> parameter_a,
                                        cudf::device_span<uint64_t const> parameter_b,
                                        cudf::size_type width,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  using HashFunction = cudf::hashing::detail::MurmurHash3_x64_128<cudf::string_view>;
  auto hashes =
    detail::minhash_fn<HashFunction>(input, seed, parameter_a, parameter_b, width, stream, mr);
  return build_list_result(input.parent(), std::move(hashes), parameter_a.size(), stream, mr);
}

std::unique_ptr<cudf::column> word_minhash(cudf::lists_column_view const& input,
                                           cudf::device_span<uint32_t const> seeds,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  using HashFunction = cudf::hashing::detail::MurmurHash3_x86_32<cudf::string_view>;
  auto hashes        = detail::word_minhash_fn<HashFunction>(input, seeds, stream, mr);
  return build_list_result(input.parent(), std::move(hashes), seeds.size(), stream, mr);
}

std::unique_ptr<cudf::column> word_minhash64(cudf::lists_column_view const& input,
                                             cudf::device_span<uint64_t const> seeds,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  using HashFunction = cudf::hashing::detail::MurmurHash3_x64_128<cudf::string_view>;
  auto hashes        = detail::word_minhash_fn<HashFunction>(input, seeds, stream, mr);
  return build_list_result(input.parent(), std::move(hashes), seeds.size(), stream, mr);
}
}  // namespace detail

std::unique_ptr<cudf::column> minhash(cudf::strings_column_view const& input,
                                      cudf::numeric_scalar<uint32_t> seed,
                                      cudf::size_type width,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::minhash(input, seed, width, stream, mr);
}

std::unique_ptr<cudf::column> minhash(cudf::strings_column_view const& input,
                                      cudf::device_span<uint32_t const> seeds,
                                      cudf::size_type width,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::minhash(input, seeds, width, stream, mr);
}

std::unique_ptr<cudf::column> minhash_permuted(cudf::strings_column_view const& input,
                                               uint32_t seed,
                                               cudf::device_span<uint32_t const> parameter_a,
                                               cudf::device_span<uint32_t const> parameter_b,
                                               cudf::size_type width,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::minhash(input, seed, parameter_a, parameter_b, width, stream, mr);
}

std::unique_ptr<cudf::column> minhash64(cudf::strings_column_view const& input,
                                        cudf::numeric_scalar<uint64_t> seed,
                                        cudf::size_type width,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::minhash64(input, seed, width, stream, mr);
}

std::unique_ptr<cudf::column> minhash64(cudf::strings_column_view const& input,
                                        cudf::device_span<uint64_t const> seeds,
                                        cudf::size_type width,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::minhash64(input, seeds, width, stream, mr);
}

std::unique_ptr<cudf::column> minhash64_permuted(cudf::strings_column_view const& input,
                                                 uint64_t seed,
                                                 cudf::device_span<uint64_t const> parameter_a,
                                                 cudf::device_span<uint64_t const> parameter_b,
                                                 cudf::size_type width,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::minhash64(input, seed, parameter_a, parameter_b, width, stream, mr);
}

std::unique_ptr<cudf::column> word_minhash(cudf::lists_column_view const& input,
                                           cudf::device_span<uint32_t const> seeds,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::word_minhash(input, seeds, stream, mr);
}

std::unique_ptr<cudf::column> word_minhash64(cudf::lists_column_view const& input,
                                             cudf::device_span<uint64_t const> seeds,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::word_minhash64(input, seeds, stream, mr);
}
}  // namespace nvtext
