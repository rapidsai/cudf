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

#include <cuda/atomic>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include <limits>

namespace nvtext {
namespace detail {
namespace {

constexpr cudf::thread_index_type block_size = 256;
// for tuning independently from block_size
constexpr cudf::thread_index_type tile_size = block_size;

/**
 * @brief Compute the minhash of each string for each seed
 *
 * This is a block-per-string algorithm where parallel threads within a block
 * work on a single string row.
 *
 * @tparam HashFunction hash function to use on each substring
 *
 * @param d_strings Strings column to process
 * @param seeds Seeds for hashing each string
 * @param width Substring window size in characters
 * @param working_memory Memory used to hold intermediate hash values
 * @param d_hashes Minhash output values for each string
 */
template <
  typename HashFunction,
  typename hash_value_type = std::
    conditional_t<std::is_same_v<typename HashFunction::result_type, uint32_t>, uint32_t, uint64_t>>
CUDF_KERNEL void minhash_kernel(cudf::column_device_view const d_strings,
                                cudf::device_span<hash_value_type const> seeds,
                                cudf::size_type width,
                                hash_value_type* working_memory,
                                hash_value_type* d_hashes)
{
  auto const idx     = cudf::detail::grid_1d::global_thread_id();
  auto const str_idx = idx / tile_size;
  if (str_idx >= d_strings.size()) { return; }
  if (d_strings.is_null(str_idx)) { return; }

  auto const d_str    = d_strings.element<cudf::string_view>(str_idx);
  auto const init     = d_str.empty() ? 0 : std::numeric_limits<hash_value_type>::max();
  auto const lane_idx = idx % tile_size;

  auto tile_hashes = working_memory + (str_idx * tile_size * seeds.size());

  // initialize working memory
  for (std::size_t seed_idx = lane_idx; seed_idx < seeds.size(); seed_idx += tile_size) {
    auto begin = tile_hashes + (seed_idx * tile_size);
    thrust::uninitialized_fill(thrust::seq, begin, begin + tile_size, init);
  }
  __syncthreads();

  auto const d_output = d_hashes + (str_idx * seeds.size());

  auto const begin = d_str.data() + lane_idx;
  auto const end   = d_str.data() + d_str.size_bytes();

  // each lane hashes 'width' substrings of d_str
  for (auto itr = begin; itr < end; itr += tile_size) {
    if (cudf::strings::detail::is_utf8_continuation_char(*itr)) { continue; }
    auto const check_str =  // used for counting 'width' characters
      cudf::string_view(itr, static_cast<cudf::size_type>(thrust::distance(itr, end)));
    auto const [bytes, left] = cudf::strings::detail::bytes_to_character_position(check_str, width);
    if ((itr != d_str.data()) && (left > 0)) { continue; }  // true if past the end of the string

    auto const hash_str = cudf::string_view(itr, bytes);
    for (std::size_t seed_idx = 0; seed_idx < seeds.size(); ++seed_idx) {
      auto const hasher = HashFunction(seeds[seed_idx]);
      hash_value_type hv;
      if constexpr (std::is_same_v<hash_value_type, uint32_t>) {
        hv = hasher(hash_str);
      } else {
        hv = thrust::get<0>(hasher(hash_str));
      }
      tile_hashes[(seed_idx * tile_size) + lane_idx] =
        cuda::std::min(hv, tile_hashes[(seed_idx * tile_size) + lane_idx]);
    }
  }
  __syncthreads();

  // compute final result
  for (std::size_t seed_idx = lane_idx; seed_idx < seeds.size(); seed_idx += tile_size) {
    auto begin = tile_hashes + (seed_idx * tile_size);
    auto hv    = thrust::reduce(thrust::seq, begin, begin + tile_size, init, thrust::minimum{});
    d_output[seed_idx] = hv;
  }
}

template <
  typename HashFunction,
  typename hash_value_type = std::
    conditional_t<std::is_same_v<typename HashFunction::result_type, uint32_t>, uint32_t, uint64_t>>
CUDF_KERNEL void minhash_permuted_kernel(cudf::column_device_view const d_strings,
                                         cudf::device_span<hash_value_type const> parmA,
                                         cudf::device_span<hash_value_type const> parmB,
                                         cudf::size_type width,
                                         hash_value_type* d_hashes)
{
  auto const idx     = cudf::detail::grid_1d::global_thread_id();
  auto const str_idx = idx / tile_size;
  if (str_idx >= d_strings.size()) { return; }
  if (d_strings.is_null(str_idx)) { return; }

  auto const d_str    = d_strings.element<cudf::string_view>(str_idx);
  auto const init     = d_str.empty() ? 0 : std::numeric_limits<hash_value_type>::max();
  auto const lane_idx = idx % tile_size;

  auto const d_output = d_hashes + (str_idx * parmA.size());

  auto const begin = d_str.data() + (lane_idx);
  auto const end   = d_str.data() + d_str.size_bytes();

  constexpr std::size_t seed_chunk   = 16;  // based on block-size==256
  constexpr uint64_t mersenne_prime  = (1UL << 61) - 1;
  constexpr hash_value_type hash_max = std::numeric_limits<hash_value_type>::max();

  extern __shared__ char shmem[];
  auto const block_hashes = reinterpret_cast<hash_value_type*>(shmem);

  for (std::size_t i = 0; i < parmA.size(); i += seed_chunk) {
    // initialize working memory
    auto const tile_hashes = block_hashes + (lane_idx * seed_chunk);
    thrust::uninitialized_fill(thrust::seq, tile_hashes, tile_hashes + seed_chunk, init);
    __syncthreads();

    auto const seed_count = cuda::std::min(seed_chunk, parmA.size() - i);

    // each lane hashes 'width' substrings of d_str
    for (auto itr = begin; itr < end; itr += tile_size) {
      if (cudf::strings::detail::is_utf8_continuation_char(*itr)) { continue; }
      auto const check_str =  // used for counting 'width' characters
        cudf::string_view(itr, static_cast<cudf::size_type>(thrust::distance(itr, end)));
      auto const [bytes, left] =
        cudf::strings::detail::bytes_to_character_position(check_str, width);
      if ((itr != d_str.data()) && (left > 0)) { continue; }  // true if past the end of the string

      auto const hash_str = cudf::string_view(itr, bytes);
      auto const hasher   = HashFunction(parmA[0]);
      hash_value_type hv1;
      if constexpr (std::is_same_v<hash_value_type, uint32_t>) {
        hv1 = hasher(hash_str);
      } else {
        hv1 = thrust::get<0>(hasher(hash_str));
      }

      for (std::size_t seed_idx = i; seed_idx < (i + seed_count); ++seed_idx) {
        hash_value_type const hv =
          seed_idx == 0 ? hv1
                        : ((hv1 * parmA[seed_idx] + parmB[seed_idx]) % mersenne_prime) & hash_max;
        auto const block_idx    = ((seed_idx % seed_chunk) * tile_size) + lane_idx;
        block_hashes[block_idx] = cuda::std::min(hv, block_hashes[block_idx]);
      }
    }
    __syncthreads();

    if (lane_idx < seed_count) {
      auto const hvs = block_hashes + (lane_idx * tile_size);
      auto const hv  = thrust::reduce(thrust::seq, hvs, hvs + tile_size, init, thrust::minimum{});
      d_output[lane_idx + i] = hv;
    }
    __syncthreads();
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

  auto const wm_size  = cudf::util::round_up_safe(seeds.size() * tile_size * input.size(),
                                                 static_cast<std::size_t>(block_size));
  auto working_memory = rmm::device_uvector<hash_value_type>(wm_size, stream);

  cudf::detail::grid_1d grid{static_cast<cudf::thread_index_type>(input.size()) * tile_size,
                             block_size};
  minhash_kernel<HashFunction><<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
    *d_strings, seeds, width, working_memory.data(), d_hashes);

  return hashes;
}

template <
  typename HashFunction,
  typename hash_value_type = std::
    conditional_t<std::is_same_v<typename HashFunction::result_type, uint32_t>, uint32_t, uint64_t>>
std::unique_ptr<cudf::column> minhash_fn(cudf::strings_column_view const& input,
                                         cudf::device_span<hash_value_type const> parmA,
                                         cudf::device_span<hash_value_type const> parmB,
                                         cudf::size_type width,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(!parmA.empty(), "Parameters A and B cannot be empty", std::invalid_argument);
  CUDF_EXPECTS(width >= 2,
               "Parameter width should be an integer value of 2 or greater",
               std::invalid_argument);
  CUDF_EXPECTS((static_cast<std::size_t>(input.size()) * parmA.size()) <
                 static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()),
               "The number of seeds times the number of input rows exceeds the column size limit",
               std::overflow_error);
  CUDF_EXPECTS(parmA.size() == parmB.size(),
               "Parameters A and B should have the same number of elements",
               std::invalid_argument);

  auto const output_type = cudf::data_type{cudf::type_to_id<hash_value_type>()};
  if (input.is_empty()) { return cudf::make_empty_column(output_type); }

  auto const d_strings = cudf::column_device_view::create(input.parent(), stream);

  auto hashes   = cudf::make_numeric_column(output_type,
                                          input.size() * static_cast<cudf::size_type>(parmA.size()),
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          mr);
  auto d_hashes = hashes->mutable_view().data<hash_value_type>();

  // 16 seeds can be held in shared-memory: 32K/block_size(256)/sizeof(hash_value_type) = ~16
  auto const shmem_size = block_size * 16 * sizeof(hash_value_type);

  cudf::detail::grid_1d grid{static_cast<cudf::thread_index_type>(input.size()) * tile_size,
                             block_size};
  minhash_permuted_kernel<HashFunction>
    <<<grid.num_blocks, grid.num_threads_per_block, shmem_size, stream.value()>>>(
      *d_strings, parmA, parmB, width, d_hashes);

  return hashes;
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
                                      cudf::device_span<uint32_t const> parmA,
                                      cudf::device_span<uint32_t const> parmB,
                                      cudf::size_type width,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  using HashFunction = cudf::hashing::detail::MurmurHash3_x86_32<cudf::string_view>;
  auto hashes        = detail::minhash_fn<HashFunction>(input, parmA, parmB, width, stream, mr);
  return build_list_result(input.parent(), std::move(hashes), parmA.size(), stream, mr);
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
                                        cudf::device_span<uint64_t const> parmA,
                                        cudf::device_span<uint64_t const> parmB,
                                        cudf::size_type width,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  using HashFunction = cudf::hashing::detail::MurmurHash3_x64_128<cudf::string_view>;
  auto hashes        = detail::minhash_fn<HashFunction>(input, parmA, parmB, width, stream, mr);
  return build_list_result(input.parent(), std::move(hashes), parmA.size(), stream, mr);
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
                                               cudf::device_span<uint32_t const> parmA,
                                               cudf::device_span<uint32_t const> parmB,
                                               cudf::size_type width,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::minhash(input, parmA, parmB, width, stream, mr);
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
                                                 cudf::device_span<uint64_t const> parmA,
                                                 cudf::device_span<uint64_t const> parmB,
                                                 cudf::size_type width,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::minhash64(input, parmA, parmB, width, stream, mr);
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
