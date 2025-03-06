/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.
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

#include "text/bpe/byte_pair_encoding.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/algorithm.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/functional.hpp>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/split/split.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvtext/byte_pair_encoding.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuco/static_map.cuh>
#include <cuda/functional>
#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/merge.h>
#include <thrust/pair.h>
#include <thrust/remove.h>
#include <thrust/unique.h>

#include <cstdint>
#include <type_traits>

namespace nvtext {

/**
 * @brief Access the bpe_merge_pairs impl member
 *
 * This is used by the encoder to access the impl member functions.
 *
 * @param bpe The merge pairs struct
 * @return The impl object with detailed, internal member data
 */
bpe_merge_pairs::bpe_merge_pairs_impl const* get_bpe_merge_pairs_impl(bpe_merge_pairs const& bpe)
{
  return bpe.impl;
}

namespace detail {
namespace {

using string_hasher_type = cudf::hashing::detail::MurmurHash3_x86_32<cudf::string_view>;
using hash_value_type    = string_hasher_type::result_type;
using merge_pair_type    = thrust::pair<cudf::string_view, cudf::string_view>;
using cuco_storage       = cuco::storage<1>;

/**
 * @brief Hasher function used for building and using the cuco static-map
 *
 * This takes advantage of heterogeneous lookup feature in cuco static-map which
 * allows inserting with one type (index) and looking up with a different type (merge_pair_type).
 *
 * The merge-pairs are in adjacent rows so each index will access two rows of string values.
 * The hash of each string is combined for the returned result.
 */
struct bpe_hasher {
  cudf::column_device_view const d_strings;
  string_hasher_type hasher{};
  // used by insert
  __device__ hash_value_type operator()(cudf::size_type index) const
  {
    index *= 2;
    auto const lhs = d_strings.element<cudf::string_view>(index);
    auto const rhs = d_strings.element<cudf::string_view>(index + 1);
    return cudf::hashing::detail::hash_combine(hasher(lhs), hasher(rhs));
  }
  // used by find
  __device__ hash_value_type operator()(merge_pair_type const& mp) const
  {
    return cudf::hashing::detail::hash_combine(hasher(mp.first), hasher(mp.second));
  }
};

/**
 * @brief Equal function used for building and using the cuco static-map
 *
 * This takes advantage of heterogeneous lookup feature in cuco static-map which
 * allows inserting with one type (index) and looking up with a different type (merge_pair_type).
 *
 * The merge-pairs are in adjacent rows so each index will access two rows of string values.
 * All rows from the input merge-pairs are unique.
 */
struct bpe_equal {
  cudf::column_device_view const d_strings;
  // used by insert
  __device__ bool operator()(cudf::size_type lhs, cudf::size_type rhs) const noexcept
  {
    return lhs == rhs;  // all rows are unique
  }
  // used by find
  __device__ bool operator()(merge_pair_type const& lhs, cudf::size_type rhs) const noexcept
  {
    rhs *= 2;
    auto const left  = d_strings.element<cudf::string_view>(rhs);
    auto const right = d_strings.element<cudf::string_view>(rhs + 1);
    return (left == lhs.first) && (right == lhs.second);
  }
};

using bpe_probe_scheme = cuco::linear_probing<1, bpe_hasher>;

using merge_pairs_map_type = cuco::static_map<cudf::size_type,
                                              cudf::size_type,
                                              cuco::extent<std::size_t>,
                                              cuda::thread_scope_device,
                                              bpe_equal,
                                              bpe_probe_scheme,
                                              cudf::detail::cuco_allocator<char>,
                                              cuco_storage>;

/**
 * @brief Hasher function used for building and using the cuco static-map
 *
 * This takes advantage of heterogeneous lookup feature in cuco static-map which
 * allows inserting with one type (index) and looking up with a different type (merge_pair_type).
 *
 * Each component of the merge-pairs (left and right) are stored individually in the map.
 */
struct mp_hasher {
  cudf::column_device_view const d_strings;
  string_hasher_type hasher{};
  // used by insert
  __device__ hash_value_type operator()(cudf::size_type index) const
  {
    auto const d_str = d_strings.element<cudf::string_view>(index);
    return hasher(d_str);
  }
  // used by find
  __device__ hash_value_type operator()(cudf::string_view const& d_str) const
  {
    return hasher(d_str);
  }
};

/**
 * @brief Equal function used for building and using the cuco static-map
 *
 * This takes advantage of heterogeneous lookup feature in cuco static-map which
 * allows inserting with one type (index) and looking up with a different type (string).
 */
struct mp_equal {
  cudf::column_device_view const d_strings;
  // used by insert
  __device__ bool operator()(cudf::size_type lhs, cudf::size_type rhs) const noexcept
  {
    auto const left  = d_strings.element<cudf::string_view>(lhs);
    auto const right = d_strings.element<cudf::string_view>(rhs);
    return left == right;
  }
  // used by find
  __device__ bool operator()(cudf::string_view const& lhs, cudf::size_type rhs) const noexcept
  {
    auto const right = d_strings.element<cudf::string_view>(rhs);
    return lhs == right;
  }
};

using mp_probe_scheme = cuco::linear_probing<1, mp_hasher>;

using mp_table_map_type = cuco::static_map<cudf::size_type,
                                           cudf::size_type,
                                           cuco::extent<std::size_t>,
                                           cuda::thread_scope_device,
                                           mp_equal,
                                           mp_probe_scheme,
                                           cudf::detail::cuco_allocator<char>,
                                           cuco_storage>;

std::unique_ptr<detail::merge_pairs_map_type> initialize_merge_pairs_map(
  cudf::column_device_view const& input, rmm::cuda_stream_view stream)
{
  auto merge_pairs_map = std::make_unique<merge_pairs_map_type>(
    static_cast<size_t>(input.size()),
    cuco::empty_key{-1},
    cuco::empty_value{-1},
    bpe_equal{input},
    bpe_probe_scheme{bpe_hasher{input}},
    cuco::thread_scope_device,
    cuco_storage{},
    cudf::detail::cuco_allocator<char>{rmm::mr::polymorphic_allocator<char>{}, stream},
    stream.value());

  auto iter = cudf::detail::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<cuco::pair<cudf::size_type, cudf::size_type>>(
      [] __device__(cudf::size_type idx) { return cuco::make_pair(idx, idx); }));

  merge_pairs_map->insert_async(iter, iter + (input.size() / 2), stream.value());
  std::cout << "initialize_merge_pairs_map=" << (int)cudaStreamSynchronize(stream.value())
            << std::endl;

  return merge_pairs_map;
}

std::unique_ptr<detail::mp_table_map_type> initialize_mp_table_map(
  cudf::column_device_view const& input, rmm::cuda_stream_view stream)
{
  auto mp_table_map = std::make_unique<mp_table_map_type>(
    static_cast<size_t>(input.size() * 2),
    cuco::empty_key{-1},
    cuco::empty_value{-1},
    mp_equal{input},
    mp_probe_scheme{mp_hasher{input}},
    cuco::thread_scope_device,
    cuco_storage{},
    cudf::detail::cuco_allocator<char>{rmm::mr::polymorphic_allocator<char>{}, stream},
    stream.value());

  auto iter = cudf::detail::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<cuco::pair<cudf::size_type, cudf::size_type>>(
      [] __device__(cudf::size_type idx) { return cuco::make_pair(idx, idx); }));

  mp_table_map->insert_async(iter, iter + input.size(), stream.value());
  std::cout << "initialize_mp_table_map=" << (int)cudaStreamSynchronize(stream.value())
            << std::endl;

  return mp_table_map;
}

std::unique_ptr<bpe_merge_pairs::bpe_merge_pairs_impl> create_bpe_merge_pairs_impl(
  std::unique_ptr<cudf::column>&& input, rmm::cuda_stream_view stream)
{
  auto d_input      = cudf::column_device_view::create(input->view(), stream);
  auto merge_pairs  = initialize_merge_pairs_map(*d_input, stream);
  auto mp_table_map = initialize_mp_table_map(*d_input, stream);
  return std::make_unique<nvtext::bpe_merge_pairs::bpe_merge_pairs_impl>(
    std::move(input), std::move(d_input), std::move(merge_pairs), std::move(mp_table_map));
}

std::unique_ptr<bpe_merge_pairs::bpe_merge_pairs_impl> create_bpe_merge_pairs_impl(
  cudf::strings_column_view const& input,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto pairs =
    cudf::strings::split_record(input, cudf::string_scalar(" ", true, stream, mr), 1, stream, mr);
  auto content = pairs->release();
  return create_bpe_merge_pairs_impl(std::move(content.children.back()), stream);
}

}  // namespace

std::unique_ptr<bpe_merge_pairs> load_merge_pairs(cudf::strings_column_view const& merge_pairs,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(!merge_pairs.is_empty(), "Merge pairs must not be empty");
  CUDF_EXPECTS(!merge_pairs.has_nulls(), "Merge pairs may not contain nulls");
  return std::make_unique<bpe_merge_pairs>(merge_pairs, stream, mr);
}

}  // namespace detail

// since column_device_view::create() returns is a little more than
// std::unique_ptr<column_device_view> this helper simplifies the return type for us
using col_device_view = std::invoke_result_t<decltype(&cudf::column_device_view::create),
                                             cudf::column_view,
                                             rmm::cuda_stream_view>;

struct bpe_merge_pairs::bpe_merge_pairs_impl {
  std::unique_ptr<cudf::column> const merge_pairs;
  col_device_view const d_merge_pairs;
  std::unique_ptr<detail::merge_pairs_map_type> merge_pairs_map;  // for BPE
  std::unique_ptr<detail::mp_table_map_type> mp_table_map;        // for locating unpairables

  bpe_merge_pairs_impl(std::unique_ptr<cudf::column>&& merge_pairs,
                       col_device_view&& d_merge_pairs,
                       std::unique_ptr<detail::merge_pairs_map_type>&& merge_pairs_map,
                       std::unique_ptr<detail::mp_table_map_type>&& mp_table_map);

  auto const get_merge_pairs() const { return *d_merge_pairs; }
  auto get_merge_pairs_ref() const { return merge_pairs_map->ref(cuco::op::find); }
  auto get_mp_table_ref() const { return mp_table_map->ref(cuco::op::find); }
};

std::unique_ptr<bpe_merge_pairs> load_merge_pairs(cudf::strings_column_view const& merge_pairs,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::load_merge_pairs(merge_pairs, stream, mr);
}

bpe_merge_pairs::bpe_merge_pairs_impl::bpe_merge_pairs_impl(
  std::unique_ptr<cudf::column>&& merge_pairs,
  std::unique_ptr<cudf::column_device_view, std::function<void(cudf::column_device_view*)>>&&
    d_merge_pairs,
  std::unique_ptr<detail::merge_pairs_map_type>&& merge_pairs_map,
  std::unique_ptr<detail::mp_table_map_type>&& mp_table_map)
  : merge_pairs(std::move(merge_pairs)),
    d_merge_pairs(std::move(d_merge_pairs)),
    merge_pairs_map(std::move(merge_pairs_map)),
    mp_table_map(std::move(mp_table_map))
{
}

bpe_merge_pairs::bpe_merge_pairs(std::unique_ptr<cudf::column>&& input,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref)
  : impl(detail::create_bpe_merge_pairs_impl(std::move(input), stream).release())
{
}

bpe_merge_pairs::bpe_merge_pairs(cudf::strings_column_view const& input,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
  : impl(detail::create_bpe_merge_pairs_impl(input, stream, mr).release())
{
}

bpe_merge_pairs::bpe_merge_pairs() = default;
bpe_merge_pairs::~bpe_merge_pairs() { delete impl; }

namespace detail {

namespace {

constexpr int block_size = 512;

/**
 * @brief Produces offsets to unpairable locations in the given chars array
 *
 * Launched as a thread per byte of the chars array.
 * The output is non-zero offsets to locations of unpairable substrings.
 * An unpairable substring does not exist in the given map and so will
 * never be paired. Fortunately, this can be used as an artificial
 * boundary providing increased parallelism in the BPE kernel.
 *
 * @tparam MapRefType The type of the map finder object
 */
template <typename MapRefType>
struct bpe_unpairable_offsets_fn {
  cudf::device_span<char const> d_chars;
  int64_t offset;
  MapRefType const d_map;
  __device__ int64_t operator()(int64_t idx)
  {
    if (!cudf::strings::detail::is_begin_utf8_char(d_chars[idx])) { return 0; }

    auto const itr  = d_chars.data() + idx;
    auto const end  = d_chars.end();
    auto const lhs  = cudf::string_view(itr, cudf::strings::detail::bytes_in_utf8_byte(*itr));
    auto const next = itr + lhs.size_bytes();
    auto output     = 0L;
    if (next < end) {
      auto const rhs = cudf::string_view(next, cudf::strings::detail::bytes_in_utf8_byte(*next));
      // see if both halves exist anywhere in the table, if not these are unpairable
      if (d_map.find(lhs) == d_map.end() && d_map.find(rhs) == d_map.end()) {
        output = idx + lhs.size_bytes() + offset;  // offset for artificial boundary
      }
      printf(
        "%ld: [%c/%d],[%c/%d]=%ld\n", idx, *itr, lhs.size_bytes(), *next, rhs.size_bytes(), output);
    }
    return output;
  }
};

/**
 * @brief Performs byte-pair-encoding
 *
 * Computes the locations where the separator will be inserted in `d_spaces_data`.
 * This is launched as a string per block.
 *
 * The process first initializes all characters to 1 per position in `d_spaces_data`.
 * All pairs are realized and their ranks stored in `d_ranks_data`.
 *
 * Iteratively, the minimum rank is located, the corresponding `d_spaces_data` location
 * is set to 0 resulting in new potential pairs. The process repeats accounting for
 * the rank of the newly formed pairs.
 *
 * Once there are no more rankable pairs, the process finishes and the `d_spaces_data`
 * values identify the location to insert the separator.
 *
 * @tparam MapRefType The type of the map finder object
 * @param d_strings Input data
 * @param d_map For looking up individual string candidates
 * @param d_spaces_data Output the location where separator will be inserted
 * @param d_ranks_data Working memory to hold pair ranks
 * @param d_rerank_data Working memory to hold locations where reranking is required
 */
template <typename MapRefType>
CUDF_KERNEL void bpe_parallel_fn(cudf::column_device_view const d_strings,
                                 char const* d_input_chars,
                                 MapRefType const d_map,
                                 int8_t* d_spaces_data,          // working memory
                                 cudf::size_type* d_ranks_data,  // more working memory
                                 int8_t* d_rerank_data           // and one more working memory
)
{
  // string per block
  auto const str_idx =
    static_cast<cudf::size_type>(cudf::detail::grid_1d::global_thread_id() / block_size);
  auto const lane_idx = static_cast<cudf::size_type>(threadIdx.x);

  auto const d_str  = d_strings.element<cudf::string_view>(str_idx);
  auto const offset = thrust::distance(d_input_chars, d_str.data());

  auto const d_spaces   = d_spaces_data + offset;
  auto const end_spaces = d_spaces + d_str.size_bytes();
  auto const d_ranks    = d_ranks_data + offset;
  auto const end_ranks  = d_ranks + d_str.size_bytes();
  auto const d_rerank   = d_rerank_data + offset;
  auto const end_rerank = d_rerank + d_str.size_bytes();

  auto constexpr max_rank = cuda::std::numeric_limits<cudf::size_type>::max();

  __shared__ cudf::size_type block_min_rank;
  using block_reduce = cub::BlockReduce<cudf::size_type, block_size>;
  __shared__ typename block_reduce::TempStorage temp_storage;
  auto const num_valid = block_size < d_str.size_bytes() ? block_size : d_str.size_bytes();

  // init all the re-rank identifiers to zero
  for (auto itr = d_rerank + lane_idx; itr < end_rerank; itr += block_size) {
    *itr = 0;
  }
  // init all ranks to max
  for (auto itr = d_ranks + lane_idx; itr < end_ranks; itr += block_size) {
    *itr = max_rank;
  }
  // init all spaces to 1 as appropriate
  for (auto itr = d_spaces + lane_idx; itr < end_spaces; itr += block_size) {
    auto const index = thrust::distance(d_spaces, itr);
    *itr = static_cast<int8_t>(cudf::strings::detail::is_begin_utf8_char(d_str.data()[index]));
  }
  __syncthreads();

  // for finding the next half of a pair
  auto next_substr = [d_str, d_spaces, end = end_spaces](int8_t* begin) {
    auto const next = thrust::find(thrust::seq, begin + 1, end, 1);
    auto const size = static_cast<cudf::size_type>(thrust::distance(begin, next));
    return cudf::string_view(d_str.data() + thrust::distance(d_spaces, begin), size);
  };
  // for locating adjacent pairs after merging a pair
  auto find_prev = [begin = d_spaces](int8_t* ptr) {
    while (ptr > begin && *ptr == 0) {
      --ptr;
    }
    return ptr;
  };

  auto min_rank = max_rank;

  // store all the initial ranks for each pair
  // every character but the first one will have a initial rank
  //
  // Example:
  // string:   abcdefghij
  // spaces:   1111111111
  // ranks:    *948516327
  for (auto itr = d_spaces + lane_idx; itr < end_spaces; itr += block_size) {
    if (*itr == 0) { continue; }  // skips any UTF-8 continuation bytes
    // resolve pair and lookup its rank
    auto const lhs      = next_substr(itr);  // retrieve lhs of the pair
    auto const next_itr = itr + lhs.size_bytes();
    if (next_itr < end_spaces) {
      auto const rhs = next_substr(next_itr);  // retrieve rhs of the pair
      if (!rhs.empty()) {
        auto rank          = max_rank;
        auto const mp      = merge_pair_type{lhs, rhs};
        auto const map_itr = d_map.find(mp);                     // lookup pair in merges table;
        if (map_itr != d_map.end()) { rank = map_itr->second; }  // found a match;
        d_ranks[thrust::distance(d_spaces, next_itr)] = rank;    // store the rank
        if (rank < min_rank) { min_rank = rank; }
      }
    }
  }
  // compute the min rank across the block
  auto const reduce_rank =
    block_reduce(temp_storage).Reduce(min_rank, cudf::detail::minimum{}, num_valid);
  if (lane_idx == 0) {
    block_min_rank = reduce_rank;
    // printf("%d: min_rank=%d\n", str_idx, block_min_rank);
  }
  __syncthreads();

  // loop through the ranks processing the current minimum until there are no more
  while (block_min_rank < max_rank) {
    // search the d_ranks for matches to block_min_rank
    for (auto itr = d_ranks + lane_idx; itr < end_ranks; itr += block_size) {
      if (*itr == block_min_rank) {
        auto ptr = itr - 1;  // check for adjacent min-rank (edge-case)
        while (ptr > d_ranks && *ptr == max_rank) {
          --ptr;
        }
        // set the output value to 0 at this position (erases separator, merges pair)
        // using example string above, the min-rank is 1 at position 5
        // string: abcdefghij
        // spaces: 1111101111  (set position 5 to 0)
        if (*ptr != block_min_rank) { d_spaces[thrust::distance(d_ranks, itr)] = 0; }
      }
    }
    __syncthreads();

    // identify all the re-rank locations (logic above invalidated adjacent pairs)
    // using example string above, the adjacent pairs have to be re-ranked
    // string: abcdefghij
    // spaces: 1111101111 (pair 'e,f' is now merged)
    // rerank: 0000101000 ('ef' and 'fg' need re-ranking as 'd,ef' and 'ef,g'
    for (auto itr = d_ranks + lane_idx; itr < end_ranks; itr += block_size) {
      auto const index = thrust::distance(d_ranks, itr);
      if (*itr == block_min_rank && d_spaces[index] == 0) {
        // find previous pair mid-point
        auto ptr = find_prev(d_spaces + index - 1);
        if (ptr > d_spaces) { d_rerank[thrust::distance(d_spaces, ptr)] = 1; }
        // find next pair mid-point
        ptr = thrust::find(thrust::seq, d_spaces + index + 1, end_spaces, 1);
        if (ptr < end_spaces) { d_rerank[thrust::distance(d_spaces, ptr)] = 1; }
        *itr = max_rank;  // reset this rank
      }
    }
    __syncthreads();

    // compute the ranks for the newly created pairs
    min_rank = max_rank;  // and record the new minimum along the way
    for (auto itr = d_rerank + lane_idx; itr < end_rerank; itr += block_size) {
      auto const index = thrust::distance(d_rerank, itr);
      auto rank        = d_ranks[index];
      if (*itr) {
        *itr = 0;  // reset re-rank
        // build lhs of pair
        auto const ptr  = find_prev(d_spaces + index - 1);
        auto const size = static_cast<cudf::size_type>(thrust::distance(ptr, d_spaces + index));
        auto const lhs  = cudf::string_view(d_str.data() + thrust::distance(d_spaces, ptr), size);
        auto const rhs  = next_substr(d_spaces + index);  // retrieve rhs of pair
        rank            = max_rank;
        if (!rhs.empty()) {
          auto const mp      = merge_pair_type{lhs, rhs};
          auto const map_itr = d_map.find(mp);                     // lookup rank for this pair;
          if (map_itr != d_map.end()) { rank = map_itr->second; }  // found a match
        }
        d_ranks[index] = rank;  // store new rank
      }
      if (rank < min_rank) { min_rank = rank; }
    }

    // re-compute the minimum rank across the block (since new pairs are created above)
    auto const reduce_rank =
      block_reduce(temp_storage).Reduce(min_rank, cudf::detail::minimum{}, num_valid);
    if (lane_idx == 0) { block_min_rank = reduce_rank; }
    __syncthreads();
  }  // if no min ranks are found we are done, otherwise start again
}

/**
 * @brief Computes the output size of each strings row
 *
 * This launches as a string per block.
 * The non-zero values in `d_spaces_data` for each string is added to
 * the current string size to produce the total output bytes.
 *
 * @param d_strings Input data
 * @param d_spaces_data Output the location where separator will be inserted
 * @param d_sizes Output sizes of each row
 */
CUDF_KERNEL void bpe_finalize(cudf::column_device_view const d_strings,
                              char const* d_input_chars,
                              int8_t* d_spaces_data,    // where separators are inserted
                              cudf::size_type* d_sizes  // output sizes of encoded strings
)
{
  // string per block
  auto const str_idx =
    static_cast<cudf::size_type>(cudf::detail::grid_1d::global_thread_id() / block_size);
  auto const lane_idx = static_cast<cudf::size_type>(threadIdx.x);

  if (d_strings.is_null(str_idx)) {
    d_sizes[str_idx] = 0;
    return;
  }
  auto const d_str = d_strings.element<cudf::string_view>(str_idx);
  if (d_str.empty()) {
    d_sizes[str_idx] = 0;
    return;
  }

  auto const offset = thrust::distance(d_input_chars, d_str.data());

  auto const d_spaces   = d_spaces_data + offset;
  auto const end_spaces = d_spaces + d_str.size_bytes();
  auto const num_valid  = block_size < d_str.size_bytes() ? block_size : d_str.size_bytes();

  using block_reduce = cub::BlockReduce<cudf::size_type, block_size>;
  __shared__ typename block_reduce::TempStorage temp_storage;

  // reset the first position -- no separator to be added here
  if (lane_idx == 0) { *d_spaces = 0; }

  // compute the output size for this string by counting the resulting separator positions
  auto bytes = 0;
  for (auto itr = d_spaces + lane_idx; itr < end_spaces; itr += block_size) {
    bytes += (*itr > 0);
  }
  auto const total_bytes = block_reduce(temp_storage).Sum(bytes, num_valid);
  if (lane_idx == 0) { d_sizes[str_idx] = total_bytes + d_str.size_bytes(); }
}

}  // namespace

std::unique_ptr<cudf::column> byte_pair_encoding(cudf::strings_column_view const& input,
                                                 bpe_merge_pairs const& merge_pairs,
                                                 cudf::string_scalar const& separator,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  if (input.is_empty() || input.chars_size(stream) == 0) {
    return cudf::make_empty_column(cudf::type_id::STRING);
  }

  CUDF_EXPECTS(separator.is_valid(stream), "separator parameter must be valid");
  auto const d_separator = separator.value(stream);
  CUDF_EXPECTS(d_separator.size_bytes() == 1, "for now, separator must be a single-byte character");

  auto const d_strings = cudf::column_device_view::create(input.parent(), stream);

  auto const first_offset  = (input.offset() == 0) ? 0L
                                                   : cudf::strings::detail::get_offset_value(
                                                      input.offsets(), input.offset(), stream);
  auto const last_offset   = (input.offset() == 0 && input.size() == input.offsets().size() - 1)
                               ? static_cast<int64_t>(input.chars_size(stream))
                               : cudf::strings::detail::get_offset_value(
                                 input.offsets(), input.size() + input.offset(), stream);
  auto const chars_size    = last_offset - first_offset;
  auto const d_input_chars = input.chars_begin(stream) + first_offset;

  rmm::device_uvector<int8_t> d_spaces(chars_size, stream);  // identifies non-merged pairs
  // used for various purposes below: unpairable-offsets, pair ranks, separator insert positions
  rmm::device_uvector<int64_t> d_working(chars_size, stream);

  auto const chars_begin = thrust::counting_iterator<int64_t>(0);
  auto const chars_end   = thrust::counting_iterator<int64_t>(chars_size);

  {
    // this kernel locates unpairable sections of strings to create artificial string row
    // boundaries; the boundary values are recorded as offsets in d_up_offsets
    auto const d_up_offsets = d_working.data();  // store unpairable offsets here
    auto const mp_map = get_bpe_merge_pairs_impl(merge_pairs)->get_mp_table_ref();  // lookup table
    auto const d_chars_span = cudf::device_span<char const>(d_input_chars, chars_size);
    auto up_fn = bpe_unpairable_offsets_fn<decltype(mp_map)>{d_chars_span, first_offset, mp_map};
    thrust::transform(rmm::exec_policy_nosync(stream), chars_begin, chars_end, d_up_offsets, up_fn);
    auto const up_end =  // remove all but the unpairable offsets
      thrust::remove(rmm::exec_policy_nosync(stream), d_up_offsets, d_up_offsets + chars_size, 0L);
    auto const unpairables = thrust::distance(d_up_offsets, up_end);  // number of unpairables

    // new string boundaries created by combining unpairable offsets with the existing offsets
    auto tmp_offsets = rmm::device_uvector<int64_t>(unpairables + input.size() + 1, stream);
    auto input_offsets =
      cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset());
    thrust::merge(rmm::exec_policy_nosync(stream),
                  input_offsets,
                  input_offsets + input.size() + 1,
                  d_up_offsets,
                  up_end,
                  tmp_offsets.begin());
    // remove any adjacent duplicate offsets (i.e. empty or null rows)
    auto const offsets_end =
      thrust::unique(rmm::exec_policy_nosync(stream), tmp_offsets.begin(), tmp_offsets.end());
    auto const offsets_total =
      static_cast<cudf::size_type>(thrust::distance(tmp_offsets.begin(), offsets_end));
    tmp_offsets.resize(offsets_total, stream);

    // temp column created with the merged offsets and the original chars data
    auto const col_offsets = cudf::column_view(cudf::device_span<int64_t const>(tmp_offsets));
    auto const tmp_size    = offsets_total - 1;
    auto const tmp_input   = cudf::column_view(
      input.parent().type(), tmp_size, input.chars_begin(stream), nullptr, 0, 0, {col_offsets});
    auto const d_tmp_strings = cudf::column_device_view::create(tmp_input, stream);

    // launch the byte-pair-encoding kernel on the temp column
    rmm::device_uvector<int8_t> d_rerank(chars_size, stream);  // more working memory;
    rmm::device_uvector<cudf::size_type> d_ranks(chars_size, stream);
    auto const pair_map = get_bpe_merge_pairs_impl(merge_pairs)->get_merge_pairs_ref();
    bpe_parallel_fn<decltype(pair_map)><<<tmp_size, block_size, 0, stream.value()>>>(
      *d_tmp_strings, d_input_chars, pair_map, d_spaces.data(), d_ranks.data(), d_rerank.data());
  }

  // compute the output sizes
  auto output_sizes = rmm::device_uvector<cudf::size_type>(input.size(), stream);
  bpe_finalize<<<input.size(), block_size, 0, stream.value()>>>(
    *d_strings, d_input_chars, d_spaces.data(), output_sizes.data());

  // convert sizes to offsets in-place
  auto [offsets, bytes] = cudf::strings::detail::make_offsets_child_column(
    output_sizes.begin(), output_sizes.end(), stream, mr);

  // build the output: inserting separators to the input character data
  rmm::device_uvector<char> chars(bytes, stream, mr);
  auto d_chars = chars.data();

  auto const d_inserts     = d_working.data();  // stores the insert positions
  auto offsets_at_non_zero = [d_spaces = d_spaces.data()] __device__(auto idx) {
    return d_spaces[idx] > 0;  // separator to be inserted here
  };
  auto const copy_end =
    cudf::detail::copy_if_safe(chars_begin + 1, chars_end, d_inserts, offsets_at_non_zero, stream);

  // this will insert the single-byte separator into positions specified in d_inserts
  auto const sep_char = thrust::constant_iterator<char>(separator.to_string(stream)[0]);
  thrust::merge_by_key(rmm::exec_policy_nosync(stream),
                       d_inserts,      // where to insert separator byte
                       copy_end,       //
                       chars_begin,    // all indices
                       chars_end,      //
                       sep_char,       // byte to insert
                       d_input_chars,  // original data
                       thrust::make_discard_iterator(),
                       d_chars);  // result

  return cudf::make_strings_column(input.size(),
                                   std::move(offsets),
                                   chars.release(),
                                   input.null_count(),
                                   cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

}  // namespace detail

std::unique_ptr<cudf::column> byte_pair_encoding(cudf::strings_column_view const& input,
                                                 bpe_merge_pairs const& merges_table,
                                                 cudf::string_scalar const& separator,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::byte_pair_encoding(input, merges_table, separator, stream, mr);
}

}  // namespace nvtext
