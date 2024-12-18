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

#include "text/utilities/tokenize_ops.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvtext/wordpiece_tokenize.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cub/cub.cuh>
#include <cuco/static_map.cuh>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/logical.h>
#include <thrust/transform.h>

namespace nvtext {
namespace detail {
namespace {

using string_hasher_type = cudf::hashing::detail::MurmurHash3_x86_32<cudf::string_view>;
using hash_value_type    = string_hasher_type::result_type;

struct vocab_hasher {
  cudf::column_device_view const d_strings;
  string_hasher_type hasher{};
  __device__ hash_value_type operator()(cudf::size_type index) const
  {
    return hasher(d_strings.element<cudf::string_view>(index));
  }
  __device__ hash_value_type operator()(cudf::string_view const& s) const { return hasher(s); }
};
struct vocab_equal {
  cudf::column_device_view const d_strings;
  __device__ bool operator()(cudf::size_type lhs, cudf::size_type rhs) const noexcept
  {
    return lhs == rhs;  // all rows are expected to be unique
  }
  __device__ bool operator()(cudf::string_view const& lhs, cudf::size_type rhs) const noexcept
  {
    return d_strings.element<cudf::string_view>(rhs) == lhs;
  }
};

using cuco_storage        = cuco::storage<1>;
using probe_scheme        = cuco::linear_probing<1, vocab_hasher>;
using vocabulary_map_type = cuco::static_map<cudf::size_type,
                                             cudf::size_type,
                                             cuco::extent<std::size_t>,
                                             cuda::thread_scope_device,
                                             vocab_equal,
                                             probe_scheme,
                                             cudf::detail::cuco_allocator<char>,
                                             cuco_storage>;

struct vocab_hasher2 {
  cudf::column_device_view const d_strings;
  string_hasher_type hasher{};
  __device__ hash_value_type operator()(cudf::size_type index) const
  {
    auto const d_str = d_strings.element<cudf::string_view>(index);
    return hasher(cudf::string_view(d_str.data() + 2, d_str.size_bytes() - 2));
  }
  __device__ hash_value_type operator()(cudf::string_view const& s) const { return hasher(s); }
};
struct vocab_equal2 {
  cudf::column_device_view const d_strings;
  __device__ bool operator()(cudf::size_type lhs, cudf::size_type rhs) const noexcept
  {
    return lhs == rhs;  // all rows are expected to be unique
  }
  __device__ bool operator()(cudf::string_view const& lhs, cudf::size_type rhs) const noexcept
  {
    auto const d_str = d_strings.element<cudf::string_view>(rhs);
    return lhs == cudf::string_view(d_str.data() + 2, d_str.size_bytes() - 2);
  }
};

using probe_scheme2        = cuco::linear_probing<1, vocab_hasher2>;
using vocabulary_map_type2 = cuco::static_map<cudf::size_type,
                                              cudf::size_type,
                                              cuco::extent<std::size_t>,
                                              cuda::thread_scope_device,
                                              vocab_equal2,
                                              probe_scheme2,
                                              cudf::detail::cuco_allocator<char>,
                                              cuco_storage>;
}  // namespace
}  // namespace detail

// since column_device_view::create returns is a little more than
// std::unique_ptr<column_device_view> this helper simplifies the return type in a maintainable way
using col_device_view = std::invoke_result_t<decltype(&cudf::column_device_view::create),
                                             cudf::column_view,
                                             rmm::cuda_stream_view>;

struct wordpiece_vocabulary::wordpiece_vocabulary_impl {
  std::unique_ptr<cudf::column> const vocabulary;
  col_device_view const d_vocabulary;
  std::unique_ptr<detail::vocabulary_map_type> vocabulary_map;
  std::unique_ptr<detail::vocabulary_map_type2> vocabulary_map2;

  auto get_map_ref() const { return vocabulary_map->ref(cuco::op::find); }
  auto get_map2_ref() const { return vocabulary_map2->ref(cuco::op::find); }

  wordpiece_vocabulary_impl(std::unique_ptr<cudf::column>&& vocab,
                            col_device_view&& d_vocab,
                            std::unique_ptr<detail::vocabulary_map_type>&& map,
                            std::unique_ptr<detail::vocabulary_map_type2>&& map2)
    : vocabulary(std::move(vocab)),
      d_vocabulary(std::move(d_vocab)),
      vocabulary_map(std::move(map)),
      vocabulary_map2(std::move(map2))
  {
  }
};

namespace {
struct key_pair {
  __device__ auto operator()(cudf::size_type idx) const noexcept
  {
    return cuco::make_pair(idx, idx);
  }
};
struct copy_pieces_fn {
  cudf::column_device_view d_strings;

  __device__ bool operator()(cudf::size_type idx)
  {
    auto const d_str = d_strings.element<cudf::string_view>(idx);
    if (d_str.size_bytes() < 2) { return false; }
    return (d_str.data()[0] == '#') and (d_str.data()[1] == '#');
  }
};
}  // namespace

wordpiece_vocabulary::wordpiece_vocabulary(cudf::strings_column_view const& input,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(not input.is_empty(), "vocabulary must not be empty");
  CUDF_EXPECTS(not input.has_nulls(), "vocabulary must not have nulls");

  // need to hold a copy of the input
  auto vocabulary   = std::make_unique<cudf::column>(input.parent(), stream, mr);
  auto d_vocabulary = cudf::column_device_view::create(vocabulary->view(), stream);

  auto vocab_map = std::make_unique<detail::vocabulary_map_type>(
    static_cast<size_t>(vocabulary->size() * 2),
    cuco::empty_key{-1},
    cuco::empty_value{-1},
    detail::vocab_equal{*d_vocabulary},
    detail::probe_scheme{detail::vocab_hasher{*d_vocabulary}},
    cuco::thread_scope_device,
    detail::cuco_storage{},
    cudf::detail::cuco_allocator<char>{rmm::mr::polymorphic_allocator<char>{}, stream},
    stream.value());
  // the row index is the token id (value for each key in the map)
  auto iter = cudf::detail::make_counting_transform_iterator(0, key_pair{});
  vocab_map->insert_async(iter, iter + vocabulary->size(), stream.value());

  // setup 2nd map with just the ##prefixed items
  // get an index map of all the ##s
  auto map2_indices = rmm::device_uvector<cudf::size_type>(vocabulary->size(), stream);
  auto end          = thrust::copy_if(rmm::exec_policy(stream),
                             thrust::counting_iterator<cudf::size_type>(),
                             thrust::counting_iterator<cudf::size_type>(map2_indices.size()),
                             map2_indices.begin(),
                             copy_pieces_fn{*d_vocabulary});
  map2_indices.resize(thrust::distance(map2_indices.begin(), end), stream);
  // insert them without the ##prefix
  auto vocab_map2 = std::make_unique<detail::vocabulary_map_type2>(
    map2_indices.size() * 2,
    cuco::empty_key{-1},
    cuco::empty_value{-1},
    detail::vocab_equal2{*d_vocabulary},
    detail::probe_scheme2{detail::vocab_hasher2{*d_vocabulary}},
    cuco::thread_scope_device,
    detail::cuco_storage{},
    cudf::detail::cuco_allocator<char>{rmm::mr::polymorphic_allocator<char>{}, stream},
    stream.value());

  auto iter2 = thrust::make_transform_iterator(map2_indices.begin(), key_pair{});
  vocab_map2->insert_async(iter2, iter2 + map2_indices.size(), stream.value());

  _impl = new wordpiece_vocabulary_impl(
    std::move(vocabulary), std::move(d_vocabulary), std::move(vocab_map), std::move(vocab_map2));
}
wordpiece_vocabulary::~wordpiece_vocabulary() { delete _impl; }

std::unique_ptr<wordpiece_vocabulary> load_wordpiece_vocabulary(
  cudf::strings_column_view const& input,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return std::make_unique<wordpiece_vocabulary>(input, stream, mr);
}

namespace detail {
namespace {

template <typename MapRefType, typename MapRefType2>
__device__ cudf::size_type wp_tokenize_fn(cudf::column_device_view const& d_strings,
                                          cudf::string_view token,
                                          MapRefType d_map,
                                          MapRefType2 d_map2,
                                          cudf::size_type* d_tokens)
{
  // if (token.length() >= 200) {  // max word length
  //   d_tokens[token_idx++] = cuda::std::numeric_limits<cudf::size_type>::max();
  //   return token_idx;
  // }

  // lookup token in map
  auto token_idx = 0;
  auto itr       = d_map.find(token);
  if (itr != d_map.end()) {
    d_tokens[token_idx++] = itr->second;
    return token_idx;
  }
  // reduce string by one character and try again
  auto piece = token.substr(0, token.length() - 1);
  while (!piece.empty()) {
    itr = d_map.find(piece);
    if (itr == d_map.end()) {
      piece = piece.substr(0, piece.length() - 1);
      continue;
    }
    d_tokens[token_idx++] = itr->second;
    break;
  }
  if (piece.empty()) {
    // did not find anything; this is uncommon
    auto const unk        = cudf::string_view("[UNK]", 5);
    auto const itr        = d_map.find(unk);
    d_tokens[token_idx++] = itr != d_map.end() ? itr->second : -1;
    return token_idx;
  }

  // token = token.substr(piece.length(), token.length() - piece.length());
  token =
    cudf::string_view(token.data() + piece.size_bytes(), token.size_bytes() - piece.size_bytes());
  piece = token;
  while (!piece.empty()) {
    auto itr = d_map2.find(piece);
    if (itr == d_map2.end()) {
      piece = piece.substr(0, piece.length() - 1);
      continue;
    }
    d_tokens[token_idx++] = itr->second;

    // token = token.substr(piece.length(), token.length() - piece.length());
    token =
      cudf::string_view(token.data() + piece.size_bytes(), token.size_bytes() - piece.size_bytes());
    piece = token;
  }
  if (!token.empty()) {
    // very uncommon
    auto const unk = cudf::string_view("[UNK]", 5);
    auto const itr = d_map.find(unk);
    d_tokens[0]    = itr != d_map.end() ? itr->second : -1;
    // need to reset any previous ids too
    for (auto i = 1; i < token_idx; ++i) {
      d_tokens[i] = cuda::std::numeric_limits<cudf::size_type>::max();
    }
  }

  return token_idx;
}

constexpr int block_size = 256;

template <typename MapRefType, typename MapRefType2>
CUDF_KERNEL void tokenize_kernel(cudf::column_device_view const d_strings,
                                 MapRefType d_map,
                                 MapRefType2 d_map2,
                                 cudf::size_type max_tokens,
                                 cudf::size_type* d_tokens,
                                 cudf::size_type* d_token_counts)
{
  // string per block
  auto const idx     = cudf::detail::grid_1d::global_thread_id();
  auto const str_idx = idx / block_size;
  if (str_idx >= d_strings.size()) { return; }
  d_token_counts[str_idx] = 0;
  if (d_strings.is_null(str_idx)) { return; }
  auto const d_str = d_strings.element<cudf::string_view>(str_idx);
  if (d_str.empty()) { return; }

  auto const begin        = d_str.data();
  auto const end          = begin + d_str.size_bytes();
  auto const d_output     = d_tokens + (idx * max_tokens);
  auto const d_output_end = d_output + max_tokens;

  __shared__ cudf::size_type s_tokens[block_size];
  __shared__ cudf::size_type token_count;
  using block_reduce = cub::BlockReduce<cudf::size_type, block_size>;
  __shared__ typename block_reduce::TempStorage temp_storage;

  auto const lane_idx = idx % block_size;
  // init
  s_tokens[lane_idx] = cuda::std::numeric_limits<cudf::size_type>::max();
  token_count        = 0;
  __syncthreads();

  auto itr  = begin + lane_idx;
  auto oitr = d_output;

  while (token_count < max_tokens) {
    cudf::size_type tcount = 0;
    if (itr < end) {
      auto const ch = *itr;
      if ((ch != ' ') && ((lane_idx == 0) || (*(itr - 1) == ' '))) {
        // at the front edge of a word; find the end
        auto wend = itr + 1;
        while (wend < end && *wend != ' ') {
          ++wend;
        }
        auto bytes = static_cast<cudf::size_type>(thrust::distance(itr, wend));
        // lookup token and place them in (s_tokens+lane_idx)
        auto token = cudf::string_view{itr, bytes};
        tcount     = wp_tokenize_fn(d_strings, token, d_map, d_map2, s_tokens + lane_idx);
      }
    }
    auto count = block_reduce(temp_storage).Sum(tcount);
    __syncthreads();

    if (lane_idx == 0) {
      // need to read the non-max s_tokens into global memory
      // up to the max_tokens and re-init if we are continuing
      count = cuda::std::min(count, max_tokens - token_count);
      for (auto i = 0; i < count && oitr < d_output_end; ++i) {
        if (s_tokens[i] < cuda::std::numeric_limits<cudf::size_type>::max()) {
          *oitr++ = s_tokens[i];
        }
      }
      token_count += count;
    }
    __syncthreads();

    s_tokens[lane_idx] = cuda::std::numeric_limits<cudf::size_type>::max();
    __syncthreads();

    // maybe also keep track of the actual tokens in this string
    // and return that number in order to build the final offsets
  }

  if (lane_idx == 0) { d_token_counts[str_idx] = token_count; }
}

}  // namespace

std::unique_ptr<cudf::column> wordpiece_tokenize(cudf::strings_column_view const& input,
                                                 wordpiece_vocabulary const& vocabulary,
                                                 cudf::size_type max_tokens_per_row,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  auto const output_type = cudf::data_type{cudf::type_to_id<cudf::size_type>()};
  if (input.size() == input.null_count()) { return cudf::make_empty_column(output_type); }
  CUDF_EXPECTS(max_tokens_per_row > 0, "maximum tokens must be greater than 0");

  // count the tokens per string and build the offsets from the counts
  auto const d_strings = cudf::column_device_view::create(input.parent(), stream);
  auto map_ref         = vocabulary._impl->get_map_ref();
  auto map2_ref        = vocabulary._impl->get_map2_ref();

  rmm::device_uvector<cudf::size_type> d_token_counts(input.size(), stream);
  rmm::device_uvector<cudf::size_type> d_tokens(input.size() * max_tokens_per_row, stream);

  // launch block per string to compute tokens
  cudf::detail::grid_1d grid{input.size() * block_size, block_size};
  tokenize_kernel<decltype(map_ref), decltype(map2_ref)>
    <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
      *d_strings, map_ref, map2_ref, max_tokens_per_row, d_tokens.data(), d_token_counts.data());
  auto [token_offsets, total_count] = cudf::detail::make_offsets_child_column(
    d_token_counts.begin(), d_token_counts.end(), stream, mr);

  // d_tokens.resize(total_count, stream);
  auto tokens =
    cudf::make_numeric_column(output_type, total_count, cudf::mask_state::UNALLOCATED, stream, mr);
  cudaMemcpyAsync(tokens->mutable_view().head(),
                  d_tokens.data(),
                  total_count * sizeof(cudf::size_type),
                  cudaMemcpyDeviceToDevice,
                  stream.value());

  return cudf::make_lists_column(input.size(),
                                 std::move(token_offsets),
                                 std::move(tokens),
                                 input.null_count(),
                                 cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                 stream,
                                 mr);
}

}  // namespace detail

std::unique_ptr<cudf::column> wordpiece_tokenize(cudf::strings_column_view const& input,
                                                 wordpiece_vocabulary const& vocabulary,
                                                 cudf::size_type max_tokens_per_row,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::wordpiece_tokenize(input, vocabulary, max_tokens_per_row, stream, mr);
}

}  // namespace nvtext
