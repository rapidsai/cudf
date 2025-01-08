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

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/cuco_helpers.hpp>
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
#include <thrust/find.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/remove.h>

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

constexpr int block_size = 64;
// constexpr int tile_size = 32;
constexpr auto no_token = cuda::std::numeric_limits<cudf::size_type>::max();

template <typename MapRefType, typename MapRefType2>
__device__ cudf::size_type wp_tokenize_fn(cudf::column_device_view const& d_strings,
                                          cudf::string_view word,
                                          MapRefType d_map,
                                          MapRefType2 d_map2,
                                          cudf::size_type* d_tokens)
{
  // lookup word in map
  auto token_idx = 0;
  auto itr       = d_map.find(word);
  if (itr != d_map.end()) {
    d_tokens[token_idx++] = itr->second;
    return token_idx;
  }
  // reduce word by one character and try again
  auto piece = word.substr(0, word.length() - 1);
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
    d_tokens[token_idx++] = itr != d_map.end() ? itr->second : no_token;
    // printf("[unk] %c %d\n", word.data()[0], token_idx);
    return token_idx;
  }

  // token = token.substr(piece.length(), token.length() - piece.length());
  word =
    cudf::string_view(word.data() + piece.size_bytes(), word.size_bytes() - piece.size_bytes());
  piece = word;
  while (!piece.empty()) {
    auto itr = d_map2.find(piece);
    if (itr == d_map2.end()) {
      piece = piece.substr(0, piece.length() - 1);
      continue;
    }
    d_tokens[token_idx++] = itr->second;

    word =
      cudf::string_view(word.data() + piece.size_bytes(), word.size_bytes() - piece.size_bytes());
    piece = word;
  }
  if (!word.empty()) {
    // very uncommon
    auto const unk = cudf::string_view("[UNK]", 5);
    auto const itr = d_map.find(unk);
    d_tokens[0]    = itr != d_map.end() ? itr->second : no_token;
    // need to reset any previous ids too
    for (auto i = 1; i < token_idx; ++i) {
      d_tokens[i] = no_token;
    }
    // printf("<unk> %c %d\n", word.data()[0], token_idx);
    token_idx = 1;
  }

  return token_idx;
}

[[maybe_unused]] __device__ void print_s(int id, cudf::string_view d_str)
{
  char str[24];
  auto len = min(d_str.size_bytes(), (int)sizeof(str) - 1);
  memcpy(str, d_str.data(), len);
  str[len] = 0;
  printf("%d: [%s]\n", id, str);
}

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
  auto const d_output     = d_tokens + (str_idx * max_tokens);
  auto const d_output_end = d_output + max_tokens;

  __shared__ cudf::size_type start_words[block_size];
  __shared__ cudf::size_type end_words[block_size];
  constexpr int tokens_size = block_size * 2;
  __shared__ cudf::size_type s_tokens[tokens_size];
  __shared__ cudf::size_type token_count;
  __shared__ cudf::size_type byte_count;
  __shared__ cudf::size_type words_found;
  using block_reduce = cub::BlockReduce<cudf::size_type, block_size>;
  __shared__ typename block_reduce::TempStorage temp_storage;

  auto const lane_idx     = idx % block_size;
  constexpr auto no_token = cuda::std::numeric_limits<cudf::size_type>::max();

  // if (lane_idx == 0) { printf("%ld: %d bytes\n", str_idx, d_str.size_bytes()); }

  // initialize data
  token_count = 0;
  byte_count  = 0;
  words_found = 0;
  for (auto i = lane_idx; i < max_tokens; i += block_size) {
    d_output[i] = no_token;
  }
  __syncthreads();

  auto first_token  = no_token;
  auto first_length = no_token;

  auto itr  = begin + lane_idx;
  auto oitr = d_output;

  // each thread processes one byte of the d_str;
  // continue until all bytes have been consumed or the max token count has been reached
  while (token_count < max_tokens && byte_count < d_str.size_bytes()) {
    //
    s_tokens[lane_idx]              = no_token;
    s_tokens[lane_idx + block_size] = no_token;

    start_words[lane_idx] = lane_idx == 0 ? first_token : no_token;
    end_words[lane_idx]   = lane_idx == 0 ? first_length : no_token;
    __syncthreads();

    // look for word beginnings and store the position/sizes into lane-indexed arrays
    // if (itr < end) {
    auto k = lane_idx / 2;
    for (int j = 0; j < 2 && itr < end; ++j) {
      if ((*itr != ' ') && ((itr == begin) || (*(itr - 1) == ' '))) {
        // itr is the front edge of a word; find its end
        auto const word_end = thrust::find(thrust::seq, itr, end, ' ');
        start_words[k]      = static_cast<cudf::size_type>(thrust::distance(begin, itr));
        end_words[k]        = static_cast<cudf::size_type>(thrust::distance(itr, word_end));
        // printf("%ld/%ld: (%d,%d)\n", lane_idx, k, start_words[k], end_words[k]);
      }
      k += (block_size / 2);
      itr += block_size;
    }
    __syncthreads();

    if (lane_idx == 0) {
      thrust::remove(thrust::seq, start_words, start_words + block_size, no_token);
      auto wend   = thrust::remove(thrust::seq, end_words, end_words + block_size, no_token);
      words_found = static_cast<int>(thrust::distance(end_words, wend));
      thrust::exclusive_scan(thrust::seq, end_words, end_words + words_found + 1, end_words);
      // printf("%ld: wf=%d\n", str_idx, words_found);
    }
    __syncthreads();

    cudf::size_type word_tokens = 0;

    // each thread now processes a word
    if (lane_idx < words_found) {
      auto const word_pos = start_words[lane_idx];
      auto const offset   = end_words[lane_idx];  // these are offsets now
      auto const size     = end_words[lane_idx + 1] - offset;
      if ((offset + size) <= tokens_size) {
        // lookup token(s) for this word and place them in (s_tokens+lane_idx)
        auto word   = cudf::string_view{d_str.data() + word_pos, size};
        word_tokens = wp_tokenize_fn(d_strings, word, d_map, d_map2, s_tokens + offset);
        // if (str_idx == 0) { print_s(word_tokens, word); }
      }
    }
    auto count = block_reduce(temp_storage).Sum(word_tokens);

    if (lane_idx == 0) {
      // printf("%ld: tc=%d\n", str_idx, count);
      //  read the valid s_tokens into global memory
      for (auto i = 0; (i < tokens_size) && (oitr < d_output_end); ++i) {
        if (s_tokens[i] != no_token) { *oitr++ = s_tokens[i]; }
      }
      token_count += cuda::std::min(count, max_tokens - token_count);
      byte_count += block_size * 2;
      auto last_offset = end_words[words_found];
      first_token =
        (last_offset > tokens_size) && (words_found > 0) ? start_words[words_found - 1] : no_token;
      first_length = (last_offset > tokens_size) && (words_found > 1)
                       ? end_words[words_found - 1] - end_words[words_found - 2]
                       : no_token;
    }
    __syncthreads();
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

  auto tokens =
    cudf::make_numeric_column(output_type, total_count, cudf::mask_state::UNALLOCATED, stream, mr);
  auto output         = tokens->mutable_view().begin<cudf::size_type>();
  auto const sentinel = std::numeric_limits<cudf::size_type>::max();
  thrust::remove_copy(rmm::exec_policy(stream), d_tokens.begin(), d_tokens.end(), output, sentinel);

  return cudf::make_lists_column(input.size(),
                                 std::move(token_offsets),
                                 std::move(tokens),
                                 input.null_count(),
                                 cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                 stream,
                                 mr);
}

std::unique_ptr<cudf::column> wordpiece_tokenize2(cudf::strings_column_view const& input,
                                                  wordpiece_vocabulary const& vocabulary,
                                                  cudf::size_type max_tokens_per_row,
                                                  rmm::cuda_stream_view stream,
                                                  rmm::device_async_resource_ref mr)
{
  auto const output_type = cudf::data_type{cudf::type_to_id<cudf::size_type>()};
  if (input.size() == input.null_count()) { return cudf::make_empty_column(output_type); }
  CUDF_EXPECTS(max_tokens_per_row > 0, "maximum tokens must be greater than 0");

  auto const first_offset  = (input.offset() == 0) ? 0
                                                   : cudf::strings::detail::get_offset_value(
                                                      input.offsets(), input.offset(), stream);
  auto const last_offset   = (input.offset() == 0 && input.size() == input.offsets().size() - 1)
                               ? input.chars_size(stream)
                               : cudf::strings::detail::get_offset_value(
                                 input.offsets(), input.size() + input.offset(), stream);
  auto const chars_size    = last_offset - first_offset;
  auto const d_input_chars = input.chars_begin(stream) + first_offset;

  rmm::device_uvector<cudf::size_type> d_token_counts(input.size(), stream);
  // rmm::device_uvector<cudf::size_type> d_tokens(input.size() * max_tokens_per_row, stream);

  auto const d_strings = cudf::column_device_view::create(input.parent(), stream);

  // find the word-edges and merge them with the offsets
  rmm::device_uvector<int64_t> d_edges(chars_size / 2, stream);
  auto edges_end =
    thrust::copy_if(rmm::exec_policy(stream),
                    thrust::make_counting_iterator<int64_t>(0),
                    thrust::counting_iterator<int64_t>(chars_size),
                    d_edges.begin(),
                    [d_input_chars] __device__(auto idx) {
                      if (idx == 0) { return false; }
                      return (d_input_chars[idx] != ' ' && d_input_chars[idx - 1] == ' ');
                    });

  auto edges = thrust::distance(d_edges.begin(), edges_end);
  CUDF_EXPECTS(edges + input.size() + 1 < std::numeric_limits<cudf::size_type>::max(),
               "output exceeds column size limit");
  auto tmp_offsets = rmm::device_uvector<int64_t>(edges + input.size() + 1, stream);
  auto input_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset());
  thrust::merge(rmm::exec_policy_nosync(stream),
                input_offsets,
                input_offsets + input.size() + 1,
                d_edges.begin(),
                edges_end,
                tmp_offsets.begin());

  rmm::device_uvector<int64_t> d_tokens(chars_size / 2, stream);
  auto map_ref  = vocabulary._impl->get_map_ref();
  auto map2_ref = vocabulary._impl->get_map2_ref();

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
