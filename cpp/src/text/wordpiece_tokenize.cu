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
#include <cudf/detail/utilities/algorithm.cuh>
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

#include <cooperative_groups.h>
#include <cooperative_groups/scan.h>
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
                                             cuda::thread_scope_thread,
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
                                              cuda::thread_scope_thread,
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
  cudf::size_type unk_id{};

  auto get_map_ref() const { return vocabulary_map->ref(cuco::op::find); }
  auto get_map2_ref() const { return vocabulary_map2->ref(cuco::op::find); }

  wordpiece_vocabulary_impl(std::unique_ptr<cudf::column>&& vocab,
                            col_device_view&& d_vocab,
                            std::unique_ptr<detail::vocabulary_map_type>&& map,
                            std::unique_ptr<detail::vocabulary_map_type2>&& map2,
                            cudf::size_type unk_id)
    : vocabulary(std::move(vocab)),
      d_vocabulary(std::move(d_vocab)),
      vocabulary_map(std::move(map)),
      vocabulary_map2(std::move(map2)),
      unk_id{unk_id}
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

template <typename MapRefType>
struct resolve_unk_id {
  MapRefType d_map;
  __device__ cudf::size_type operator()(int idx)
  {
    // look for both since the normalizer may change the case to match the vocab table
    auto const unk = idx == 0 ? cudf::string_view("[UNK]", 5) : cudf::string_view("[unk]", 5);
    auto const fnd = d_map.find(unk);
    return fnd != d_map.end() ? fnd->second : -1;
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
    cuco::thread_scope_thread,
    detail::cuco_storage{},
    cudf::detail::cuco_allocator<char>{rmm::mr::polymorphic_allocator<char>{}, stream},
    stream.value());
  // the row index is the token id (value for each key in the map)
  auto iter = cudf::detail::make_counting_transform_iterator(0, key_pair{});
  vocab_map->insert_async(iter, iter + vocabulary->size(), stream.value());
  auto const zero_itr = thrust::counting_iterator<cudf::size_type>(0);

  // setup 2nd map with just the ##prefixed items
  // get an index map of all the ##s
  auto map2_indices = rmm::device_uvector<cudf::size_type>(vocabulary->size(), stream);
  auto const end    = thrust::copy_if(rmm::exec_policy(stream),
                                   zero_itr,
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
    cuco::thread_scope_thread,
    detail::cuco_storage{},
    cudf::detail::cuco_allocator<char>{rmm::mr::polymorphic_allocator<char>{}, stream},
    stream.value());

  auto iter2 = thrust::make_transform_iterator(map2_indices.begin(), key_pair{});
  vocab_map2->insert_async(iter2, iter2 + map2_indices.size(), stream.value());

  // prefetch the [unk] vocab entry
  auto unk_ids = rmm::device_uvector<cudf::size_type>(2, stream);
  auto d_map   = vocab_map->ref(cuco::op::find);
  thrust::transform(rmm::exec_policy_nosync(stream),
                    zero_itr,
                    zero_itr + 1,
                    unk_ids.begin(),
                    resolve_unk_id<decltype(d_map)>{d_map});
  auto const id0    = unk_ids.front_element(stream);
  auto const id1    = unk_ids.back_element(stream);
  auto const unk_id = id0 >= 0 ? id0 : id1;

  _impl = new wordpiece_vocabulary_impl(std::move(vocabulary),
                                        std::move(d_vocabulary),
                                        std::move(vocab_map),
                                        std::move(vocab_map2),
                                        unk_id);
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

constexpr int block_size    = 64;
constexpr auto no_token     = cuda::std::numeric_limits<cudf::size_type>::max();
constexpr int max_word_size = 128;

__device__ cudf::string_view remove_last_char(cudf::string_view d_str)
{
  if (d_str.size_bytes() < 2) { return cudf::string_view(); }
  auto const begin = d_str.data();
  auto end         = begin + d_str.size_bytes() - 1;
  while ((end > begin) && cudf::strings::detail::is_utf8_continuation_char(*end)) {
    --end;
  }
  auto const size = static_cast<cudf::size_type>(thrust::distance(begin, end));
  return cudf::string_view(begin, size);
}

template <typename MapRefType, typename MapRefType2>
__device__ cudf::size_type wp_tokenize_fn(cudf::string_view word,
                                          MapRefType const& d_map,
                                          MapRefType2 const& d_map2,
                                          cudf::size_type unk_id,
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
  auto piece = remove_last_char(word);
  while (!piece.empty()) {
    itr = d_map.find(piece);
    if (itr == d_map.end()) {
      piece = remove_last_char(piece);
      continue;
    }
    d_tokens[token_idx++] = itr->second;
    break;
  }
  if (piece.empty()) {
    // did not find anything; this is uncommon
    d_tokens[token_idx++] = unk_id;
    return token_idx;
  }

  word =
    cudf::string_view(word.data() + piece.size_bytes(), word.size_bytes() - piece.size_bytes());
  piece = word;
  while (!piece.empty()) {
    auto itr = d_map2.find(piece);
    if (itr == d_map2.end()) {
      piece = remove_last_char(piece);
      continue;
    }
    d_tokens[token_idx++] = itr->second;

    word =
      cudf::string_view(word.data() + piece.size_bytes(), word.size_bytes() - piece.size_bytes());
    piece = word;
  }
  if (!word.empty()) {
    // very uncommon
    d_tokens[0] = unk_id;
    // need to reset any previous ids too
    for (auto i = 1; i < token_idx; ++i) {
      d_tokens[i] = no_token;
    }
    token_idx = 1;
  }

  return token_idx;
}

template <typename MapRefType, typename MapRefType2>
CUDF_KERNEL void tokenize_kernel2(cudf::device_span<int64_t const> d_edges,
                                  char const* d_chars,
                                  int64_t offset,
                                  MapRefType const d_map,
                                  MapRefType2 const d_map2,
                                  cudf::size_type unk_id,
                                  cudf::size_type* d_tokens)
{
  //
  auto const idx = cudf::detail::grid_1d::global_thread_id();
  if (idx >= (d_edges.size() - 1)) { return; }
  auto const begin    = d_chars + d_edges[idx];
  auto const end      = d_chars + d_edges[idx + 1];
  auto const word_end = thrust::find(thrust::seq, begin, end, ' ');
  auto const size     = static_cast<cudf::size_type>(thrust::distance(begin, word_end));
  if (size == 0) { return; }
  auto d_output = d_tokens + d_edges[idx] - offset;
  if (size >= max_word_size) { *d_output = unk_id; }
  auto const word = cudf::string_view{begin, size};
  wp_tokenize_fn(word, d_map, d_map2, unk_id, d_output);
}

template <typename OffsetType>
rmm::device_uvector<cudf::size_type> count_tokens(cudf::size_type const* d_tokens,
                                                  OffsetType offsets,
                                                  int64_t offset,
                                                  cudf::size_type size,
                                                  rmm::cuda_stream_view stream)
{
  auto d_counts = rmm::device_uvector<cudf::size_type>(size, stream);

  auto d_in = cudf::detail::make_counting_transform_iterator(
    0, cuda::proclaim_return_type<cudf::size_type>([d_tokens] __device__(auto idx) {
      return static_cast<cudf::size_type>(d_tokens[idx] != no_token);
    }));
  std::size_t temp = 0;
  auto d_out       = d_counts.data();
  nvtxRangePushA("segmented_reduce");
  if (offset == 0) {
    cub::DeviceSegmentedReduce::Sum(
      nullptr, temp, d_in, d_out, size, offsets, offsets + 1, stream.value());
    auto d_temp = rmm::device_buffer{temp, stream};
    cub::DeviceSegmentedReduce::Sum(
      d_temp.data(), temp, d_in, d_out, size, offsets, offsets + 1, stream.value());
  } else {
    // offsets need to be normalized for segmented-reduce to work efficiently
    auto d_offsets = rmm::device_uvector<cudf::size_type>(size + 1, stream);
    thrust::transform(rmm::exec_policy_nosync(stream),
                      offsets,
                      offsets + size + 1,
                      d_offsets.begin(),
                      [offset] __device__(auto o) { return o - offset; });
    auto const offsets_itr = d_offsets.begin();
    cub::DeviceSegmentedReduce::Sum(
      nullptr, temp, d_in, d_out, size, offsets_itr, offsets_itr + 1, stream.value());
    auto d_temp = rmm::device_buffer{temp, stream};
    cub::DeviceSegmentedReduce::Sum(
      d_temp.data(), temp, d_in, d_out, size, offsets_itr, offsets_itr + 1, stream.value());
  }
  stream.synchronize();
  nvtxRangePop();

  return d_counts;
}
}  // namespace

std::unique_ptr<cudf::column> wordpiece_tokenize(cudf::strings_column_view const& input,
                                                 wordpiece_vocabulary const& vocabulary,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  auto const output_type = cudf::data_type{cudf::type_to_id<cudf::size_type>()};
  if (input.size() == input.null_count()) { return cudf::make_empty_column(output_type); }

  auto const first_offset  = (input.offset() == 0) ? 0
                                                   : cudf::strings::detail::get_offset_value(
                                                      input.offsets(), input.offset(), stream);
  auto const last_offset   = (input.offset() == 0 && input.size() == input.offsets().size() - 1)
                               ? input.chars_size(stream)
                               : cudf::strings::detail::get_offset_value(
                                 input.offsets(), input.size() + input.offset(), stream);
  auto const chars_size    = last_offset - first_offset;
  auto const d_input_chars = input.chars_begin(stream);  // without applying first_offset

  auto input_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset());

  // find the beginning word edges
  auto d_all_edges = [&] {
    rmm::device_uvector<int64_t> d_edges(chars_size / 2, stream);
    nvtxRangePushA("copy_if_safe");
    auto edges_end = cudf::detail::copy_if_safe(
      thrust::counting_iterator<int64_t>(first_offset),
      thrust::counting_iterator<int64_t>(last_offset),
      d_edges.begin(),
      [d_input_chars, first_offset] __device__(auto idx) {
        if (idx == first_offset) { return d_input_chars[idx] == ' '; }
        return (d_input_chars[idx] != ' ' && d_input_chars[idx - 1] == ' ');
      },
      stream);
    stream.synchronize();
    nvtxRangePop();

    auto edges =
      input.size() + 1 + static_cast<int64_t>(thrust::distance(d_edges.begin(), edges_end));
    // thrust::merge may have an int32 max limit
    CUDF_EXPECTS(edges < std::numeric_limits<int32_t>::max(), "words exceed internal limit");

    auto d_all_edges = rmm::device_uvector<int64_t>(edges, stream);
    nvtxRangePushA("merge_edges");
    thrust::merge(rmm::exec_policy_nosync(stream),
                  input_offsets,
                  input_offsets + input.size() + 1,
                  d_edges.begin(),
                  edges_end,
                  d_all_edges.begin());
    stream.synchronize();
    nvtxRangePop();
    return d_all_edges;
  }();

  auto const map_ref  = vocabulary._impl->get_map_ref();
  auto const map2_ref = vocabulary._impl->get_map2_ref();
  auto const unk_id   = vocabulary._impl->unk_id;

  rmm::device_uvector<cudf::size_type> d_tokens(chars_size, stream);
  thrust::uninitialized_fill(
    rmm::exec_policy_nosync(stream), d_tokens.begin(), d_tokens.end(), no_token);

  nvtxRangePushA("tokenize_kernel2");
  cudf::detail::grid_1d grid{static_cast<cudf::size_type>(d_all_edges.size()), 512};
  tokenize_kernel2<decltype(map_ref), decltype(map2_ref)>
    <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
      d_all_edges, d_input_chars, first_offset, map_ref, map2_ref, unk_id, d_tokens.data());
  stream.synchronize();
  nvtxRangePop();

  // compute token counts using segmented-reduce to count !no_token values in d_tokens
  auto const d_token_counts =
    count_tokens(d_tokens.data(), input_offsets, first_offset, input.size(), stream);
  // convert counts to offsets
  auto [token_offsets, total_count] = cudf::detail::make_offsets_child_column(
    d_token_counts.begin(), d_token_counts.end(), stream, mr);

  auto tokens =
    cudf::make_numeric_column(output_type, total_count, cudf::mask_state::UNALLOCATED, stream, mr);
  auto output = tokens->mutable_view().begin<cudf::size_type>();
  nvtxRangePushA("remove_no_tokens");
  thrust::remove_copy(rmm::exec_policy(stream), d_tokens.begin(), d_tokens.end(), output, no_token);
  stream.synchronize();
  nvtxRangePop();

  return cudf::make_lists_column(input.size(),
                                 std::move(token_offsets),
                                 std::move(tokens),
                                 input.null_count(),
                                 cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                 stream,
                                 mr);
}

template <int tile_size = cudf::detail::warp_size>
CUDF_KERNEL void find_words_kernel4(cudf::column_device_view const d_strings,
                                    char const* d_input_chars,
                                    int64_t const* offsets,
                                    int64_t* starts,
                                    cudf::size_type* sizes)
{
  // string per block
  auto const idx     = cudf::detail::grid_1d::global_thread_id();
  auto const str_idx = idx / tile_size;
  if (str_idx >= d_strings.size()) { return; }
  if (d_strings.is_null(str_idx)) { return; }
  auto const d_str = d_strings.element<cudf::string_view>(str_idx);
  if (d_str.empty()) { return; }
  auto const str_offset = static_cast<int64_t>(thrust::distance(d_input_chars, d_str.data()));

  auto const d_start_words   = starts + offsets[str_idx];
  auto const d_word_sizes    = sizes + offsets[str_idx];
  cudf::size_type max_tokens = offsets[str_idx + 1] - offsets[str_idx];

  constexpr int bytes_per_thread = 6;  // avg 5 chars per word plus space
  constexpr int words_size       = block_size * bytes_per_thread;
  __shared__ cudf::size_type s_start_words[words_size];
  __shared__ cudf::size_type s_end_words[words_size];

  namespace cg     = cooperative_groups;
  auto const block = cg::this_thread_block();
  auto const tile  = cg::tiled_partition<tile_size>(block);

  auto const lane_idx    = tile.thread_rank();
  auto const warp_idx    = tile.meta_group_rank();
  auto const warp_words  = words_size / tile.meta_group_size();
  constexpr auto no_word = cuda::std::numeric_limits<cudf::size_type>::max();

  cudf::size_type word_count = 0;
  cudf::size_type byte_count = 0;

  auto first_word  = no_word;  // only used by lane_idx==0
  auto const begin = d_str.data();
  auto const end   = begin + d_str.size_bytes();

  auto start_words = s_start_words + (warp_idx * warp_words);
  auto end_words   = s_end_words + (warp_idx * warp_words);

  // continue until all bytes have been consumed or the max token count has been reached
  auto itr = begin + lane_idx;
  while (word_count < max_tokens && byte_count < d_str.size_bytes()) {
    // initialize all intermediate results
    start_words[lane_idx] = lane_idx > 0 ? no_word : first_word;
    end_words[lane_idx]   = no_word;
    for (auto j = lane_idx + tile_size; j < warp_words; j += tile_size) {
      start_words[j] = no_word;
      end_words[j]   = no_word;
    }
    // init 2 lanes/thread above might eliminate this
    tile.sync();

    int last_idx = 0;
    // each thread processes bytes_per_thread of the d_str
    for (auto k = lane_idx; k < warp_words && itr < end; k += tile_size) {
      if ((*itr != ' ') && ((itr == begin) || (*(itr - 1) == ' '))) {
        last_idx              = (k / 2) + 1;
        start_words[last_idx] = static_cast<cudf::size_type>(thrust::distance(begin, itr));
      }
      if (((itr + 1) == end) || ((itr != begin) && (*itr == ' ') && (*(itr - 1) != ' '))) {
        cudf::size_type const adjust = (*itr != ' ');
        last_idx                     = (k / 2) + adjust;
        end_words[last_idx] = static_cast<cudf::size_type>(thrust::distance(begin, itr)) + adjust;
      }
      itr += tile_size;
    }
    last_idx = cg::reduce(tile, last_idx, cg::greater<int>{}) + 1;  // implicit sync?

    int output_count = 0;
    if (lane_idx == 0) {
      auto const count       = static_cast<int>(thrust::distance(
        start_words, thrust::remove(thrust::seq, start_words, start_words + last_idx, no_word)));
      auto const words_found = static_cast<int>(thrust::distance(
        end_words, thrust::remove(thrust::seq, end_words, end_words + last_idx, no_word)));
      // this partial word wraps around for the next iteration
      first_word   = (count > words_found) ? start_words[words_found] : no_word;
      output_count = min(words_found, max_tokens - word_count);
    }
    auto out_starts = d_start_words + word_count;
    auto out_sizes  = d_word_sizes + word_count;
    output_count    = tile.shfl(output_count, 0);  // implicit tile.sync()?

    // copy results to the output
    for (auto k = lane_idx; k < output_count; k += tile_size) {
      auto const start = start_words[k];
      out_starts[k]    = start + str_offset;
      out_sizes[k]     = end_words[k] - start;
    }

    word_count += output_count;
    byte_count += tile_size * bytes_per_thread;
    tile.sync();
  }

  // fill in remainder of the output
  auto out_starts = d_start_words + word_count;
  auto out_sizes  = d_word_sizes + word_count;
  for (auto k = lane_idx; k < (max_tokens - word_count); k += tile_size) {
    out_starts[k] = cuda::std::numeric_limits<int64_t>::max();
    out_sizes[k]  = no_word;
  }
}

namespace {
template <typename MapRefType, typename MapRefType2>
CUDF_KERNEL void tokenize_kernel4(cudf::device_span<int64_t const> d_starts,
                                  cudf::device_span<int const> d_sizes,
                                  char const* d_chars,
                                  MapRefType const d_map,
                                  MapRefType2 const d_map2,
                                  cudf::size_type unk_id,
                                  cudf::size_type* d_tokens)
{
  auto const idx = cudf::detail::grid_1d::global_thread_id();
  if (idx >= d_starts.size()) { return; }
  auto const size = d_sizes[idx];
  if (size == 0) { return; }  // not expected
  auto const start = d_starts[idx];
  auto const begin = d_chars + start;
  auto d_output    = d_tokens + start;
  if (size >= max_word_size) { *d_output = unk_id; }
  auto const word = cudf::string_view{begin, size};
  wp_tokenize_fn(word, d_map, d_map2, unk_id, d_output);
}
}  // namespace

std::unique_ptr<cudf::column> wordpiece_tokenize(cudf::strings_column_view const& input,
                                                 wordpiece_vocabulary const& vocabulary,
                                                 cudf::size_type max_words_per_row,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  auto const output_type = cudf::data_type{cudf::type_to_id<cudf::size_type>()};
  if (input.size() == input.null_count()) { return cudf::make_empty_column(output_type); }
  CUDF_EXPECTS(max_words_per_row > 0, "maximum words must be greater than 0");

  auto const first_offset  = (input.offset() == 0) ? 0
                                                   : cudf::strings::detail::get_offset_value(
                                                      input.offsets(), input.offset(), stream);
  auto const last_offset   = (input.offset() == 0 && input.size() == input.offsets().size() - 1)
                               ? input.chars_size(stream)
                               : cudf::strings::detail::get_offset_value(
                                 input.offsets(), input.size() + input.offset(), stream);
  auto const chars_size    = last_offset - first_offset;
  auto const d_input_chars = input.chars_begin(stream) + first_offset;

  auto const d_strings  = cudf::column_device_view::create(input.parent(), stream);
  auto max_word_offsets = rmm::device_uvector<int64_t>(input.size() + 1, stream);
  nvtxRangePushA("compute_offsets");
  thrust::transform(rmm::exec_policy_nosync(stream),
                    thrust::counting_iterator<cudf::size_type>(0),
                    thrust::counting_iterator<cudf::size_type>(input.size()),
                    max_word_offsets.begin(),
                    cuda::proclaim_return_type<cudf::size_type>(
                      [d_strings = *d_strings, max_words_per_row] __device__(auto idx) {
                        if (idx >= d_strings.size()) { return 0; }
                        if (d_strings.is_null(idx)) { return 0; }
                        auto const d_str = d_strings.element<cudf::string_view>(idx);
                        return min(max_words_per_row, d_str.size_bytes() / 2);
                      }));

  auto const max_size = cudf::detail::sizes_to_offsets(
    max_word_offsets.begin(), max_word_offsets.end(), max_word_offsets.begin(), stream);
  nvtxRangePop();

  auto start_words = rmm::device_uvector<int64_t>(max_size, stream);
  auto word_sizes  = rmm::device_uvector<cudf::size_type>(max_size, stream);

  // find start/end for upto max_words_per_row words
  // compute diff and store word positions in start_words and sizes in word_sizes
  nvtxRangePushA("find_word_boundaries");
  cudf::detail::grid_1d grid{input.size() * cudf::detail::warp_size, block_size};
  find_words_kernel4<cudf::detail::warp_size>
    <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
      *d_strings, d_input_chars, max_word_offsets.data(), start_words.data(), word_sizes.data());
  stream.synchronize();
  nvtxRangePop();

  nvtxRangePushA("remove_non_words");
  auto const end   = thrust::remove(rmm::exec_policy(stream),
                                  start_words.begin(),
                                  start_words.end(),
                                  std::numeric_limits<int64_t>::max());
  auto const check = thrust::remove(rmm::exec_policy(stream),
                                    word_sizes.begin(),
                                    word_sizes.end(),
                                    std::numeric_limits<int32_t>::max());
  stream.synchronize();
  nvtxRangePop();

  auto const total_words = static_cast<int64_t>(thrust::distance(start_words.begin(), end));
  CUDF_EXPECTS(total_words == static_cast<int64_t>(thrust::distance(word_sizes.begin(), check)),
               "error resolving word locations from input column");
  start_words.resize(total_words, stream);
  word_sizes.resize(total_words, stream);

  auto const map_ref  = vocabulary._impl->get_map_ref();
  auto const map2_ref = vocabulary._impl->get_map2_ref();
  auto const unk_id   = vocabulary._impl->unk_id;

  rmm::device_uvector<cudf::size_type> d_tokens(chars_size, stream);
  thrust::uninitialized_fill(
    rmm::exec_policy_nosync(stream), d_tokens.begin(), d_tokens.end(), no_token);

  nvtxRangePushA("tokenize_kernel4");
  cudf::detail::grid_1d grid2{total_words, 512};
  tokenize_kernel4<decltype(map_ref), decltype(map2_ref)>
    <<<grid2.num_blocks, grid2.num_threads_per_block, 0, stream.value()>>>(
      start_words, word_sizes, d_input_chars, map_ref, map2_ref, unk_id, d_tokens.data());
  stream.synchronize();
  nvtxRangePop();

  auto const input_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset());

  // compute token counts by doing a segmented reduce with a transform-iterator
  // that can be used to count !no_token values in d_tokens
  auto const d_token_counts =
    count_tokens(d_tokens.data(), input_offsets, first_offset, input.size(), stream);

  auto [token_offsets, total_count] = cudf::detail::make_offsets_child_column(
    d_token_counts.begin(), d_token_counts.end(), stream, mr);

  auto tokens =
    cudf::make_numeric_column(output_type, total_count, cudf::mask_state::UNALLOCATED, stream, mr);
  auto output = tokens->mutable_view().begin<cudf::size_type>();
  nvtxRangePushA("remove_no_tokens");
  thrust::remove_copy(
    rmm::exec_policy_nosync(stream), d_tokens.begin(), d_tokens.end(), output, no_token);
  stream.synchronize();
  nvtxRangePop();

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
                                                 cudf::size_type max_words_per_row,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  /*if (max_words_per_row == 0)*/ {
    return detail::wordpiece_tokenize(input, vocabulary, stream, mr);
  }
  // return detail::wordpiece_tokenize(input, vocabulary, max_words_per_row, stream, mr);
}

}  // namespace nvtext
