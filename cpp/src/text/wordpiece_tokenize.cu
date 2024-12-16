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

/**
 * @brief Threshold to decide on using string or warp parallel functions.
 *
 * If the average byte length of a string in a column exceeds this value then
 * the warp-parallel function is used to compute the output sizes.
 * Otherwise, a regular string-parallel function is used.
 *
 * This value was found using the vocab_tokenize benchmark results.
 */
constexpr cudf::size_type AVG_CHAR_BYTES_THRESHOLD = 128;

constexpr int block_size = 256;

__device__ bool is_delimiter(cudf::char_utf8 chr) { return (chr <= ' '); }

struct mark_delimiters_fn {
  char const* d_chars;
  int8_t* d_results;

  __device__ void operator()(cudf::size_type idx) const
  {
    auto const ptr = d_chars + idx;
    if (cudf::strings::detail::is_utf8_continuation_char(*ptr)) { return; }
    cudf::char_utf8 chr = 0;
    auto ch_size        = cudf::strings::detail::to_char_utf8(ptr, chr);
    auto const output   = is_delimiter(chr);
    while (ch_size > 0) {
      d_results[idx++] = output;
      --ch_size;
    }
  }
};

CUDF_KERNEL void token_counts_fn(cudf::column_device_view const d_strings,
                                 cudf::size_type* d_counts,
                                 int8_t* d_results)
{
  // string per warp
  auto const idx     = cudf::detail::grid_1d::global_thread_id();
  auto const str_idx = static_cast<cudf::size_type>(idx / cudf::detail::warp_size);
  if (str_idx >= d_strings.size()) { return; }
  auto const lane_idx = static_cast<cudf::size_type>(idx % cudf::detail::warp_size);

  if (d_strings.is_null(str_idx)) {
    d_counts[str_idx] = 0;
    return;
  }
  auto const d_str = d_strings.element<cudf::string_view>(str_idx);
  if (d_str.empty()) {
    d_counts[str_idx] = 0;
    return;
  }

  auto const offsets     = d_strings.child(cudf::strings_column_view::offsets_column_index);
  auto const offsets_itr = cudf::detail::input_offsetalator(offsets.head(), offsets.type());
  auto const offset = offsets_itr[str_idx + d_strings.offset()] - offsets_itr[d_strings.offset()];
  auto const chars_begin = d_strings.data<char>() + offsets_itr[d_strings.offset()];

  auto const begin        = d_str.data();
  auto const end          = begin + d_str.size_bytes();
  auto const d_output     = d_results + offset;
  auto const d_output_end = d_output + d_str.size_bytes();

  using warp_reduce = cub::WarpReduce<cudf::size_type>;
  __shared__ typename warp_reduce::TempStorage warp_storage;

  cudf::size_type count = 0;
  if (lane_idx == 0) {
    cudf::char_utf8 chr = 0;
    auto ch_size        = cudf::strings::detail::to_char_utf8(begin, chr);
    auto output         = 1;
    if (begin > chars_begin) {
      auto ptr = begin - 1;
      while (ptr > chars_begin && cudf::strings::detail::is_utf8_continuation_char(*ptr)) {
        --ptr;
      }
      cudf::strings::detail::to_char_utf8(ptr, chr);
      output = !is_delimiter(chr);
    }
    auto ptr = d_output;
    while (ch_size > 0) {
      *ptr++ = output;
      --ch_size;
    }
    count = ((begin + ch_size) == end);
  }
  __syncwarp();

  for (auto itr = d_output + lane_idx + 1; itr < d_output_end; itr += cudf::detail::warp_size) {
    // add one if at the edge of a token or if at the string's end
    if (*itr) {
      count += !(*(itr - 1));
    } else {
      count += (itr + 1 == d_output_end);
    }
  }
  __syncwarp();

  // add up the counts from the other threads to compute the total token count for this string
  auto const total_count = warp_reduce(warp_storage).Reduce(count, cub::Sum());
  if (lane_idx == 0) { d_counts[str_idx] = total_count; }
}

template <typename MapRefType, typename MapRefType2>
struct vocabulary_tokenizer_fn {
  cudf::column_device_view const d_strings;
  MapRefType d_map;
  MapRefType2 d_map2;
  cudf::detail::input_offsetalator d_offsets;
  cudf::size_type* d_results;

  __device__ void operator()(cudf::size_type idx) const
  {
    if (d_strings.is_null(idx)) { return; }

    auto const d_str = d_strings.element<cudf::string_view>(idx);
    characters_tokenizer tokenizer(d_str, cudf::string_view{});
    auto d_tokens = d_results + d_offsets[idx];

    cudf::size_type token_idx = 0;
    while (tokenizer.next_token()) {
      auto const pos   = tokenizer.token_byte_positions();
      auto const token = cudf::string_view{d_str.data() + pos.first, (pos.second - pos.first)};
      // lookup token in map
      auto const itr = d_map.find(token);
      if (itr != d_map.end()) {
        d_tokens[token_idx++] = itr->second;
        continue;
      }
      // reduce string by one character and try again
      auto piece = token.substr(0, token.length() - 1);
      while (piece.length() > 0) {
        auto const itr = d_map.find(piece);
        if (itr == d_map.end()) {
          piece = piece.substr(0, piece.length() - 1);
          continue;
        }
        d_tokens[token_idx++] = itr->second;
      }
      piece = token.substr(piece.length(), token.length() - piece.length());
      while (piece.length() > 0) {
        auto const itr = d_map2.find(piece);
        if (itr == d_map2.end()) {
          piece = piece.substr(0, piece.length() - 1);
          continue;
        }
        d_tokens[token_idx++] = itr->second;
      }
    }
  }
};

template <typename MapRefType>
struct transform_tokenizer_fn {
  MapRefType d_map;
  cudf::size_type const default_id;

  __device__ cudf::size_type operator()(cudf::string_view d_str) const
  {
    auto const begin = d_str.data();
    auto const end   = begin + d_str.size_bytes();

    auto itr = begin;
    while (itr < end) {
      cudf::char_utf8 chr = 0;
      auto const ch_size  = cudf::strings::detail::to_char_utf8(itr, chr);
      if (!is_delimiter(chr)) break;
      itr += ch_size;
    }

    auto const size  = static_cast<cudf::size_type>(thrust::distance(itr, end));
    auto const token = cudf::string_view{itr, size};
    // lookup token in map
    auto const fitr = d_map.find(token);
    return (fitr != d_map.end()) ? fitr->second : default_id;
  }
};

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
  auto const zero_itr  = thrust::make_counting_iterator<cudf::size_type>(0);

  if ((input.chars_size(stream) / (input.size() - input.null_count())) < AVG_CHAR_BYTES_THRESHOLD) {
    auto const sizes_itr =
      cudf::detail::make_counting_transform_iterator(0, strings_tokenizer{*d_strings});
    auto [token_offsets, total_count] =
      cudf::detail::make_offsets_child_column(sizes_itr, sizes_itr + input.size(), stream, mr);

    // build the output column to hold all the token ids
    auto tokens = cudf::make_numeric_column(
      output_type, total_count, cudf::mask_state::UNALLOCATED, stream, mr);
    auto d_tokens  = tokens->mutable_view().data<cudf::size_type>();
    auto d_offsets = cudf::detail::offsetalator_factory::make_input_iterator(token_offsets->view());
    vocabulary_tokenizer_fn<decltype(map_ref), decltype(map2_ref)> tokenizer{
      *d_strings, map_ref, map2_ref, d_offsets, d_tokens};
    thrust::for_each_n(rmm::exec_policy(stream), zero_itr, input.size(), tokenizer);
    return cudf::make_lists_column(input.size(),
                                   std::move(token_offsets),
                                   std::move(tokens),
                                   input.null_count(),
                                   cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                   stream,
                                   mr);
  }

  // longer strings perform better with warp-parallel approach

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
  rmm::device_uvector<int8_t> d_marks(chars_size, stream);

  // mark position of all delimiters
  thrust::for_each_n(rmm::exec_policy(stream),
                     zero_itr,
                     chars_size,
                     mark_delimiters_fn{d_input_chars, d_marks.data()});

  // launch warp per string to compute token counts
  cudf::detail::grid_1d grid{input.size() * cudf::detail::warp_size, block_size};
  token_counts_fn<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
    *d_strings, d_token_counts.data(), d_marks.data());
  auto [token_offsets, total_count] = cudf::detail::make_offsets_child_column(
    d_token_counts.begin(), d_token_counts.end(), stream, mr);

  rmm::device_uvector<cudf::size_type> d_tmp_offsets(total_count + 1, stream);
  d_tmp_offsets.set_element(total_count, chars_size, stream);
  thrust::copy_if(rmm::exec_policy(stream),
                  zero_itr,
                  thrust::counting_iterator<cudf::size_type>(chars_size),
                  d_tmp_offsets.begin(),
                  [d_marks = d_marks.data()] __device__(auto idx) {
                    if (idx == 0) return true;
                    return d_marks[idx] && !d_marks[idx - 1];
                  });

  auto tmp_offsets =
    std::make_unique<cudf::column>(std::move(d_tmp_offsets), rmm::device_buffer{}, 0);
  auto const tmp_input = cudf::column_view(
    input.parent().type(), total_count, d_input_chars, nullptr, 0, 0, {tmp_offsets->view()});

  auto const d_tmp_strings = cudf::column_device_view::create(tmp_input, stream);

  auto tokens =
    cudf::make_numeric_column(output_type, total_count, cudf::mask_state::UNALLOCATED, stream, mr);
  auto d_tokens = tokens->mutable_view().data<cudf::size_type>();

  transform_tokenizer_fn<decltype(map_ref)> tokenizer{map_ref};
  thrust::transform(rmm::exec_policy(stream),
                    d_tmp_strings->begin<cudf::string_view>(),
                    d_tmp_strings->end<cudf::string_view>(),
                    d_tokens,
                    tokenizer);

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
