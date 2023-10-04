/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.
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

#include <text/subword/bpe_tokenizer.cuh>

#include <nvtext/bpe_tokenize.hpp>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/merge.h>
#include <thrust/remove.h>
#include <thrust/transform.h>

namespace nvtext {
namespace detail {
namespace {

constexpr int block_size = 512;

template <typename MapRefType>
__global__ void bpe_up_offsets_fn(char const* d_chars,
                                  cudf::size_type chars_size,
                                  cudf::size_type offset,
                                  MapRefType const d_map,
                                  cudf::size_type* d_offsets)
{
  auto const idx = static_cast<cudf::size_type>(cudf::detail::grid_1d::global_thread_id());
  if (idx >= chars_size) { return; }
  if (!cudf::strings::detail::is_begin_utf8_char(d_chars[idx])) {
    d_offsets[idx] = 0;
    return;
  }

  auto next_substr = [d_chars, end = d_chars + chars_size](char const* begin) {
    auto const next = thrust::find_if(thrust::seq, begin + 1, end, [](auto v) {
      return cudf::strings::detail::is_begin_utf8_char(v);
    });
    auto const size = static_cast<cudf::size_type>(thrust::distance(begin, next));
    return cudf::string_view(begin, size);
  };

  auto const itr      = d_chars + idx;
  auto const end      = d_chars + chars_size;
  auto const lhs      = next_substr(itr);
  auto const next_itr = itr + lhs.size_bytes();
  auto output         = 0;
  if (next_itr < end) {
    auto const rhs = next_substr(next_itr);
    if (!rhs.empty()) {
      // see if both halves exist anywhere in the table
      if (d_map.find(lhs) == d_map.end() && d_map.find(rhs) == d_map.end()) {
        output = idx + lhs.size_bytes() + offset;  // candidate for artificial boundary
      }
    }
  }
  d_offsets[idx] = output;
}

template <typename MapRefType>
__global__ void bpe_parallel_fn(cudf::column_device_view const d_strings,
                                MapRefType const d_map,
                                int8_t* d_spaces_in,       // working memory
                                cudf::size_type* d_ranks,  // more working memory
                                int8_t* d_rerank_in        // and one more working memory
)
{
  // string per block
  auto const str_idx =
    static_cast<cudf::size_type>(cudf::detail::grid_1d::global_thread_id() / block_size);
  auto const lane_idx = static_cast<cudf::size_type>(threadIdx.x);

  if (d_strings.is_null(str_idx)) { return; }
  auto const d_str = d_strings.element<cudf::string_view>(str_idx);
  if (d_str.empty()) { return; }

  auto const offsets =
    d_strings.child(cudf::strings_column_view::offsets_column_index).data<cudf::size_type>();
  auto const offset = offsets[str_idx + d_strings.offset()] - offsets[d_strings.offset()];

  auto const d_spaces    = d_spaces_in + offset;
  auto const end_spaces  = d_spaces + d_str.size_bytes();
  auto const d_min_ranks = d_ranks + offset;
  auto const end_ranks   = d_min_ranks + d_str.size_bytes();
  auto const d_rerank    = d_rerank_in + offset;
  auto const end_rerank  = d_rerank + d_str.size_bytes();

  auto constexpr max_rank = cuda::std::numeric_limits<cudf::size_type>::max();

  __shared__ cudf::size_type block_min_rank;
  using block_reduce = cub::BlockReduce<cudf::size_type, block_size>;
  __shared__ typename block_reduce::TempStorage temp_storage;
  auto const num_valid = block_size < d_str.size_bytes() ? block_size : d_str.size_bytes();

  auto next_substr = [d_str, d_spaces, end = end_spaces](int8_t* begin) {
    auto const next = thrust::find_if(thrust::seq, begin + 1, end, [](auto v) { return v != 0; });
    auto const size = static_cast<cudf::size_type>(thrust::distance(begin, next));
    return cudf::string_view(d_str.data() + thrust::distance(d_spaces, begin), size);
  };

  // init all ranks to max
  for (auto itr = d_min_ranks + lane_idx; itr < end_ranks; itr += block_size) {
    *itr = max_rank;
  }
  // init all spaces to 1 as appropriate
  for (auto itr = d_spaces + lane_idx; itr < end_spaces; itr += block_size) {
    auto const index = thrust::distance(d_spaces, itr);
    *itr = static_cast<int8_t>(cudf::strings::detail::is_begin_utf8_char(d_str.data()[index]));
  }
  __syncthreads();

  auto min_rank = max_rank;

  // store all the initial ranks for each pair
  for (auto itr = d_spaces + lane_idx; itr < end_spaces; itr += block_size) {
    if (*itr == 0) { continue; }  // start on valid bytes only
    // resolve pair and lookup its rank
    auto const lhs      = next_substr(itr);  // retrieve lhs of the pair
    auto const next_itr = itr + lhs.size_bytes();
    if (next_itr < end_spaces) {
      auto const rhs = next_substr(next_itr);  // retrieve rhs of the pair
      if (!rhs.empty()) {
        auto rank          = max_rank;
        auto const mp      = merge_pair_type{lhs, rhs};
        auto const map_itr = d_map.find(mp);                       // lookup pair in merges table;
        if (map_itr != d_map.end()) { rank = map_itr->second; }    // found a match;
        d_min_ranks[thrust::distance(d_spaces, next_itr)] = rank;  // store the rank
        if (rank < min_rank) min_rank = rank;
      }
    }
  }
  // compute the min rank across the block
  auto const reduce_rank = block_reduce(temp_storage).Reduce(min_rank, cub::Min(), num_valid);
  if (lane_idx == 0) { block_min_rank = reduce_rank; }
  __syncthreads();

  // loop through the ranks finding the current minimum until there are no more
  while (block_min_rank < max_rank) {
    // (re)initialize all the re-rank identifiers to zero
    for (auto itr = d_rerank + lane_idx; itr < end_rerank; itr += block_size) {
      *itr = 0;
    }

    // search the d_min_ranks for all the places where the rank matches block_min_rank
    for (auto itr = d_min_ranks + lane_idx; itr < end_ranks; itr += block_size) {
      if (*itr == block_min_rank) {
        auto ptr   = itr - 1;  // check for adjacent min-rank edge-case
        auto count = 8;
        while (ptr > d_min_ranks && *ptr == max_rank && count > 0) {
          --ptr;
          --count;
        }
        // set the output value to 0 at this position (erases separator)
        if (*ptr != block_min_rank) { d_spaces[thrust::distance(d_min_ranks, itr)] = 0; }
      }
    }
    __syncthreads();

    auto find_prev = [begin = d_spaces](int8_t* ptr) {
      while (ptr > begin && *ptr == 0) {
        --ptr;
      }
      return ptr;
    };
    auto find_next = [end = end_spaces](int8_t* ptr) {
      while (ptr < end && *ptr == 0) {
        ++ptr;
      }
      return ptr;
    };

    // identify the re-rank locations
    for (auto itr = d_min_ranks + lane_idx; itr < end_ranks; itr += block_size) {
      auto const index = thrust::distance(d_min_ranks, itr);
      if (*itr == block_min_rank && d_spaces[index] == 0) {
        auto ptr = find_prev(d_spaces + index - 1);  // find previous pair mid-point
        if (ptr > d_spaces) { d_rerank[thrust::distance(d_spaces, ptr)] = 1; }
        ptr = find_next(d_spaces + index + 1);  // find next pair mid-point
        if (ptr < end_spaces) { d_rerank[thrust::distance(d_spaces, ptr)] = 1; }
        *itr = max_rank;  // reset this rank
      }
    }
    __syncthreads();

    // compute the ranks for the newly created pairs
    min_rank = max_rank;  // and record new minimum
    for (auto itr = d_rerank + lane_idx; itr < end_rerank; itr += block_size) {
      auto const index = thrust::distance(d_rerank, itr);
      auto rank        = d_min_ranks[index];
      if (*itr) {
        // build lhs of pair
        auto const ptr  = find_prev(d_spaces + index - 1);
        auto const size = static_cast<cudf::size_type>(thrust::distance(ptr, d_spaces + index));
        auto const lhs  = cudf::string_view(d_str.data() + thrust::distance(d_spaces, ptr), size);
        // retrieve rhs of pair
        auto const rhs = next_substr(d_spaces + index);
        rank           = max_rank;
        if (!rhs.empty()) {
          auto const mp      = merge_pair_type{lhs, rhs};
          auto const map_itr = d_map.find(mp);                     // lookup in merges;
          if (map_itr != d_map.end()) { rank = map_itr->second; }  // found a match
        }
        d_min_ranks[index] = rank;
      }
      if (rank < min_rank) min_rank = rank;
    }

    // compute the min rank across the block
    auto const reduce_rank = block_reduce(temp_storage).Reduce(min_rank, cub::Min(), num_valid);
    if (lane_idx == 0) { block_min_rank = reduce_rank; }
    __syncthreads();
  }  // if no mins were found we are done, otherwise start again
}

__global__ void bpe_finalize(cudf::column_device_view const d_strings,
                             int8_t* d_spaces_in,      // where separators are inserted
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

  auto const offsets =
    d_strings.child(cudf::strings_column_view::offsets_column_index).data<cudf::size_type>();
  auto const offset = offsets[str_idx + d_strings.offset()] - offsets[d_strings.offset()];

  auto const d_spaces   = d_spaces_in + offset;
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
  auto const size = block_reduce(temp_storage).Sum(bytes, num_valid);
  if (lane_idx == 0) { d_sizes[str_idx] = size + d_str.size_bytes(); }
}

}  // namespace

std::unique_ptr<cudf::column> byte_pair_encoding(cudf::strings_column_view const& input,
                                                 bpe_merge_pairs const& merge_pairs,
                                                 cudf::string_scalar const& separator,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::mr::device_memory_resource* mr)
{
  if (input.is_empty() || input.chars_size() == 0) {
    return cudf::make_empty_column(cudf::type_id::STRING);
  }

  CUDF_EXPECTS(separator.is_valid(stream), "separator parameter must be valid");
  auto const d_separator = separator.value(stream);
  CUDF_EXPECTS(d_separator.size_bytes() == 1, "for now, separator must be a single-byte character");

  auto const d_strings = cudf::column_device_view::create(input.parent(), stream);

  auto const first_offset  = (input.offset() == 0) ? 0
                                                   : cudf::detail::get_value<cudf::size_type>(
                                                      input.offsets(), input.offset(), stream);
  auto const last_offset   = (input.offset() == 0 && input.size() == input.offsets().size() - 1)
                               ? input.chars().size()
                               : cudf::detail::get_value<cudf::size_type>(
                                 input.offsets(), input.size() + input.offset(), stream);
  auto const chars_size    = last_offset - first_offset;
  auto const d_input_chars = input.chars().data<char>() + first_offset;

  auto const offset_data_type = cudf::data_type{cudf::type_to_id<cudf::size_type>()};
  auto offsets                = cudf::make_numeric_column(
    offset_data_type, input.size() + 1, cudf::mask_state::UNALLOCATED, stream, mr);
  auto d_offsets = offsets->mutable_view().data<cudf::size_type>();

  rmm::device_uvector<int8_t> d_spaces(chars_size, stream);
  rmm::device_uvector<cudf::size_type> d_ranks(chars_size, stream);  // rank per string pair;
  rmm::device_uvector<int8_t> d_rerank(chars_size, stream);          // re-ranking identifiers
  auto const map_ref  = merge_pairs.impl->get_merge_pairs_ref();
  auto const map_ref2 = merge_pairs.impl->get_mp_table_ref();

  if ((input.offset() == 0) && (input.size() == input.offsets().size() - 1)) {
    // TODO: this fails for sliced columns for some reason;
    //       we could get ride of the else{} if this was fixed

    // this path locates unpairable sections of code to create artificial string row boundaries;
    // the boundary values are recorded as offsets and stored temporarily in the d_ranks vector
    auto const block_count = (chars_size + block_size - 1) / block_size;
    bpe_up_offsets_fn<decltype(map_ref2)><<<block_count, block_size, 0, stream.value()>>>(
      d_input_chars, chars_size, input.offset(), map_ref2, d_ranks.data());
    auto const end   = thrust::remove(rmm::exec_policy(stream), d_ranks.begin(), d_ranks.end(), 0);
    auto const total = thrust::distance(d_ranks.begin(), end);  // number of unpairables

    // the new boundaries are combined with the existing offsets to build a temporary column
    auto tmp_offsets = rmm::device_uvector<cudf::size_type>(total + input.size() + 1, stream);
    thrust::merge(rmm::exec_policy(stream),
                  input.offsets_begin(),
                  input.offsets_end(),
                  d_ranks.begin(),
                  end,
                  tmp_offsets.begin());

    // the temp column is used for the encoding functions which is much faster
    // on a larger number of smaller strings
    auto const col_offsets =
      cudf::column_view(cudf::device_span<cudf::size_type const>(tmp_offsets));
    auto const tmp_input     = cudf::column_view(input.parent().type(),
                                             static_cast<cudf::size_type>(input.size() + total),
                                             nullptr,
                                             nullptr,
                                             0,
                                             0,
                                                 {col_offsets, input.chars()});
    auto const d_tmp_strings = cudf::column_device_view::create(tmp_input, stream);

    bpe_parallel_fn<decltype(map_ref)><<<tmp_input.size(), block_size, 0, stream.value()>>>(
      *d_tmp_strings, map_ref, d_spaces.data(), d_ranks.data(), d_rerank.data());
  } else {
    bpe_parallel_fn<decltype(map_ref)><<<input.size(), block_size, 0, stream.value()>>>(
      *d_strings, map_ref, d_spaces.data(), d_ranks.data(), d_rerank.data());
  }

  // compute the output sizes into the d_offsets vector
  bpe_finalize<<<input.size(), block_size, 0, stream.value()>>>(
    *d_strings, d_spaces.data(), d_offsets);

  // convert sizes to offsets in-place
  auto const bytes =
    cudf::detail::sizes_to_offsets(d_offsets, d_offsets + input.size() + 1, d_offsets, stream);
  CUDF_EXPECTS(bytes <= static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max()),
               "Size of output exceeds the column size limit",
               std::overflow_error);

  // build the output: adding separators between the remaining pairs in each string
  auto chars   = cudf::strings::detail::create_chars_child_column(bytes, stream, mr);
  auto d_chars = chars->mutable_view().data<char>();

  // we can reuse the ranks working memory to store some temporary offsets now;
  // the offsets are produced by the index of the d_spaces values
  auto const d_inserts = d_ranks.data();
  // create offsets where separators will be inserted
  auto offsets_at_non_zero = [d_spaces = d_spaces.data()] __device__(auto idx) {
    return d_spaces[idx] > 0;  // separator to be inserted here
  };
  auto const zero_itr  = thrust::counting_iterator<cudf::size_type>(0);
  auto const chars_end = thrust::counting_iterator<cudf::size_type>(chars_size);
  auto const copy_end  = thrust::copy_if(
    rmm::exec_policy(stream), zero_itr + 1, chars_end, d_inserts, offsets_at_non_zero);

  // this will insert the single-byte separator in positions specified in d_inserts
  auto const sep_char = thrust::constant_iterator<char>(separator.to_string(stream)[0]);
  thrust::merge_by_key(rmm::exec_policy(stream),
                       d_inserts,  // where separator is inserted
                       copy_end,
                       zero_itr,  // all positions
                       chars_end,
                       sep_char,  // byte to insert
                       d_input_chars,
                       thrust::make_discard_iterator(),
                       d_chars);  // result

  return cudf::make_strings_column(input.size(),
                                   std::move(offsets),
                                   std::move(chars),
                                   input.null_count(),
                                   cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

}  // namespace detail

std::unique_ptr<cudf::column> byte_pair_encoding(cudf::strings_column_view const& input,
                                                 bpe_merge_pairs const& merges_table,
                                                 cudf::string_scalar const& separator,
                                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::byte_pair_encoding(input, merges_table, separator, cudf::get_default_stream(), mr);
}

}  // namespace nvtext
