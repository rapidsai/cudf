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
#include <cudf/reduction/detail/segmented_reduction.cuh>
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
#include <thrust/transform.h>

namespace nvtext {
namespace detail {
namespace {

constexpr int block_size = 512;

__device__ __inline__ cudf::string_view next_substr(cudf::column_device_view const& d_strings,
                                                    int* d_spaces,
                                                    int* begin,
                                                    int* end,
                                                    cudf::string_view const& d_str)
{
  auto const next = thrust::find_if(thrust::seq, begin + 1, end, [](auto v) { return v != 0; });
  auto const size = static_cast<cudf::size_type>(thrust::distance(begin, next));
  return cudf::string_view(d_str.data() + thrust::distance(d_spaces, begin), size);
}

template <typename MapRefType>
__global__ void bpe_parallel_fn(cudf::column_device_view const d_strings,
                                MapRefType const d_map,
                                cudf::size_type* d_sizes,      // output size of encoded string
                                cudf::size_type* d_spaces_in,  // output per string
                                cudf::size_type* d_working     // working memory
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

  auto const d_spaces    = d_spaces_in + offset;
  auto const end_spaces  = d_spaces + d_str.size_bytes();
  auto const d_min_ranks = d_working + offset;
  auto const end_ranks   = d_min_ranks + d_str.size_bytes();
  auto const max_rank    = cuda::std::numeric_limits<cudf::size_type>::max();

  __shared__ cudf::size_type block_min_rank;
  using block_reduce = cub::BlockReduce<cudf::size_type, block_size>;
  __shared__ typename block_reduce::TempStorage temp_storage;

  if (lane_idx == 0) {
    // the first character is free so we store the string's size here
    // to help compute the encoded output size later
    *d_spaces      = d_str.size_bytes();
    block_min_rank = 0;
  }
  __syncthreads();

  // each thread processes their part of the string and records its min_rank
  while (block_min_rank < max_rank) {
    auto min_rank = max_rank;
    // initialize min ranks
    // future optimization: only invalidate ranks where mins were
    // found in the previous run
    for (auto itr = d_min_ranks + lane_idx; itr < end_ranks; itr += block_size) {
      *itr = max_rank;
    }
    __syncthreads();

    for (auto itr = d_spaces + lane_idx; itr < end_spaces; itr += block_size) {
      if (*itr == 0) { continue; }  // start on valid bytes only

      // get left half of the pair
      auto const lhs      = next_substr(d_strings, d_spaces, itr, end_spaces, d_str);
      auto const next_itr = itr + lhs.size_bytes();
      if (next_itr < end_spaces) {
        // get the right half of the pair
        auto const rhs = next_substr(d_strings, d_spaces, next_itr, end_spaces, d_str);
        if (!rhs.empty()) {
          auto const index = static_cast<int>(thrust::distance(d_spaces, next_itr));
          // this is setup for future optimization mentioned above;
          // we only want to hash/lookup if the rank is new for this pair
          auto rank = d_min_ranks[index];
          if (rank == max_rank) {
            // lookup pair in merge-pairs table
            auto const mp      = merge_pair_type{lhs, rhs};
            auto const map_itr = d_map.find(mp);
            if (map_itr != d_map.end()) {  // found a match
              rank = map_itr->second;
            }
          }
          if (rank < min_rank) { min_rank = rank; }
          d_min_ranks[index] = rank;  // store the rank
        }
      }
    }
    __syncthreads();

    // once all threads are completed, find the min-rank across the block
    block_min_rank = block_reduce(temp_storage).Reduce(min_rank, cub::Min());
    __syncthreads();

    if (block_min_rank < max_rank) {
      // search the d_min_ranks for all the places where the rank matches block_min_rank
      for (auto itr = d_min_ranks + lane_idx; itr < end_ranks; itr += block_size) {
        auto const index = static_cast<int>(thrust::distance(d_min_ranks, itr));
        if (*itr == block_min_rank) {
          // set the output value to 0 at this position
          if (index > 0 && *(itr - 1) != block_min_rank) { d_spaces[index] = 0; }
        }
      }
      __syncthreads();
    }
  }  // if no mins were found we are done, otherwise start again
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

  // initialize the spaces vector which will hold encoding information
  rmm::device_uvector<cudf::size_type> d_spaces(chars_size, stream);
  auto const zero_itr  = thrust::counting_iterator<cudf::size_type>(0);
  auto const chars_end = thrust::counting_iterator<cudf::size_type>(chars_size);
  thrust::transform(rmm::exec_policy(stream),
                    zero_itr,
                    chars_end,
                    d_spaces.begin(),
                    [d_input_chars] __device__(auto idx) {
                      return static_cast<cudf::size_type>(
                        cudf::strings::detail::is_begin_utf8_char(d_input_chars[idx]));
                    });

  rmm::device_uvector<cudf::size_type> d_working(chars_size, stream);
  auto const map_ref = merge_pairs.impl->get_merge_pairs_ref();

  // encoding step produces values in d_spaces that indicate where the separator is inserted
  // and can also be reduced to compute the output size of each row
  bpe_parallel_fn<decltype(map_ref)><<<input.size(), block_size, 0, stream.value()>>>(
    *d_strings, map_ref, d_offsets, d_spaces.data(), d_working.data());
  // compute and store the output size for this string's encoding
  auto const input_offsets = thrust::make_transform_iterator(
    input.offsets_begin(),
    [first_offset] __device__(auto offset) { return offset - first_offset; });
  cudf::reduction::detail::segmented_reduce(d_spaces.begin(),
                                            input_offsets,
                                            input_offsets + input.size() + 1,
                                            d_offsets,
                                            thrust::plus{},
                                            0,
                                            stream);
  // convert sizes to offsets
  auto const bytes =
    cudf::detail::sizes_to_offsets(d_offsets, d_offsets + input.size() + 1, d_offsets, stream);
  CUDF_EXPECTS(bytes <= static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max()),
               "Size of output exceeds the column size limit",
               std::overflow_error);

  // build the output: add spaces between the remaining pairs in each string
  auto chars   = cudf::strings::detail::create_chars_child_column(bytes, stream, mr);
  auto d_chars = chars->mutable_view().data<char>();

  // we can reuse the d_working memory to store some temporary offsets now
  auto const d_inserts = d_working.data();
  // create offsets where separators will be inserted
  auto offsets_at_one = [d_spaces = d_spaces.data()] __device__(auto idx) {
    return d_spaces[idx] == 1;  // this fails if any input string is a single byte
  };
  auto const copy_end =
    thrust::copy_if(rmm::exec_policy(stream), zero_itr + 1, chars_end, d_inserts, offsets_at_one);

  // this will insert the single-byte separator in positions specified in d_inserts
  auto const sep_char = thrust::constant_iterator<char>(separator.to_string(stream)[0]);
  thrust::merge_by_key(rmm::exec_policy(stream),
                       d_inserts,  // where separator is inserted
                       copy_end,
                       zero_itr,   // all positions
                       chars_end,
                       sep_char,   // byte to insert
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
