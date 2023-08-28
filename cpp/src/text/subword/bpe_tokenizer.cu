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
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

namespace nvtext {
namespace detail {

namespace {

/**
 * @brief Main byte pair encoding algorithm function for each string.
 *
 * @see The byte_pair_encoding_fn::operator() function below for details.
 */
template <typename MapRefType>
struct byte_pair_encoding_fn {
  cudf::column_device_view const d_merges;
  cudf::column_device_view const d_strings;
  cudf::string_view const d_separator;
  MapRefType const d_map;
  cudf::size_type* d_sizes;  // output size of encoded string
  string_hasher_type const hasher;
  cudf::size_type* d_byte_indices;

  /**
   * @brief Get the next substring of the given string.
   *
   * This will find the next sequence of characters identified by the
   * given byte indices iterator values. The beginning of the sequence
   * starts at `begin` and the end of the sequence is the first non-zero
   * index found between (begin,end) exclusive.
   *
   * @tparam Iterator The byte indices iterator type
   * @param begin Start of indices to check
   * @param end End of indices to check
   * @param d_str String to substring
   * @return The substring found.
   */
  template <typename Iterator>
  __device__ cudf::string_view next_substr(Iterator begin,
                                           Iterator end,
                                           cudf::string_view const& d_str)
  {
    auto const next = thrust::find_if(thrust::seq, begin + 1, end, [](auto v) { return v != 0; });
    auto const size = static_cast<cudf::size_type>(thrust::distance(begin, next));
    return cudf::string_view(d_str.data() + *begin, size);
  }

  /**
   * @brief Look up the pair of strings in the d_map/d_merges
   *
   * @param lhs Left half of the string
   * @param rhs Right half of the string
   * @return Position of merge pair within d_map
   */
  __device__ auto get_merge_pair(cudf::string_view const& lhs, cudf::string_view const& rhs)
  {
    auto const mp = merge_pair_type{lhs, rhs};
    return d_map.find(mp);
  }

  /**
   * @brief Byte encode each string.
   *
   * Each string is iteratively scanned for the minimum rank of adjacent substring pairs
   * as found within the `d_map` table. Once the minimum pair is located, that pair
   * is removed -- virtually by zero-ing the index value between any matching adjacent pairs.
   *
   * The iteration ends once there are no more adjacent pairs or there are no more
   * matches found in `d_map`. At the end, the indices for each string reflect the
   * encoding pattern and can be used to build the output.
   *
   * This function also computes the size of the encoded output of each string
   * by simply counting the number of non-zero indices values remaining. This saves
   * an extra kernel launch normally required to compute the offsets of the output column.
   *
   * @param idx The index of the string in `d_strings` to encode
   */
  __device__ void operator()(cudf::size_type idx)
  {
    if (d_strings.is_null(idx)) {
      d_sizes[idx] = 0;
      return;
    }
    auto const d_str = d_strings.element<cudf::string_view>(idx);
    if (d_str.empty()) {
      d_sizes[idx] = 0;
      return;
    }

    auto const offset = d_strings.child(cudf::strings_column_view::offsets_column_index)
                          .element<cudf::size_type>(idx + d_strings.offset());
    auto const d_indices = d_byte_indices + offset;

    // initialize the byte indices for this string;
    // set the index value to 0 for any intermediate UTF-8 bytes
    thrust::transform(thrust::seq,
                      thrust::make_counting_iterator<cudf::size_type>(0),
                      thrust::make_counting_iterator<cudf::size_type>(d_str.size_bytes()),
                      d_indices,
                      [data = d_str.data()](auto idx) {
                        auto const byte = static_cast<uint8_t>(data[idx]);
                        return cudf::strings::detail::is_begin_utf8_char(byte) ? idx : 0;
                      });

    auto const begin = d_indices;
    auto const end   = d_indices + d_str.size_bytes();

    // keep processing the string until there are no more adjacent pairs found in d_map
    cudf::size_type min_rank = 0;
    while (min_rank < cuda::std::numeric_limits<cudf::size_type>::max()) {
      // initialize working variables
      min_rank = cuda::std::numeric_limits<cudf::size_type>::max();

      auto lhs = next_substr(begin, end, d_str);
      auto itr = begin + lhs.size_bytes();

      auto min_itr  = itr;               // these are set along with
      auto min_size = lhs.size_bytes();  // the min_rank variable

      // check each adjacent pair against the d_map
      while (itr < end) {
        auto const rhs = next_substr(itr, end, d_str);
        if (rhs.empty()) { break; }  // no more adjacent pairs

        auto const map_itr = get_merge_pair(lhs, rhs);
        if (map_itr != d_map.end()) {
          // found a match; record the rank (and other min_ vars)
          auto const rank = map_itr->second;
          if (rank < min_rank) {
            min_rank = rank;
            min_itr  = itr;
            min_size = rhs.size_bytes();
          }
        }
        // next substring
        lhs = rhs;
        itr += rhs.size_bytes();
      }

      // if any pair matched, remove every occurrence from the string
      if (min_rank < cuda::std::numeric_limits<cudf::size_type>::max()) {
        // remove the first pair we found
        itr  = min_itr;
        *itr = 0;

        // continue scanning for other occurrences in the remainder of the string
        itr += min_size;
        if (itr < end) {
          // auto const d_pair = dissect_merge_pair(min_rank);
          auto const d_pair = dissect_merge_pair(d_merges.element<cudf::string_view>(min_rank));

          lhs = next_substr(itr, end, d_str);
          itr += lhs.size_bytes();
          while (itr < end && !lhs.empty()) {
            auto rhs = next_substr(itr, end, d_str);
            if (d_pair.first == lhs && d_pair.second == rhs) {
              *itr = 0;                   // removes the pair from this string
              itr += rhs.size_bytes();
              if (itr >= end) { break; }  // done checking for pairs
              // skip to the next adjacent pair
              rhs = next_substr(itr, end, d_str);
            }
            // next substring
            lhs = rhs;
            itr += rhs.size_bytes();
          }
        }
      }
    }

    // compute and store the output size for this string's encoding
    auto separators_size =
      thrust::count_if(
        thrust::seq, d_indices, d_indices + d_str.size_bytes(), [](auto v) { return v != 0; }) *
      d_separator.size_bytes();
    d_sizes[idx] = static_cast<cudf::size_type>(d_str.size_bytes() + separators_size);
  }
};

/**
 * @brief Build the output string encoding.
 *
 * This copies each string to the output inserting a space at each non-zero byte index.
 *
 * @code{.txt}
 * d_strings =      ["helloworld", "testthis"]
 * d_byte_indices = [ 0000050000    00004000]
 * result is ["hello world", "test this"]
 * @endcode
 */
struct build_encoding_fn {
  cudf::column_device_view const d_strings;
  cudf::string_view const d_separator;
  cudf::size_type const* d_byte_indices;
  cudf::size_type const* d_offsets;
  char* d_chars{};

  __device__ void operator()(cudf::size_type idx)
  {
    if (d_strings.is_null(idx)) { return; }
    auto const d_str = d_strings.element<cudf::string_view>(idx);
    if (d_str.empty()) { return; }

    auto const offset = d_strings.child(cudf::strings_column_view::offsets_column_index)
                          .element<cudf::size_type>(idx + d_strings.offset());
    auto const d_indices = d_byte_indices + offset;
    auto d_output        = d_chars + d_offsets[idx];

    // copy chars while indices[i]==0,
    // insert space each time indices[i]!=0
    auto const begin = d_indices;
    auto const end   = d_indices + d_str.size_bytes();
    auto d_input     = d_str.data();
    *d_output++      = *d_input++;
    auto itr         = begin + 1;
    while (itr < end) {
      if (*itr++) { d_output = cudf::strings::detail::copy_string(d_output, d_separator); }
      *d_output++ = *d_input++;
    }
    // https://github.com/rapidsai/cudf/pull/10270/files#r826319405
  }
};

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

  auto const d_strings = cudf::column_device_view::create(input.parent(), stream);

  // build working vector to hold index values per byte
  rmm::device_uvector<cudf::size_type> d_byte_indices(input.chars().size(), stream);

  auto offsets   = cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<cudf::size_type>()},
                                           static_cast<cudf::size_type>(input.size() + 1),
                                           cudf::mask_state::UNALLOCATED,
                                           stream,
                                           rmm::mr::get_current_device_resource());
  auto d_offsets = offsets->mutable_view().data<cudf::size_type>();

  auto const d_merges = merge_pairs.impl->get_merge_pairs();
  auto const map_ref  = merge_pairs.impl->get_merge_pairs_ref();
  byte_pair_encoding_fn<decltype(map_ref)> fn{d_merges,
                                              *d_strings,
                                              d_separator,
                                              map_ref,
                                              d_offsets,
                                              string_hasher_type{},
                                              d_byte_indices.data()};
  thrust::for_each_n(
    rmm::exec_policy(stream), thrust::make_counting_iterator<cudf::size_type>(0), input.size(), fn);

  // build the output: add spaces between the remaining pairs in each string
  thrust::exclusive_scan(
    rmm::exec_policy(stream), d_offsets, d_offsets + input.size() + 1, d_offsets);

  auto const bytes =
    cudf::detail::get_value<cudf::size_type>(offsets->view(), input.size(), stream);
  auto chars = cudf::strings::detail::create_chars_child_column(
    bytes, stream, rmm::mr::get_current_device_resource());
  auto d_chars = chars->mutable_view().data<char>();

  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<cudf::size_type>(0),
    input.size(),
    build_encoding_fn{*d_strings, d_separator, d_byte_indices.data(), d_offsets, d_chars});

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
