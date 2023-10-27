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

#include <text/bpe/byte_pair_encoding.cuh>

#include <nvtext/byte_pair_encoding.hpp>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/combine.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/merge.h>
#include <thrust/pair.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

namespace nvtext {
namespace detail {

namespace {

template <typename CharType>
constexpr bool is_whitespace(CharType ch)
{
  return ch <= ' ';
}

/**
 * @brief Resolve a substring up to the first whitespace character.
 *
 * This will return a substring of the input starting with the first byte
 * up to the first whitespace character found or the end of the string.
 * Any whitespace is expected only at the end of the string.
 *
 * @param d_str Input string to resolve.
 * @return Substring of the input excluding any trailing whitespace.
 */
__device__ cudf::string_view get_first_token(cudf::string_view const& d_str)
{
  auto const begin = d_str.data();
  auto const end   = thrust::find_if(
    thrust::seq, begin, begin + d_str.size_bytes(), [](auto ch) { return is_whitespace(ch); });
  auto const size = static_cast<cudf::size_type>(thrust::distance(begin, end));
  return cudf::string_view(begin, size);
}

/**
 * @brief Main byte pair encoding algorithm function for each string.
 *
 * @see The byte_pair_encoding_fn::operator() function below for details.
 */
template <typename MapRefType>
struct byte_pair_encoding_fn {
  cudf::column_device_view const d_merges;
  cudf::column_device_view const d_strings;
  MapRefType const d_map;
  cudf::size_type* d_sizes;  // output size of encoded string
  string_hasher_type const hasher;
  cudf::size_type* d_byte_indices;

  /**
   * @brief Parse the merge pair into components.
   *
   * The two substrings are separated by a single space.
   *
   * @param idx Index of merge pair to dissect.
   * @return The left and right halves of the merge pair.
   */
  __device__ thrust::pair<cudf::string_view, cudf::string_view> dissect_merge_pair(
    cudf::size_type idx)
  {
    auto const d_pair  = d_merges.element<cudf::string_view>(idx);
    auto const lhs     = d_pair.data();
    auto const end_str = d_pair.data() + d_pair.size_bytes();
    auto const rhs     = thrust::find(thrust::seq, lhs, end_str, ' ');  // space always expected
    // check for malformed pair entry to prevent segfault
    if (rhs == end_str) { return thrust::make_pair(cudf::string_view{}, cudf::string_view{}); }
    auto const lhs_size = static_cast<cudf::size_type>(thrust::distance(lhs, rhs));
    auto const rhs_size = static_cast<cudf::size_type>(thrust::distance(rhs + 1, end_str));
    return thrust::make_pair(cudf::string_view(lhs, lhs_size),
                             cudf::string_view(rhs + 1, rhs_size));
  }

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
    __shared__ char shmem[48 * 1024];  // max for Pascal
    auto const total_size         = lhs.size_bytes() + rhs.size_bytes() + 1;
    auto const thread_memory_size = static_cast<cudf::size_type>(sizeof(shmem) / blockDim.x);

    // Edge case check.
    // Empirically found only two merge pair strings that were greater than 70 bytes
    // and they both looked like ignorable errors.
    if (thread_memory_size < total_size) { return d_map.end(); }

    // build the target string in shared memory
    char* ptr = &shmem[threadIdx.x * thread_memory_size];

    // build a temp string like:  temp = lhs + ' ' + rhs
    memcpy(ptr, lhs.data(), lhs.size_bytes());
    memcpy(ptr + lhs.size_bytes(), " ", 1);
    memcpy(ptr + lhs.size_bytes() + 1, rhs.data(), rhs.size_bytes());

    auto const d_str = cudf::string_view(ptr, total_size);
    return d_map.find(d_str);
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
    auto const d_str = get_first_token(d_strings.element<cudf::string_view>(idx));
    if (d_str.empty()) {
      d_sizes[idx] = 0;
      return;
    }

    auto const offset = d_strings.child(cudf::strings_column_view::offsets_column_index)
                          .element<cudf::size_type>(idx);
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
        if (rhs.empty()) break;  // no more adjacent pairs

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
          auto const d_pair = dissect_merge_pair(min_rank);

          lhs = next_substr(itr, end, d_str);
          itr += lhs.size_bytes();
          while (itr < end) {
            auto rhs = next_substr(itr, end, d_str);
            if (d_pair.first == lhs && d_pair.second == rhs) {
              *itr = 0;  // removes the pair from this string
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
    auto const encoded_size = d_str.size_bytes() +  // number of original bytes +
                              thrust::count_if(     // number of non-zero byte indices
                                thrust::seq,
                                d_indices,
                                d_indices + d_str.size_bytes(),
                                [](auto v) { return v != 0; });
    d_sizes[idx] = static_cast<cudf::size_type>(encoded_size);
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
  cudf::size_type const* d_byte_indices;
  cudf::size_type const* d_offsets;
  char* d_chars{};

  __device__ void operator()(cudf::size_type idx)
  {
    if (d_strings.is_null(idx)) { return; }
    auto const d_str = get_first_token(d_strings.element<cudf::string_view>(idx));
    if (d_str.empty()) { return; }

    auto const offset = d_strings.child(cudf::strings_column_view::offsets_column_index)
                          .element<cudf::size_type>(idx);
    auto const d_indices = d_byte_indices + offset;
    auto d_output        = d_chars ? d_chars + d_offsets[idx] : nullptr;

    // copy chars while indices[i]==0,
    // insert space each time indices[i]!=0
    auto const begin = d_indices;
    auto const end   = d_indices + d_str.size_bytes();
    auto d_input     = d_str.data();
    *d_output++      = *d_input++;
    auto itr         = begin + 1;
    while (itr < end) {
      if (*itr++) *d_output++ = ' ';
      *d_output++ = *d_input++;
    }
    // https://github.com/rapidsai/cudf/pull/10270/files#r826319405
  }
};

/**
 * @brief Perform byte pair encoding on each string in the input column.
 *
 * The result is a strings column of the same size where each string has been encoded.
 *
 * The encoding is performed iteratively. Each pass determines the string's lowest
 * ranked merge pair as determined by the strings in `merges_table`. This pair
 * is removed (virtually) from each string before starting the next iteration.
 *
 * Once all pairs have exhausted for all strings, the output is constructed from
 * the results by adding spaces between each remaining pair in each string.
 *
 * @param input Strings to encode.
 * @param merge_pairs Merge pairs data and map used for encoding.
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
std::unique_ptr<cudf::column> byte_pair_encoding(
  cudf::strings_column_view const& input,
  bpe_merge_pairs::bpe_merge_pairs_impl const& merge_pairs,
  rmm::cuda_stream_view stream)
{
  auto const d_merges = merge_pairs.get_merge_pairs();
  CUDF_EXPECTS(d_merges.size() > 0, "Merge pairs table must not be empty");

  // build working vector to hold index values per byte
  rmm::device_uvector<cudf::size_type> d_byte_indices(input.chars().size(), stream);

  auto const d_strings = cudf::column_device_view::create(input.parent(), stream);

  auto offsets   = cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<cudf::size_type>()},
                                           static_cast<cudf::size_type>(input.size() + 1),
                                           cudf::mask_state::UNALLOCATED,
                                           stream,
                                           rmm::mr::get_current_device_resource());
  auto d_offsets = offsets->mutable_view().data<cudf::size_type>();

  auto map_ref = merge_pairs.get_merge_pairs_ref();
  byte_pair_encoding_fn<decltype(map_ref)> fn{
    d_merges, *d_strings, map_ref, d_offsets, string_hasher_type{}, d_byte_indices.data()};
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

  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<cudf::size_type>(0),
                     input.size(),
                     build_encoding_fn{*d_strings, d_byte_indices.data(), d_offsets, d_chars});

  return make_strings_column(
    input.size(), std::move(offsets), std::move(chars), 0, rmm::device_buffer{});
}

/**
 * @brief Detect space to not-space transitions inside each string.
 *
 * This handles sliced input and null strings as well.
 * It is parallelized over bytes and returns true only for valid left edges
 * -- non-space preceded by a space.
 */
struct edge_of_space_fn {
  cudf::column_device_view const d_strings;
  __device__ bool operator()(cudf::size_type offset)
  {
    auto const d_chars =
      d_strings.child(cudf::strings_column_view::chars_column_index).data<char>();
    if (is_whitespace(d_chars[offset]) || !is_whitespace(d_chars[offset - 1])) { return false; }

    auto const offsets   = d_strings.child(cudf::strings_column_view::offsets_column_index);
    auto const d_offsets = offsets.data<cudf::size_type>() + d_strings.offset();
    // ignore offsets outside sliced range
    if (offset < d_offsets[0] || offset >= d_offsets[d_strings.size()]) { return false; }

    auto itr =
      thrust::lower_bound(thrust::seq, d_offsets, d_offsets + d_strings.size() + 1, offset);
    // ignore offsets at existing string boundaries
    if (*itr == offset) { return false; }

    // count only edges for valid strings
    auto const index = static_cast<cudf::size_type>(thrust::distance(d_offsets, itr)) - 1;
    return d_strings.is_valid(index);
  }
};

/**
 * @brief Create new offsets by identifying substrings by whitespace.
 *
 * This is similar to cudf::strings::split_record but does not fully split
 * and only returns new offsets. The behavior is more like a view-only slice
 * of the chars child with the result still including trailing delimiters.
 *
 * The encoding algorithm ignores the trailing whitespace of each string.
 *
 * @param input Strings to tokenize.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return New offsets including those at the edge of each space.
 */
std::unique_ptr<cudf::column> space_offsets(cudf::strings_column_view const& input,
                                            cudf::column_device_view const& d_strings,
                                            rmm::cuda_stream_view stream)
{
  // count space offsets
  auto const begin = thrust::make_counting_iterator<cudf::size_type>(1);
  auto const end   = thrust::make_counting_iterator<cudf::size_type>(input.chars().size());
  edge_of_space_fn edge_of_space{d_strings};
  auto const space_count = thrust::count_if(rmm::exec_policy(stream), begin, end, edge_of_space);

  // copy space offsets
  rmm::device_uvector<cudf::size_type> space_offsets(space_count, stream);
  thrust::copy_if(rmm::exec_policy(stream), begin, end, space_offsets.data(), edge_of_space);

  // create output offsets
  auto result =
    cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<cudf::size_type>()},
                              static_cast<cudf::size_type>(space_count + input.size() + 1),
                              cudf::mask_state::UNALLOCATED,
                              stream,
                              rmm::mr::get_current_device_resource());

  // combine current offsets with space offsets
  thrust::merge(rmm::exec_policy(stream),
                input.offsets_begin(),
                input.offsets_end(),
                space_offsets.begin(),
                space_offsets.end(),
                result->mutable_view().begin<cudf::size_type>());

  return result;
}

/**
 * @brief Build new offsets that can be used to build a list column for calling join.
 *
 * This essentially returns the number of tokens for each string.
 */
struct list_offsets_fn {
  cudf::column_device_view const d_strings;
  __device__ cudf::size_type operator()(cudf::size_type idx)
  {
    if (d_strings.is_null(idx)) return 0;
    auto const d_str = d_strings.element<cudf::string_view>(idx);
    if (d_str.empty()) return 1;  // empty is a single valid result

    auto const begin = thrust::make_counting_iterator<cudf::size_type>(1);
    auto const end   = thrust::make_counting_iterator<cudf::size_type>(d_str.size_bytes());

    // this counts the number of non-adjacent delimiters
    auto const result =
      thrust::count_if(thrust::seq, begin, end, [data = d_str.data()](auto chidx) {
        return !is_whitespace(data[chidx]) && is_whitespace(data[chidx - 1]);
      });
    return static_cast<cudf::size_type>(result) + 1;
  }
};

}  // namespace

std::unique_ptr<cudf::column> byte_pair_encoding(cudf::strings_column_view const& input,
                                                 bpe_merge_pairs const& merge_pairs,
                                                 cudf::string_scalar const& separator,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::mr::device_memory_resource* mr)
{
  if (input.is_empty() || input.chars_size() == 0)
    return cudf::make_empty_column(cudf::type_id::STRING);

  auto const d_strings = cudf::column_device_view::create(input.parent(), stream);
  auto const offsets   = space_offsets(input, *d_strings, stream);

  // build a view using the new offsets and the current input chars column
  auto const input_view = cudf::column_view(cudf::data_type{cudf::type_id::STRING},
                                            offsets->size() - 1,
                                            nullptr,  // no parent data
                                            nullptr,  // null-mask
                                            0,        // null-count
                                            0,        // offset
                                            {offsets->view(), input.chars()});

  // run BPE on this view
  auto const bpe_column =
    byte_pair_encoding(cudf::strings_column_view(input_view), *(merge_pairs.impl), stream);

  // recombine the result:
  // compute the offsets needed to build a list view
  auto const list_offsets = [d_strings = *d_strings, stream] {
    auto offsets_itr = thrust::make_transform_iterator(
      thrust::make_counting_iterator<cudf::size_type>(0), list_offsets_fn{d_strings});
    auto offsets_column = std::get<0>(cudf::detail::make_offsets_child_column(
      offsets_itr, offsets_itr + d_strings.size(), stream, rmm::mr::get_current_device_resource()));
    return offsets_column;
  }();

  // build a list column_view using the BPE output and the list_offsets
  auto const list_join = cudf::column_view(cudf::data_type{cudf::type_id::LIST},
                                           input.size(),
                                           nullptr,  // no parent data in list column
                                           input.null_mask(),
                                           input.null_count(),
                                           0,
                                           {list_offsets->view(), bpe_column->view()});

  // build the output strings column
  auto result =
    cudf::strings::detail::join_list_elements(cudf::lists_column_view(list_join),
                                              separator,
                                              cudf::string_scalar(""),
                                              cudf::strings::separator_on_nulls::NO,
                                              cudf::strings::output_if_empty_list::EMPTY_STRING,
                                              stream,
                                              mr);
  return result;
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
