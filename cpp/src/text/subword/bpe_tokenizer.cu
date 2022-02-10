/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <nvtext/bpe_tokenize.hpp>

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/strings/detail/combine.hpp>
#include <cudf/strings/detail/split.hpp>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/for_each.h>
#include <thrust/reduce.h>

namespace nvtext {
namespace detail {

namespace {

/**
 * @brief Initialize the byte indices and the pair rank for each string.
 */
struct initialize_indices_fn {
  cudf::column_device_view const d_merges;
  cudf::column_device_view const d_strings;
  cudf::size_type* d_byte_indices;
  cudf::size_type* d_min_ranks;

  __device__ void operator()(cudf::size_type idx)
  {
    d_min_ranks[idx] = cuda::std::numeric_limits<cudf::size_type>::max();

    if (d_strings.is_null(idx)) { return; }

    auto const d_str = d_strings.element<cudf::string_view>(idx);
    if (d_str.empty()) { return; }

    auto const offset = d_strings.child(cudf::strings_column_view::offsets_column_index)
                          .element<cudf::offset_type>(idx);
    auto d_indices = d_byte_indices + offset;

    // set the index value for each byte
    for (auto i = 0; i < d_str.size_bytes(); ++i) {
      auto const byte = static_cast<uint8_t>(d_str.data()[i]);
      // for intermediate UTF-8 bytes set the index value to 0
      d_indices[i] = cudf::strings::detail::is_begin_utf8_char(byte) ? i : 0;
    }
  }
};

/**
 * @brief Parse the merge pair into components.
 *
 * The two substrings are separated by a single space.
 *
 * @param d_pair String to dissect
 * @return The left and right halves of the input pair.
 */
__device__ thrust::pair<cudf::string_view, cudf::string_view> dissect_merge_pair(
  cudf::string_view const& d_pair)
{
  auto const lhs      = d_pair.data();
  auto const end_str  = d_pair.data() + d_pair.size_bytes();
  auto const rhs      = thrust::find(thrust::seq, lhs, end_str, ' ') + 1;
  auto const lhs_size = static_cast<cudf::size_type>(thrust::distance(lhs, rhs - 1));
  auto const rhs_size = static_cast<cudf::size_type>(thrust::distance(rhs, end_str));
  return thrust::make_pair(cudf::string_view(lhs, lhs_size), cudf::string_view(rhs, rhs_size));
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
 * @brief Iterate over the merge pairs and to find the minimum rank in each string.
 *
 * As a merge pair is located in each string, the minimum rank is accumulated in
 * the output `d_min_ranks`. The rank is simply the position of the merge pair
 * in the `d_merges` column.
 */
struct find_minimum_pair_fn {
  cudf::column_device_view const d_merges;
  cudf::column_device_view const d_strings;
  cudf::size_type* d_byte_indices;
  cudf::size_type* d_min_ranks;

  // index is over the merges table
  __device__ void operator()(cudf::size_type index)
  {
    auto const d_pair = dissect_merge_pair(d_merges.element<cudf::string_view>(index));

    // locate this pair in each string
    for (auto idx = 0; idx < d_strings.size(); ++idx) {
      if (d_strings.is_null(idx)) continue;
      auto const d_str = d_strings.element<cudf::string_view>(idx);
      if (d_str.empty()) continue;

      auto const offset = d_strings.child(cudf::strings_column_view::offsets_column_index)
                            .element<cudf::offset_type>(idx);
      auto d_indices = d_byte_indices + offset;

      auto const begin = d_indices;
      auto const end   = d_indices + d_str.size_bytes();

      // check for the merge-pair in this string
      auto lhs = next_substr(begin, end, d_str);
      auto itr = begin + lhs.size_bytes();
      while (itr < end) {
        auto rhs = next_substr(itr, end, d_str);
        if (rhs.empty()) break;

        if (d_pair.first == lhs && d_pair.second == rhs) {
          // found a match, record the rank
          atomicMin(d_min_ranks + idx, index);
          break;  // done with this string
        }

        // next substring
        lhs = rhs;
        itr += rhs.size_bytes();
      }
    }
  }
};

/**
 * @brief Remove merge pair from each string.
 *
 * The minimum rank found for each string used to identify the pair(s)
 * to be removed. The pairs are removed by just zeroing the byte index
 * found between the adjacent substrings.
 *
 * @code{.txt}
 * d_strings =        ["helloworld", "testisthis"]
 * d_byte_indices =   [ 0123456789    01234567]
 * d_merges[d_min_ranks] = [ "ll o", "i s" ]
 *
 * d_bytes_indices -> [ 0123056789 01234060 ]
 * d_min_ranks is reset to [ max, max ]
 * @endcode
 *
 */
struct remove_pair_fn {
  cudf::column_device_view const d_merges;
  cudf::column_device_view const d_strings;
  cudf::size_type* d_byte_indices;
  cudf::size_type* d_min_ranks;

  __device__ void operator()(cudf::size_type idx)
  {
    if (d_strings.is_null(idx)) return;
    auto const d_str = d_strings.element<cudf::string_view>(idx);
    if (d_str.empty()) return;

    auto rank = d_min_ranks[idx];
    if (rank == cuda::std::numeric_limits<cudf::size_type>::max()) return;

    auto const d_pair = dissect_merge_pair(d_merges.element<cudf::string_view>(rank));

    // resolve byte indices for this string
    auto const offset = d_strings.child(cudf::strings_column_view::offsets_column_index)
                          .element<cudf::offset_type>(idx);
    auto d_indices = d_byte_indices + offset;

    auto const begin = d_indices;
    auto const end   = d_indices + d_str.size_bytes();

    // locate d_pair and remove it from this string
    auto lhs = next_substr(begin, end, d_str);
    auto itr = begin + lhs.size_bytes();
    while (itr < end) {
      auto rhs = next_substr(itr, end, d_str);
      if (d_pair.first == lhs && d_pair.second == rhs) {
        *itr = 0;  // removes the pair from this string
        itr += rhs.size_bytes();
        if (itr < end) {
          rhs = next_substr(itr, end, d_str);  // skip to the next pair
        } else {
          break;  // done with this string
        }
      }
      // next substring
      lhs = rhs;
      itr += rhs.size_bytes();
    }

    // reset for next iteration
    d_min_ranks[idx] = cuda::std::numeric_limits<cudf::size_type>::max();
  }
};

/**
 * @brief Computes the output size of each string.
 *
 * The output size is the size of the current string plus the
 * number of spaces to be added between adjacent substrings.
 * The number of spaces will equal the number of non-zero byte indices
 * for the string.
 */
struct compute_sizes_fn {
  cudf::column_device_view const d_strings;
  cudf::size_type* d_byte_indices;

  __device__ cudf::size_type operator()(cudf::size_type idx)
  {
    if (d_strings.is_null(idx)) return 0;
    auto const d_str = d_strings.element<cudf::string_view>(idx);
    auto offset      = d_strings.child(cudf::strings_column_view::offsets_column_index)
                    .element<cudf::offset_type>(idx);
    auto d_indices = d_byte_indices + offset;
    return d_str.size_bytes() + thrust::count_if(  // number of non-zero byte indices
                                  thrust::seq,
                                  d_indices,
                                  d_indices + d_str.size_bytes(),
                                  [](auto v) { return v != 0; });
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
  cudf::size_type* d_byte_indices;
  cudf::offset_type const* d_offsets;
  char* d_chars{};

  __device__ void operator()(cudf::size_type idx)
  {
    if (d_strings.is_null(idx)) return;
    auto const d_str = d_strings.element<cudf::string_view>(idx);
    if (d_str.empty()) return;

    auto offset = d_strings.child(cudf::strings_column_view::offsets_column_index)
                    .element<cudf::offset_type>(idx);
    auto d_indices = d_byte_indices + offset;
    auto d_output  = d_chars ? d_chars + d_offsets[idx] : nullptr;

    // copy chars while indices==0, add space each time indices!=0
    auto begin   = d_indices;
    auto end     = d_indices + d_str.size_bytes();
    auto d_input = d_str.data();
    *d_output++  = *d_input++;
    auto itr     = begin + 1;
    while (itr < end) {
      if (*itr++) *d_output++ = ' ';
      *d_output++ = *d_input++;
    }
  }
};

/**
 * @brief Perform byte pair encoding on each string in the input column.
 *
 * The result is a strings column of the same size where each string has been encoded.
 *
 * The encoding is performed iteratively. Each pass determines the string's lowest
 * ranked merge pair as determined by the strings in `merges_table`. This pair
 * is the removed (virtually) from each string before starting the next iteration.
 *
 * Once all pairs have exhausted for all strings, the output is constructed from
 * the results by adding spaces between each remaining pair in each string.
 */
std::unique_ptr<cudf::column> byte_pair_encoding(
  cudf::strings_column_view const& input,
  bpe_merge_pairs const& merges_table,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  // build working vectors
  rmm::device_uvector<cudf::size_type> d_byte_indices(input.chars().size(), stream);
  rmm::device_uvector<cudf::size_type> d_min_ranks(input.size(), stream);

  auto d_merges  = cudf::column_device_view::create(merges_table.merge_pairs->view(), stream);
  auto d_strings = cudf::column_device_view::create(input.parent(), stream);

  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<cudf::size_type>(0),
    input.size(),
    initialize_indices_fn{*d_merges, *d_strings, d_byte_indices.data(), d_min_ranks.data()});

  cudf::size_type min_rank = 0;
  while (min_rank < std::numeric_limits<cudf::size_type>::max()) {
    // find minimum merge pair for each string
    thrust::for_each_n(
      rmm::exec_policy(stream),
      thrust::make_counting_iterator<cudf::size_type>(0),
      d_merges->size(),
      find_minimum_pair_fn{*d_merges, *d_strings, d_byte_indices.data(), d_min_ranks.data()});

    // get the minimum rank over all strings;
    // this is only used to see if we are finished
    min_rank = thrust::reduce(rmm::exec_policy(stream),
                              d_min_ranks.begin(),
                              d_min_ranks.end(),
                              std::numeric_limits<cudf::size_type>::max(),
                              thrust::minimum<cudf::size_type>{});

    // check if any pairs have been found;
    // if so, remove that pair from each string
    if (min_rank < std::numeric_limits<cudf::size_type>::max()) {
      thrust::for_each_n(
        rmm::exec_policy(stream),
        thrust::make_counting_iterator<cudf::size_type>(0),
        input.size(),
        remove_pair_fn{*d_merges, *d_strings, d_byte_indices.data(), d_min_ranks.data()});
    }
  }

  // build the output:
  // add spaces between the remaining pairs in each string
  auto offsets_itr =
    thrust::make_transform_iterator(thrust::make_counting_iterator<cudf::size_type>(0),
                                    compute_sizes_fn{*d_strings, d_byte_indices.data()});
  auto offsets = cudf::strings::detail::make_offsets_child_column(
    offsets_itr, offsets_itr + input.size(), stream, mr);
  auto d_offsets = offsets->view().data<cudf::offset_type>();

  auto const bytes = cudf::detail::get_value<int32_t>(offsets->view(), input.size(), stream);
  auto chars       = cudf::strings::detail::create_chars_child_column(bytes, stream, mr);
  auto d_chars     = chars->mutable_view().data<char>();
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<cudf::size_type>(0),
                     input.size(),
                     build_encoding_fn{*d_strings, d_byte_indices.data(), d_offsets, d_chars});

  return make_strings_column(input.size(),
                             std::move(offsets),
                             std::move(chars),
                             input.null_count(),
                             cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

}  // namespace

std::unique_ptr<cudf::column> byte_pair_encoding(cudf::strings_column_view const& input,
                                                 bpe_merge_pairs const& merges_table,
                                                 cudf::string_scalar const& separator,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::mr::device_memory_resource* mr)
{
  auto const strings_count = input.size();
  if (strings_count == 0 || input.chars_size() == 0)
    return cudf::make_empty_column(cudf::type_id::STRING);
  CUDF_EXPECTS(!merges_table.merge_pairs->view().is_empty(), "Merge pairs table must not be empty");

  // split input on whitespace
  auto split_result = cudf::strings::detail::split_record(
    input, cudf::string_scalar(""), -1, stream, rmm::mr::get_current_device_resource());
  auto split_view = cudf::lists_column_view(split_result->view());

  // run BPE on the strings child column
  auto bpe_column = byte_pair_encoding(split_view.child(), merges_table, stream);

  // recombine the result:
  // use the offsets from split_record and the strings from byte_pair_encoding
  // to build a lists column_view
  auto list_join = cudf::column_view(cudf::data_type{cudf::type_id::LIST},
                                     strings_count,
                                     nullptr,  // no parent data in list column
                                     split_view.null_mask(),
                                     split_view.null_count(),
                                     0,
                                     {split_view.offsets(), bpe_column->view()});

  // use join_list_elements to build the output strings column
  return cudf::strings::detail::join_list_elements(
    cudf::lists_column_view(list_join),
    separator,
    cudf::string_scalar(""),
    cudf::strings::separator_on_nulls::NO,
    cudf::strings::output_if_empty_list::EMPTY_STRING,
    stream,
    mr);
}

}  // namespace detail

std::unique_ptr<cudf::column> byte_pair_encoding(cudf::strings_column_view const& input,
                                                 bpe_merge_pairs const& merges_table,
                                                 cudf::string_scalar const& separator,
                                                 rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::byte_pair_encoding(input, merges_table, separator, rmm::cuda_stream_default, mr);
}

}  // namespace nvtext
