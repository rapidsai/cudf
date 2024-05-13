/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include "strings/split/split.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/algorithm.cuh>
#include <cudf/strings/detail/replace.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/replace.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/functional>
#include <thrust/binary_search.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {

/**
 * @brief Threshold to decide on using string or character-parallel functions.
 *
 * If the average byte length of a string in a column exceeds this value then
 * the character-parallel function is used.
 * Otherwise, a regular string-parallel function is used.
 *
 * This value was found using the replace-multi benchmark results using an
 * RTX A6000.
 */
constexpr size_type AVG_CHAR_BYTES_THRESHOLD = 256;

/**
 * @brief Helper functions for performing character-parallel replace
 */
struct replace_parallel_chars_fn {
  __device__ inline char const* get_base_ptr() const { return d_strings.head<char>(); }

  __device__ inline string_view const get_string(size_type idx) const
  {
    return d_strings.element<string_view>(idx);
  }

  __device__ inline bool is_valid(size_type idx) const { return d_strings.is_valid(idx); }

  /**
   * @brief Returns true if the target string is found at the given byte position
   * in the input strings column and is legally within a string row
   *
   * @param idx Index of the byte position in the chars column
   */
  __device__ bool is_target_within_row(int64_t idx) const
  {
    auto const d_offsets = d_strings_offsets;
    auto const d_chars   = get_base_ptr() + idx;
    auto const d_tgt     = d_target;
    auto const chars_end = chars_bytes + d_offsets[0];
    if (!d_tgt.empty() && (idx + d_tgt.size_bytes() <= chars_end) &&
        (d_tgt.compare(d_chars, d_tgt.size_bytes()) == 0)) {
      auto const idx_itr =
        thrust::upper_bound(thrust::seq, d_offsets, d_offsets + d_strings.size(), idx);
      auto str_idx = static_cast<size_type>(thrust::distance(d_offsets, idx_itr) - 1);
      auto d_str   = get_string(str_idx);
      if ((d_chars + d_tgt.size_bytes()) <= (d_str.data() + d_str.size_bytes())) { return true; }
    }
    return false;
  }

  /**
   * @brief Returns true if the target string found at the given byte position
   *
   * @param idx Index of the byte position in the chars column
   */
  __device__ bool has_target(int64_t idx) const
  {
    auto const d_chars = get_base_ptr() + d_strings_offsets[0] + idx;
    return (!d_target.empty() && (idx + d_target.size_bytes() <= chars_bytes) &&
            (d_target.compare(d_chars, d_target.size_bytes()) == 0));
  }

  /**
   * @brief Count the number of strings that will be produced by the replace
   *
   * This includes segments of the string that are not replaced as well as those
   * that are replaced.
   *
   * @param idx Index of the row in d_strings to be processed
   * @param d_positions Positions of the targets found in the chars column
   * @param d_targets_offsets Offsets identify which target positions go with the current string
   * @return Number of substrings resulting from the replace operations on this row
   */
  __device__ size_type count_strings(size_type idx,
                                     int64_t const* d_positions,
                                     cudf::detail::input_offsetalator d_targets_offsets) const
  {
    if (!is_valid(idx)) { return 0; }

    auto const d_str     = get_string(idx);
    auto const d_str_end = d_str.data() + d_str.size_bytes();
    auto const base_ptr  = get_base_ptr();
    auto max_n           = (maxrepl < 0) ? d_str.length() : maxrepl;

    auto const target_offset = d_targets_offsets[idx];
    auto const targets_size  = static_cast<size_type>(d_targets_offsets[idx + 1] - target_offset);
    auto const positions     = d_positions + target_offset;

    size_type count = 1;  // always at least one string
    auto str_ptr    = d_str.data();
    for (std::size_t i = 0; (i < targets_size) && (max_n > 0); ++i) {
      auto const tgt_ptr = base_ptr + positions[i];
      if (str_ptr <= tgt_ptr && tgt_ptr < d_str_end) {
        auto const keep_size = static_cast<size_type>(thrust::distance(str_ptr, tgt_ptr));
        if (keep_size > 0) { count++; }  // don't bother counting empty strings
        if (!d_replacement.empty()) { count++; }
        str_ptr += keep_size + d_target.size_bytes();
        --max_n;
      }
    }
    return count;
  }

  /**
   * @brief Retrieve the strings for each row
   *
   * This will return string segments as string_index_pair objects for
   * parts of the string that are not replaced interlaced with the
   * appropriate replacement string where replacement targets are found.
   *
   * This function is called only once to produce both the string_index_pair objects
   * and the output row size in bytes.
   *
   * @param idx Index of the row in d_strings
   * @param d_offsets Offsets to identify where to store the results of the replace for this string
   * @param d_positions The target positions found in the chars column
   * @param d_targets_offsets The offsets to identify which target positions go with this string
   * @param d_all_strings The output of all the produced string segments
   * @return The size in bytes of the output string for this row
   */
  __device__ size_type get_strings(size_type idx,
                                   cudf::detail::input_offsetalator const d_offsets,
                                   int64_t const* d_positions,
                                   cudf::detail::input_offsetalator d_targets_offsets,
                                   string_index_pair* d_all_strings) const
  {
    if (!is_valid(idx)) { return 0; }

    auto const d_output  = d_all_strings + d_offsets[idx];
    auto const d_str     = get_string(idx);
    auto const d_str_end = d_str.data() + d_str.size_bytes();
    auto const base_ptr  = get_base_ptr();
    auto max_n           = (maxrepl < 0) ? d_str.length() : maxrepl;

    auto const target_offset = d_targets_offsets[idx];
    auto const targets_size  = static_cast<size_type>(d_targets_offsets[idx + 1] - target_offset);
    auto const positions     = d_positions + target_offset;

    size_type output_idx  = 0;
    size_type output_size = 0;
    auto str_ptr          = d_str.data();
    for (std::size_t i = 0; (i < targets_size) && (max_n > 0); ++i) {
      auto const tgt_ptr = base_ptr + positions[i];
      if (str_ptr <= tgt_ptr && tgt_ptr < d_str_end) {
        auto const keep_size = static_cast<size_type>(thrust::distance(str_ptr, tgt_ptr));
        if (keep_size > 0) { d_output[output_idx++] = string_index_pair{str_ptr, keep_size}; }
        output_size += keep_size;

        if (!d_replacement.empty()) {
          d_output[output_idx++] =
            string_index_pair{d_replacement.data(), d_replacement.size_bytes()};
        }
        output_size += d_replacement.size_bytes();

        str_ptr += keep_size + d_target.size_bytes();
        --max_n;
      }
    }
    // include any leftover parts of the string
    if (str_ptr <= d_str_end) {
      auto const left_size = static_cast<size_type>(thrust::distance(str_ptr, d_str_end));
      d_output[output_idx] = string_index_pair{str_ptr, left_size};
      output_size += left_size;
    }
    return output_size;
  }

  replace_parallel_chars_fn(column_device_view const& d_strings,
                            cudf::detail::input_offsetalator d_strings_offsets,
                            int64_t chars_bytes,
                            string_view d_target,
                            string_view d_replacement,
                            cudf::size_type maxrepl)
    : d_strings(d_strings),
      d_strings_offsets(d_strings_offsets),
      chars_bytes(chars_bytes),
      d_target{d_target},
      d_replacement{d_replacement},
      maxrepl(maxrepl)
  {
  }

 protected:
  column_device_view d_strings;
  cudf::detail::input_offsetalator d_strings_offsets;
  int64_t chars_bytes;
  string_view d_target;
  string_view d_replacement;
  cudf::size_type maxrepl;
};

std::unique_ptr<column> replace_character_parallel(strings_column_view const& input,
                                                   string_view const& d_target,
                                                   string_view const& d_replacement,
                                                   cudf::size_type maxrepl,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  auto d_strings = column_device_view::create(input.parent(), stream);

  auto const strings_count = input.size();
  auto const chars_offset  = get_offset_value(input.offsets(), input.offset(), stream);
  auto const chars_bytes =
    get_offset_value(input.offsets(), input.offset() + strings_count, stream) - chars_offset;

  auto const offsets_begin =
    cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset());

  replace_parallel_chars_fn fn{
    *d_strings, offsets_begin, chars_bytes, d_target, d_replacement, maxrepl};

  // Count the number of targets in the entire column.
  // Note this may over-count in the case where a target spans adjacent strings.
  auto target_count = thrust::count_if(rmm::exec_policy_nosync(stream),
                                       thrust::make_counting_iterator<int64_t>(0),
                                       thrust::make_counting_iterator<int64_t>(chars_bytes),
                                       [fn] __device__(int64_t idx) { return fn.has_target(idx); });

  // Create a vector of every target position in the chars column.
  // These may also include overlapping targets which will be resolved later.
  auto targets_positions = rmm::device_uvector<int64_t>(target_count, stream);
  auto const copy_itr    = thrust::counting_iterator<int64_t>(chars_offset);
  auto const copy_end    = cudf::detail::copy_if_safe(
    copy_itr,
    copy_itr + chars_bytes + chars_offset,
    targets_positions.begin(),
    [fn] __device__(int64_t idx) { return fn.is_target_within_row(idx); },
    stream);

  // adjust target count since the copy-if may have eliminated some invalid targets
  target_count = std::min(std::distance(targets_positions.begin(), copy_end), target_count);
  targets_positions.resize(target_count, stream);
  auto d_positions = targets_positions.data();

  // create a vector of offsets to each string's set of target positions
  auto const targets_offsets = create_offsets_from_positions(
    input, targets_positions, stream, rmm::mr::get_current_device_resource());
  auto const d_targets_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(targets_offsets->view());

  // compute the number of string segments produced by replace in each string
  auto counts = rmm::device_uvector<size_type>(strings_count, stream);
  thrust::transform(rmm::exec_policy_nosync(stream),
                    thrust::counting_iterator<size_type>(0),
                    thrust::counting_iterator<size_type>(strings_count),
                    counts.begin(),
                    cuda::proclaim_return_type<size_type>(
                      [fn, d_positions, d_targets_offsets] __device__(size_type idx) -> size_type {
                        return fn.count_strings(idx, d_positions, d_targets_offsets);
                      }));

  // create offsets from the counts
  auto [offsets, total_strings] =
    cudf::detail::make_offsets_child_column(counts.begin(), counts.end(), stream, mr);
  auto const d_strings_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(offsets->view());

  // build a vector of all the positions for all the strings
  auto indices   = rmm::device_uvector<string_index_pair>(total_strings, stream);
  auto d_indices = indices.data();
  auto d_sizes   = counts.data();  // reusing this vector to hold output sizes now
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream),
    thrust::make_counting_iterator<size_type>(0),
    strings_count,
    [fn, d_strings_offsets, d_positions, d_targets_offsets, d_indices, d_sizes] __device__(
      size_type idx) {
      d_sizes[idx] =
        fn.get_strings(idx, d_strings_offsets, d_positions, d_targets_offsets, d_indices);
    });

  // use this utility to gather the string parts into a contiguous chars column
  auto chars      = make_strings_column(indices.begin(), indices.end(), stream, mr);
  auto chars_data = chars->release().data;

  // create offsets from the sizes
  offsets = std::get<0>(
    cudf::strings::detail::make_offsets_child_column(counts.begin(), counts.end(), stream, mr));

  // build the strings columns from the chars and offsets
  return make_strings_column(strings_count,
                             std::move(offsets),
                             std::move(chars_data.release()[0]),
                             input.null_count(),
                             cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

/**
 * @brief Function logic for the replace_string_parallel
 *
 * Performs the multi-replace operation with a thread per string.
 * This performs best on smaller strings. @see AVG_CHAR_BYTES_THRESHOLD
 */
struct replace_fn {
  column_device_view const d_strings;
  string_view d_target;
  string_view d_replacement;
  cudf::size_type maxrepl;
  cudf::size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;

  __device__ void operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) {
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }
    auto const d_str   = d_strings.element<string_view>(idx);
    char const* in_ptr = d_str.data();

    size_type bytes = d_str.size_bytes();
    size_type spos  = 0;
    size_type lpos  = 0;
    char* out_ptr   = d_chars ? d_chars + d_offsets[idx] : nullptr;
    auto max_n      = (maxrepl < 0) ? d_str.length() : maxrepl;

    // check each character against each target
    while (spos < d_str.size_bytes() && (max_n > 0)) {
      auto const d_tgt = d_target;
      if ((d_tgt.size_bytes() <= (d_str.size_bytes() - spos)) &&    // check fit
          (d_tgt.compare(in_ptr + spos, d_tgt.size_bytes()) == 0))  // and match
      {
        auto const d_repl = d_replacement;
        bytes += d_repl.size_bytes() - d_tgt.size_bytes();
        if (out_ptr) {
          out_ptr = copy_and_increment(out_ptr, in_ptr + lpos, spos - lpos);
          out_ptr = copy_string(out_ptr, d_repl);
          lpos    = spos + d_tgt.size_bytes();
        }
        spos += d_tgt.size_bytes() - 1;
        --max_n;
      }
      ++spos;
    }
    if (out_ptr) {  // copy remainder
      memcpy(out_ptr, in_ptr + lpos, d_str.size_bytes() - lpos);
    } else {
      d_sizes[idx] = bytes;
    }
  }
};

std::unique_ptr<column> replace_string_parallel(strings_column_view const& input,
                                                string_view const& d_target,
                                                string_view const& d_replacement,
                                                cudf::size_type maxrepl,
                                                rmm::cuda_stream_view stream,
                                                rmm::device_async_resource_ref mr)
{
  auto d_strings = column_device_view::create(input.parent(), stream);

  auto [offsets_column, chars] = make_strings_children(
    replace_fn{*d_strings, d_target, d_replacement, maxrepl}, input.size(), stream, mr);

  return make_strings_column(input.size(),
                             std::move(offsets_column),
                             chars.release(),
                             input.null_count(),
                             cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

}  // namespace

std::unique_ptr<column> replace(strings_column_view const& input,
                                string_scalar const& target,
                                string_scalar const& repl,
                                cudf::size_type maxrepl,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return make_empty_column(type_id::STRING); }
  if (maxrepl == 0) { return std::make_unique<cudf::column>(input.parent(), stream, mr); }
  CUDF_EXPECTS(repl.is_valid(stream), "Parameter repl must be valid.");
  CUDF_EXPECTS(target.is_valid(stream), "Parameter target must be valid.");
  CUDF_EXPECTS(target.size() > 0, "Parameter target must not be empty string.");

  string_view d_target(target.data(), target.size());
  string_view d_repl(repl.data(), repl.size());

  return (input.size() == input.null_count() ||
          ((input.chars_size(stream) / (input.size() - input.null_count())) <
           AVG_CHAR_BYTES_THRESHOLD))
           ? replace_string_parallel(input, d_target, d_repl, maxrepl, stream, mr)
           : replace_character_parallel(input, d_target, d_repl, maxrepl, stream, mr);
}

}  // namespace detail

// external API

std::unique_ptr<column> replace(strings_column_view const& strings,
                                string_scalar const& target,
                                string_scalar const& repl,
                                cudf::size_type maxrepl,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::replace(strings, target, repl, maxrepl, stream, mr);
}

}  // namespace strings
}  // namespace cudf
