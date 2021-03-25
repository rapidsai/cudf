/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/replace.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/replace.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/span.hpp>
#include <strings/utilities.cuh>
#include <strings/utilities.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/distance.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {

/**
 * @brief Average string byte-length threshold for deciding character-level vs row-level parallel
 * algorithm.
 *
 * This value was determined by running the replace string scalar benchmark against different
 * power-of-2 string lengths and observing the point at which the performance only improved for
 * all trials.
 */
constexpr size_type BYTES_PER_VALID_ROW_THRESHOLD = 64;

/**
 * @brief Function logic for the row-level parallelism replace API.
 *
 * This will perform a replace operation on each string.
 */
struct replace_row_parallel_fn {
  column_device_view const d_strings;
  string_view const d_target;
  string_view const d_repl;
  int32_t const max_repl;
  int32_t* d_offsets{};
  char* d_chars{};

  __device__ void operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) {
      if (!d_chars) d_offsets[idx] = 0;
      return;
    }
    auto const d_str   = d_strings.element<string_view>(idx);
    const char* in_ptr = d_str.data();

    char* out_ptr = d_chars ? d_chars + d_offsets[idx] : nullptr;
    auto max_n    = (max_repl < 0) ? d_str.length() : max_repl;
    auto bytes    = d_str.size_bytes();
    auto position = d_str.find(d_target);

    size_type last_pos = 0;
    while ((position >= 0) && (max_n > 0)) {
      if (out_ptr) {
        auto const curr_pos = d_str.byte_offset(position);
        out_ptr = copy_and_increment(out_ptr, in_ptr + last_pos, curr_pos - last_pos);  // copy left
        out_ptr = copy_string(out_ptr, d_repl);                                         // copy repl
        last_pos = curr_pos + d_target.size_bytes();
      } else {
        bytes += d_repl.size_bytes() - d_target.size_bytes();
      }
      position = d_str.find(d_target, position + d_target.size_bytes());
      --max_n;
    }
    if (out_ptr)  // copy whats left (or right depending on your point of view)
      memcpy(out_ptr, in_ptr + last_pos, d_str.size_bytes() - last_pos);
    else
      d_offsets[idx] = bytes;
  }
};

/**
 * @brief Functor for detecting falsely-overlapped target positions.
 *
 * This functor examines target positions that have been flagged as potentially overlapped by
 * a previous target position and identifies the overlaps that are false. A false overlap can occur
 * when a target position is overlapped by another target position that is itself overlapped.
 *
 * For example, a target string of "+++" and string to search of "++++++" will generate 4 potential
 * target positions at char offsets 0 through 3. The targets at offsets 1, 2, and 3 will be flagged
 * as potential overlaps since a prior target position is within range of the target string length.
 * The targets at offset 1 and 2 are true overlaps, since the footprint of the valid target at
 * offset 0 overlaps with them. The target at offset 3 is not truly overlapped because it is only
 * overlapped by invalid targets, targets that were themselves overlapped by a valid target.
 */
struct target_false_overlap_filter_fn {
  size_type const* const d_overlap_pos_indices{};
  size_type const* const d_target_positions{};
  size_type const target_size{};

  __device__ bool operator()(size_type overlap_idx) const
  {
    if (overlap_idx == 0) {
      // The first overlap has no prior overlap to chain, so it should be kept as an overlap.
      return false;
    }

    size_type const this_pos_idx = d_overlap_pos_indices[overlap_idx];

    // Searching backwards for the first target position index of an overlap that is not adjacent
    // to its overlap predecessor. The result will be the first overlap in this chain of overlaps.
    size_type first_overlap_idx = overlap_idx;
    size_type first_pos_idx     = this_pos_idx;
    while (first_overlap_idx > 0) {
      size_type prev_pos_idx = d_overlap_pos_indices[--first_overlap_idx];
      if (prev_pos_idx + 1 != first_pos_idx) { break; }
      first_pos_idx = prev_pos_idx;
    }

    // The prior target position to the first overlapped position in the chain is a valid target.
    size_type valid_pos_idx = first_pos_idx - 1;
    size_type valid_pos     = d_target_positions[valid_pos_idx];

    // Walk forward from this valid target. Any targets within the range of this valid one are true
    // overlaps. The first overlap beyond the range of this valid target is another valid target,
    // as it was falsely overlapped by a target that was itself overlapped. Repeat until we get to
    // the overlapped position being queried by this call.
    while (valid_pos_idx < this_pos_idx) {
      size_type next_pos_idx = valid_pos_idx + 1;
      size_type next_pos     = d_target_positions[next_pos_idx];
      // Every target position within the range of a valid target position is a true overlap.
      while (next_pos < valid_pos + target_size) {
        if (next_pos_idx == this_pos_idx) { return false; }
        next_pos = d_target_positions[++next_pos_idx];
      }
      valid_pos_idx = next_pos_idx;
      valid_pos     = next_pos;
    }

    // This was overlapped only by false overlaps and therefore is a valid target.
    return true;
  }
};

/**
 * @brief Functor for replacing each target string with the replacement string.
 *
 * This will perform a replace operation at each target position.
 */
struct target_replacer_fn {
  device_span<size_type const> const d_target_positions;
  char const* const d_in_chars{};
  char* const d_out_chars{};
  size_type const target_size{};
  string_view const d_repl;
  int32_t const in_char_offset = 0;

  __device__ void operator()(size_type input_idx) const
  {
    // Calculate the adjustment from input index to output index for each prior target position.
    auto const repl_size         = d_repl.size_bytes();
    auto const idx_delta_per_pos = repl_size - target_size;

    // determine the number of target positions at or before this character position
    size_type const* next_target_pos_ptr = thrust::upper_bound(
      thrust::seq, d_target_positions.begin(), d_target_positions.end(), input_idx);
    size_type const num_prev_targets = next_target_pos_ptr - d_target_positions.data();
    size_type output_idx = input_idx - in_char_offset + idx_delta_per_pos * num_prev_targets;

    if (num_prev_targets == 0) {
      // not within a target string
      d_out_chars[output_idx] = d_in_chars[input_idx];
    } else {
      // check if this input position is within a target string
      size_type const prev_target_pos = *(next_target_pos_ptr - 1);
      size_type target_idx            = input_idx - prev_target_pos;
      if (target_idx < target_size) {
        // within the target string, so the original calculation was off by one target string
        output_idx -= idx_delta_per_pos;

        // Copy the corresponding byte from the replacement string. If the replacement string is
        // larger than the target string then the thread reading the last target byte is
        // responsible for copying the remainder of the replacement string.
        if (target_idx < repl_size) {
          d_out_chars[output_idx++] = d_repl.data()[target_idx++];
          if (target_idx == target_size) {
            memcpy(d_out_chars + output_idx, d_repl.data() + target_idx, repl_size - target_idx);
          }
        }
      } else {
        // not within a target string
        d_out_chars[output_idx] = d_in_chars[input_idx];
      }
    }
  }
};

/**
 * @brief Filter target positions that are overlapped by other, valid target positions.
 *
 * This performs an in-place modification of the target positions to remove any target positions
 * that are overlapped by other, valid target positions. For example, if the target string is "++"
 * and the string to search is "+++" then there will be two potential targets at character offsets
 * 0 and 1. The target at offset 0 is valid and overlaps the target at offset 1, invalidating the
 * target at offset 1.
 *
 * @param[in,out] d_target_positions Potential target positions to filter in-place.
 * @param[in]     target_count       Number of potential target positions.
 * @param[in]     target_size        Size of the target string in bytes.
 * @param[in]     stream             CUDA stream to use for device operations.
 * @return Number of target positions after filtering.
 */
size_type filter_overlap_target_positions(size_type* d_target_positions,
                                          size_type target_count,
                                          size_type target_size,
                                          rmm::cuda_stream_view stream)
{
  auto overlap_detector = [d_target_positions, target_size] __device__(size_type pos_idx) -> bool {
    return (pos_idx > 0)
             ? d_target_positions[pos_idx] - d_target_positions[pos_idx - 1] < target_size
             : false;
  };

  // count the potential number of overlapped target positions
  size_type overlap_count =
    thrust::count_if(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<size_type>(0),
                     thrust::make_counting_iterator<size_type>(target_count),
                     overlap_detector);
  if (overlap_count == 0) { return target_count; }

  // create a vector indexing the potential overlapped target positions
  rmm::device_uvector<size_type> potential_overlapped_pos_indices(overlap_count, stream);
  auto d_potential_overlapped_pos_indices = potential_overlapped_pos_indices.data();
  thrust::copy_if(rmm::exec_policy(stream),
                  thrust::make_counting_iterator<size_type>(0),
                  thrust::make_counting_iterator<size_type>(target_count),
                  d_potential_overlapped_pos_indices,
                  overlap_detector);

  // filter out the false overlaps that are actually valid
  rmm::device_uvector<size_type> overlapped_pos_indices(overlap_count, stream);
  auto d_overlapped_pos_indices = overlapped_pos_indices.data();
  auto overlap_end =
    thrust::remove_copy_if(rmm::exec_policy(stream),
                           d_potential_overlapped_pos_indices,
                           d_potential_overlapped_pos_indices + overlap_count,
                           thrust::make_counting_iterator<size_type>(0),
                           d_overlapped_pos_indices,
                           target_false_overlap_filter_fn{
                             d_potential_overlapped_pos_indices, d_target_positions, target_size});
  overlap_count = cudf::distance(d_overlapped_pos_indices, overlap_end);

  // In-place remove any target positions that are overlapped by valid target positions
  auto target_pos_end = thrust::remove_if(
    rmm::exec_policy(stream),
    d_target_positions,
    d_target_positions + target_count,
    thrust::make_counting_iterator<size_type>(0),
    [d_overlapped_pos_indices, overlap_count] __device__(size_type target_position_idx) -> bool {
      return thrust::binary_search(thrust::seq,
                                   d_overlapped_pos_indices,
                                   d_overlapped_pos_indices + overlap_count,
                                   target_position_idx);
    });
  return cudf::distance(d_target_positions, target_pos_end);
}

/**
 * @brief Filter target positions to remove any invalid target positions.
 *
 * This performs an in-place modification of the target positions to remove any target positions
 * that are invalid, either by the target string overlapping a row boundary or being overlapped by
 * another valid target string.
 *
 * @param[in,out] target_positions Potential target positions to filter in-place.
 * @param[in]     d_offsets_span   Memory range encompassing the string column offsets.
 * @param[in]     target_size      Size of the target string in bytes.
 * @param[in]     stream           CUDA stream to use for device operations.
 * @return Number of target positions after filtering.
 */
size_type filter_false_target_positions(rmm::device_uvector<size_type>& target_positions,
                                        device_span<int32_t const> d_offsets_span,
                                        size_type target_size,
                                        rmm::cuda_stream_view stream)
{
  // In-place remove any positions for target strings that crossed string boundaries.
  auto d_target_positions = target_positions.data();
  auto target_pos_end =
    thrust::remove_if(rmm::exec_policy(stream),
                      d_target_positions,
                      d_target_positions + target_positions.size(),
                      [d_offsets_span, target_size] __device__(size_type target_pos) -> bool {
                        // find the end of the string containing the start of this target
                        size_type const* offset_ptr = thrust::upper_bound(
                          thrust::seq, d_offsets_span.begin(), d_offsets_span.end(), target_pos);
                        return target_pos + target_size > *offset_ptr;
                      });
  auto const target_count = cudf::distance(d_target_positions, target_pos_end);
  if (target_count == 0) { return 0; }

  // Filter out target positions that are the result of overlapping target matches.
  return (target_count > 1)
           ? filter_overlap_target_positions(d_target_positions, target_count, target_size, stream)
           : target_count;
}

/**
 * @brief Filter target positions beyond the maximum target replacements per row limit.
 *
 * This performs an in-place modification of the target positions to remove any target positions
 * corresponding to targets that should not be replaced due to the maximum target replacement per
 * row limit.
 *
 * @param[in,out] target_positions Target positions to filter in-place.
 * @param[in]     target_count     Number of target positions.
 * @param[in]     d_offsets_span   Memory range encompassing the string column offsets.
 * @param[in]     max_repl_per_row Maximum target replacements per row limit.
 * @param[in]     stream           CUDA stream to use for device operations.
 * @return Number of target positions after filtering.
 */
size_type filter_maxrepl_target_positions(size_type* d_target_positions,
                                          size_type target_count,
                                          device_span<int32_t const> d_offsets_span,
                                          size_type max_repl_per_row,
                                          rmm::cuda_stream_view stream)
{
  auto pos_to_row_fn = [d_offsets_span] __device__(size_type target_pos) -> size_type {
    auto upper_bound =
      thrust::upper_bound(thrust::seq, d_offsets_span.begin(), d_offsets_span.end(), target_pos);
    return thrust::distance(d_offsets_span.begin(), upper_bound);
  };

  // compute the match count per row for each target position
  rmm::device_uvector<size_type> match_counts(target_count, stream);
  auto d_match_counts = match_counts.data();
  thrust::inclusive_scan_by_key(
    rmm::exec_policy(stream),
    thrust::make_transform_iterator(d_target_positions, pos_to_row_fn),
    thrust::make_transform_iterator(d_target_positions + target_count, pos_to_row_fn),
    thrust::make_constant_iterator<size_type>(1),
    d_match_counts);

  // In-place remove any positions that exceed the per-row match limit
  auto target_pos_end =
    thrust::remove_if(rmm::exec_policy(stream),
                      d_target_positions,
                      d_target_positions + target_count,
                      d_match_counts,
                      [max_repl_per_row] __device__(size_type match_count) -> bool {
                        return match_count > max_repl_per_row;
                      });

  return cudf::distance(d_target_positions, target_pos_end);
}

/**
 * @brief Scalar string replacement using a character-level parallel algorithm.
 *
 * Replaces occurrences of the target string with the replacement string using an algorithm with
 * character-level parallelism. This algorithm will perform well when the strings in the string
 * column are relatively long.
 * @see BYTES_PER_VALID_ROW_THRESHOLD
 *
 * @param strings     String column to search for target strings.
 * @param chars_start Offset of the first character in the string column.
 * @param chars_end   Offset beyond the last character in the string column to search.
 * @param d_target    String to search for within the string column.
 * @param d_repl      Replacement string if target string is found.
 * @param maxrepl     Maximum times to replace if target appears multiple times in a string.
 * @param stream      CUDA stream to use for device operations
 * @param mr          Device memory resource used to allocate the returned column's device memory
 * @return New strings column.
 */
std::unique_ptr<column> replace_char_parallel(strings_column_view const& strings,
                                              size_type chars_start,
                                              size_type chars_end,
                                              string_view const& d_target,
                                              string_view const& d_repl,
                                              int32_t maxrepl,
                                              rmm::cuda_stream_view stream,
                                              rmm::mr::device_memory_resource* mr)
{
  auto const strings_count = strings.size();
  auto const offset_count  = strings_count + 1;
  auto const d_offsets     = strings.offsets().data<int32_t>() + strings.offset();
  auto const d_in_chars    = strings.chars().data<char>();
  auto const chars_bytes   = chars_end - chars_start;
  auto const target_size   = d_target.size_bytes();

  // detect a target match at the specified byte position
  device_span<char const> const d_chars_span(d_in_chars, chars_end);
  auto target_detector = [d_chars_span, d_target] __device__(size_type char_idx) {
    auto target_size = d_target.size_bytes();
    auto target_ptr  = d_chars_span.begin() + char_idx;
    return target_ptr + target_size <= d_chars_span.end() &&
           d_target.compare(target_ptr, target_size) == 0;
  };

  // Count target string matches across all character positions, ignoring string boundaries and
  // overlapping target strings. This may produce false-positives.
  size_type target_count = thrust::count_if(rmm::exec_policy(stream),
                                            thrust::make_counting_iterator<size_type>(chars_start),
                                            thrust::make_counting_iterator<size_type>(chars_end),
                                            target_detector);
  if (target_count == 0) {
    // nothing to replace, copy the input column
    return std::make_unique<cudf::column>(strings.parent(), stream, mr);
  }

  // create a vector of the potential target match positions
  rmm::device_uvector<size_type> target_positions(target_count, stream);
  auto d_target_positions = target_positions.data();
  thrust::copy_if(rmm::exec_policy(stream),
                  thrust::make_counting_iterator<size_type>(chars_start),
                  thrust::make_counting_iterator<size_type>(chars_end),
                  d_target_positions,
                  target_detector);

  device_span<int32_t const> d_offsets_span(d_offsets, offset_count);
  if (target_size > 1) {
    target_count =
      filter_false_target_positions(target_positions, d_offsets_span, target_size, stream);
    if (target_count == 0) {
      // nothing to replace, copy the input column
      return std::make_unique<cudf::column>(strings.parent(), stream, mr);
    }
  }

  // filter out any target positions that exceed the per-row match limit
  if (maxrepl > 0 && target_count > maxrepl) {
    target_count = filter_maxrepl_target_positions(
      d_target_positions, target_count, d_offsets_span, maxrepl, stream);
  }

  // build the offsets column
  auto offsets_column = make_numeric_column(
    data_type{type_id::INT32}, offset_count, mask_state::UNALLOCATED, stream, mr);
  auto offsets_view     = offsets_column->mutable_view();
  auto delta_per_target = d_repl.size_bytes() - target_size;
  device_span<size_type const> d_target_positions_span(d_target_positions, target_count);
  auto offsets_update_fn =
    [d_target_positions_span, delta_per_target, chars_start] __device__(int32_t offset) -> int32_t {
    // determine the number of target positions occurring before this offset
    size_type const* next_target_pos_ptr = thrust::lower_bound(
      thrust::seq, d_target_positions_span.begin(), d_target_positions_span.end(), offset);
    size_type num_prev_targets =
      thrust::distance(d_target_positions_span.data(), next_target_pos_ptr);
    return offset - chars_start + delta_per_target * num_prev_targets;
  };
  thrust::transform(rmm::exec_policy(stream),
                    d_offsets_span.begin(),
                    d_offsets_span.end(),
                    offsets_view.begin<int32_t>(),
                    offsets_update_fn);

  // build the characters column
  auto chars_column = create_chars_child_column(strings_count,
                                                strings.null_count(),
                                                chars_bytes + (delta_per_target * target_count),
                                                stream,
                                                mr);
  auto d_out_chars  = chars_column->mutable_view().data<char>();
  thrust::for_each_n(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<size_type>(chars_start),
    chars_bytes,
    target_replacer_fn{
      d_target_positions_span, d_in_chars, d_out_chars, target_size, d_repl, chars_start});

  // free the target positions buffer as it is no longer needed
  (void)target_positions.release();

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             std::move(chars_column),
                             strings.null_count(),
                             cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                             stream,
                             mr);
}

/**
 * @brief Scalar string replacement using a row-level parallel algorithm.
 *
 * Replaces occurrences of the target string with the replacement string using an algorithm with
 * row-level parallelism. This algorithm will perform well when the strings in the string
 * column are relatively short.
 * @see BYTES_PER_VALID_ROW_THRESHOLD
 *
 * @param strings     String column to search for target strings.
 * @param d_target    String to search for within the string column.
 * @param d_repl      Replacement string if target string is found.
 * @param maxrepl     Maximum times to replace if target appears multiple times in a string.
 * @param stream      CUDA stream to use for device operations
 * @param mr          Device memory resource used to allocate the returned column's device memory
 * @return New strings column.
 */
std::unique_ptr<column> replace_row_parallel(strings_column_view const& strings,
                                             string_view const& d_target,
                                             string_view const& d_repl,
                                             int32_t maxrepl,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr)
{
  auto d_strings = column_device_view::create(strings.parent(), stream);

  // this utility calls the given functor to build the offsets and chars columns
  auto children = cudf::strings::detail::make_strings_children(
    replace_row_parallel_fn{*d_strings, d_target, d_repl, maxrepl},
    strings.size(),
    strings.null_count(),
    stream,
    mr);

  return make_strings_column(strings.size(),
                             std::move(children.first),
                             std::move(children.second),
                             strings.null_count(),
                             cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                             stream,
                             mr);
}

}  // namespace

/**
 * @copydoc cudf::strings::detail::replace(strings_column_view const&, string_scalar const&,
 * string_scalar const&, int32_t, rmm::cuda_stream_view, rmm::mr::device_memory_resource*)
 */
template <>
std::unique_ptr<column> replace<replace_algorithm::AUTO>(strings_column_view const& strings,
                                                         string_scalar const& target,
                                                         string_scalar const& repl,
                                                         int32_t maxrepl,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::mr::device_memory_resource* mr)
{
  if (strings.is_empty()) return make_empty_strings_column(stream, mr);
  if (maxrepl == 0) return std::make_unique<cudf::column>(strings.parent(), stream, mr);
  CUDF_EXPECTS(repl.is_valid(), "Parameter repl must be valid.");
  CUDF_EXPECTS(target.is_valid(), "Parameter target must be valid.");
  CUDF_EXPECTS(target.size() > 0, "Parameter target must not be empty string.");

  string_view d_target(target.data(), target.size());
  string_view d_repl(repl.data(), repl.size());

  // determine range of characters in the base column
  auto const strings_count = strings.size();
  auto const offset_count  = strings_count + 1;
  auto const d_offsets     = strings.offsets().data<int32_t>() + strings.offset();
  size_type const chars_start =
    (strings.offset() == 0)
      ? 0
      : cudf::detail::get_value<int32_t>(strings.offsets(), strings.offset(), stream);
  size_type const chars_end = (offset_count == strings.offsets().size())
                                ? strings.chars_size()
                                : cudf::detail::get_value<int32_t>(
                                    strings.offsets(), strings.offset() + strings_count, stream);
  size_type const chars_bytes = chars_end - chars_start;

  auto const avg_bytes_per_row = chars_bytes / std::max(strings_count - strings.null_count(), 1);
  return (avg_bytes_per_row < BYTES_PER_VALID_ROW_THRESHOLD)
           ? replace_row_parallel(strings, d_target, d_repl, maxrepl, stream, mr)
           : replace_char_parallel(
               strings, chars_start, chars_end, d_target, d_repl, maxrepl, stream, mr);
}

template <>
std::unique_ptr<column> replace<replace_algorithm::CHAR_PARALLEL>(
  strings_column_view const& strings,
  string_scalar const& target,
  string_scalar const& repl,
  int32_t maxrepl,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  if (strings.is_empty()) return make_empty_strings_column(stream, mr);
  if (maxrepl == 0) return std::make_unique<cudf::column>(strings.parent(), stream, mr);
  CUDF_EXPECTS(repl.is_valid(), "Parameter repl must be valid.");
  CUDF_EXPECTS(target.is_valid(), "Parameter target must be valid.");
  CUDF_EXPECTS(target.size() > 0, "Parameter target must not be empty string.");

  string_view d_target(target.data(), target.size());
  string_view d_repl(repl.data(), repl.size());

  // determine range of characters in the base column
  auto const strings_count = strings.size();
  auto const offset_count  = strings_count + 1;
  auto const d_offsets     = strings.offsets().data<int32_t>() + strings.offset();
  size_type chars_start    = (strings.offset() == 0) ? 0
                                                  : cudf::detail::get_value<int32_t>(
                                                      strings.offsets(), strings.offset(), stream);
  size_type chars_end = (offset_count == strings.offsets().size())
                          ? strings.chars_size()
                          : cudf::detail::get_value<int32_t>(
                              strings.offsets(), strings.offset() + strings_count, stream);
  return replace_char_parallel(
    strings, chars_start, chars_end, d_target, d_repl, maxrepl, stream, mr);
}

template <>
std::unique_ptr<column> replace<replace_algorithm::ROW_PARALLEL>(
  strings_column_view const& strings,
  string_scalar const& target,
  string_scalar const& repl,
  int32_t maxrepl,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr)
{
  if (strings.is_empty()) return make_empty_strings_column(stream, mr);
  if (maxrepl == 0) return std::make_unique<cudf::column>(strings.parent(), stream, mr);
  CUDF_EXPECTS(repl.is_valid(), "Parameter repl must be valid.");
  CUDF_EXPECTS(target.is_valid(), "Parameter target must be valid.");
  CUDF_EXPECTS(target.size() > 0, "Parameter target must not be empty string.");

  string_view d_target(target.data(), target.size());
  string_view d_repl(repl.data(), repl.size());
  return replace_row_parallel(strings, d_target, d_repl, maxrepl, stream, mr);
}

namespace {
/**
 * @brief Function logic for the replace_slice API.
 *
 * This will perform a replace_slice operation on each string.
 */
struct replace_slice_fn {
  column_device_view const d_strings;
  string_view const d_repl;
  size_type const start;
  size_type const stop;
  int32_t* d_offsets{};
  char* d_chars{};

  __device__ void operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) {
      if (!d_chars) d_offsets[idx] = 0;
      return;
    }
    auto const d_str   = d_strings.element<string_view>(idx);
    auto const length  = d_str.length();
    char const* in_ptr = d_str.data();
    auto const begin   = d_str.byte_offset(((start < 0) || (start > length) ? length : start));
    auto const end     = d_str.byte_offset(((stop < 0) || (stop > length) ? length : stop));

    if (d_chars) {
      char* out_ptr = d_chars + d_offsets[idx];

      out_ptr = copy_and_increment(out_ptr, in_ptr, begin);  // copy beginning
      out_ptr = copy_string(out_ptr, d_repl);                // insert replacement
      out_ptr = copy_and_increment(out_ptr,                  // copy end
                                   in_ptr + end,
                                   d_str.size_bytes() - end);
    } else {
      d_offsets[idx] = d_str.size_bytes() + d_repl.size_bytes() - (end - begin);
    }
  }
};

}  // namespace

std::unique_ptr<column> replace_slice(strings_column_view const& strings,
                                      string_scalar const& repl,
                                      size_type start,
                                      size_type stop,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{
  if (strings.is_empty()) return make_empty_strings_column(stream, mr);
  CUDF_EXPECTS(repl.is_valid(), "Parameter repl must be valid.");
  if (stop > 0) CUDF_EXPECTS(start <= stop, "Parameter start must be less than or equal to stop.");

  string_view d_repl(repl.data(), repl.size());

  auto d_strings = column_device_view::create(strings.parent(), stream);

  // this utility calls the given functor to build the offsets and chars columns
  auto children =
    cudf::strings::detail::make_strings_children(replace_slice_fn{*d_strings, d_repl, start, stop},
                                                 strings.size(),
                                                 strings.null_count(),
                                                 stream,
                                                 mr);

  return make_strings_column(strings.size(),
                             std::move(children.first),
                             std::move(children.second),
                             strings.null_count(),
                             cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                             stream,
                             mr);
}

namespace {
/**
 * @brief Function logic for the replace_multi API.
 *
 * This will perform the multi-replace operation on each string.
 */
struct replace_multi_fn {
  column_device_view const d_strings;
  column_device_view const d_targets;
  column_device_view const d_repls;
  int32_t* d_offsets{};
  char* d_chars{};

  __device__ void operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) {
      if (!d_chars) d_offsets[idx] = 0;
      return;
    }
    auto const d_str   = d_strings.element<string_view>(idx);
    char const* in_ptr = d_str.data();

    size_type bytes = d_str.size_bytes();
    size_type spos  = 0;
    size_type lpos  = 0;
    char* out_ptr   = d_chars ? d_chars + d_offsets[idx] : nullptr;

    // check each character against each target
    while (spos < d_str.size_bytes()) {
      for (int tgt_idx = 0; tgt_idx < d_targets.size(); ++tgt_idx) {
        auto const d_tgt = d_targets.element<string_view>(tgt_idx);
        if ((d_tgt.size_bytes() <= (d_str.size_bytes() - spos)) &&    // check fit
            (d_tgt.compare(in_ptr + spos, d_tgt.size_bytes()) == 0))  // and match
        {
          auto const d_repl = (d_repls.size() == 1) ? d_repls.element<string_view>(0)
                                                    : d_repls.element<string_view>(tgt_idx);
          bytes += d_repl.size_bytes() - d_tgt.size_bytes();
          if (out_ptr) {
            out_ptr = copy_and_increment(out_ptr, in_ptr + lpos, spos - lpos);
            out_ptr = copy_string(out_ptr, d_repl);
            lpos    = spos + d_tgt.size_bytes();
          }
          spos += d_tgt.size_bytes() - 1;
          break;
        }
      }
      ++spos;
    }
    if (out_ptr)  // copy remainder
      memcpy(out_ptr, in_ptr + lpos, d_str.size_bytes() - lpos);
    else
      d_offsets[idx] = bytes;
  }
};

}  // namespace

std::unique_ptr<column> replace(strings_column_view const& strings,
                                strings_column_view const& targets,
                                strings_column_view const& repls,
                                rmm::cuda_stream_view stream,
                                rmm::mr::device_memory_resource* mr)
{
  if (strings.is_empty()) return make_empty_strings_column(stream, mr);
  CUDF_EXPECTS(((targets.size() > 0) && (targets.null_count() == 0)),
               "Parameters targets must not be empty and must not have nulls");
  CUDF_EXPECTS(((repls.size() > 0) && (repls.null_count() == 0)),
               "Parameters repls must not be empty and must not have nulls");
  if (repls.size() > 1)
    CUDF_EXPECTS(repls.size() == targets.size(), "Sizes for targets and repls must match");

  auto d_strings = column_device_view::create(strings.parent(), stream);
  auto d_targets = column_device_view::create(targets.parent(), stream);
  auto d_repls   = column_device_view::create(repls.parent(), stream);

  // this utility calls the given functor to build the offsets and chars columns
  auto children =
    cudf::strings::detail::make_strings_children(replace_multi_fn{*d_strings, *d_targets, *d_repls},
                                                 strings.size(),
                                                 strings.null_count(),
                                                 stream,
                                                 mr);

  return make_strings_column(strings.size(),
                             std::move(children.first),
                             std::move(children.second),
                             strings.null_count(),
                             cudf::detail::copy_bitmask(strings.parent(), stream, mr),
                             stream,
                             mr);
}

std::unique_ptr<column> replace_nulls(strings_column_view const& strings,
                                      string_scalar const& repl,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr)
{
  size_type strings_count = strings.size();
  if (strings_count == 0) return make_empty_strings_column(stream, mr);
  CUDF_EXPECTS(repl.is_valid(), "Parameter repl must be valid.");

  string_view d_repl(repl.data(), repl.size());

  auto strings_column = column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;

  // build offsets column
  auto offsets_transformer_itr = thrust::make_transform_iterator(
    thrust::make_counting_iterator<int32_t>(0), [d_strings, d_repl] __device__(size_type idx) {
      return d_strings.is_null(idx) ? d_repl.size_bytes()
                                    : d_strings.element<string_view>(idx).size_bytes();
    });
  auto offsets_column = make_offsets_child_column(
    offsets_transformer_itr, offsets_transformer_itr + strings_count, stream, mr);
  auto d_offsets = offsets_column->view().data<int32_t>();

  // build chars column
  size_type bytes   = thrust::device_pointer_cast(d_offsets)[strings_count];
  auto chars_column = strings::detail::create_chars_child_column(
    strings_count, strings.null_count(), bytes, stream, mr);
  auto d_chars = chars_column->mutable_view().data<char>();
  thrust::for_each_n(rmm::exec_policy(stream),
                     thrust::make_counting_iterator<size_type>(0),
                     strings_count,
                     [d_strings, d_repl, d_offsets, d_chars] __device__(size_type idx) {
                       string_view d_str = d_repl;
                       if (!d_strings.is_null(idx)) d_str = d_strings.element<string_view>(idx);
                       memcpy(d_chars + d_offsets[idx], d_str.data(), d_str.size_bytes());
                     });

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             std::move(chars_column),
                             0,
                             rmm::device_buffer{0, stream, mr},
                             stream,
                             mr);
}

}  // namespace detail

// external API

std::unique_ptr<column> replace(strings_column_view const& strings,
                                string_scalar const& target,
                                string_scalar const& repl,
                                int32_t maxrepl,
                                rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::replace(strings, target, repl, maxrepl, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> replace_slice(strings_column_view const& strings,
                                      string_scalar const& repl,
                                      size_type start,
                                      size_type stop,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::replace_slice(strings, repl, start, stop, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> replace(strings_column_view const& strings,
                                strings_column_view const& targets,
                                strings_column_view const& repls,
                                rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::replace(strings, targets, repls, rmm::cuda_stream_default, mr);
}

std::unique_ptr<column> replace_nulls(strings_column_view const& strings,
                                      string_scalar const& repl,
                                      rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::replace_nulls(strings, repl, rmm::cuda_stream_default, mr);
}

}  // namespace strings
}  // namespace cudf
