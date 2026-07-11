/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "segmented_sort_keys.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace cudf {
namespace detail {

/**
 * @brief Faster segmented sorted-order for a single STRING key column via iterative radix sort
 *
 * The first pass radixes one `uint64` per element laid out `[segment : S bits][class : 1 bit when
 * nullable][prefix : P bits]` (see `packed_key_layout`) -- eight byte-passes, not the twelve a
 * 96-bit key needs. Elements still tied are compacted out and re-sorted by successive eight-byte
 * windows: each pass radixes a `prefix_key96` whose most-significant field is a dense run rank
 * encoding the order resolved so far, so a pass only reorders within a tied run; elements that
 * become singletons freeze at their final positions and drop out, and the loop exits once nothing
 * stays tied. Runs still tied after the pass cap go to one comparison cleanup. Unlike rank
 * encoding, distinct strings are never sorted, so per-pass cost does not grow with key
 * cardinality.
 *
 * Ordering equals the lexicographic comparison sort under the requested `polarity`:
 * - The prefix and the windows pack leading bytes big-endian, so unsigned key compares reproduce
 *   unsigned-byte order. `P` need not be a byte multiple: a packed-key match proves only
 *   `floor(P/8)` whole bytes equal, so the windows and the cleanup both begin there and the run
 *   rank is seeded from the packed keys, leaving `P`-bit ties in one run.
 * - A string with no byte left in a window packs to zero (the minimum), reproducing the
 *   shorter-is-less length tie-break; the residual all-zero-tail case is settled by the cleanup's
 *   length compare.
 * - The class bit sits above every prefix bit, so nulls land on the requested side with no
 *   sentinel prefix (no all-0xFF collision) and are position-final after the first pass.
 * - A descending sort XOR-complements only the byte fields (prefix and windows), leaving segment
 *   and null placement intact; the complement sends an exhausted window's zero to the maximum, so
 *   the length tie-break inverts to shorter-is-greater consistently. A zero-null column relaxes to
 *   nulls-last, keeping its keys bit-identical to the shipped ascending configuration.
 *
 * @throw cudf::logic_error If `polarity.nulls_first` is requested on a column with no null mask
 */
[[nodiscard]] std::unique_ptr<column> fast_segmented_sorted_order_strings_prefix(
  column_view const& input,
  column_view const& segment_offsets,
  sort_polarity polarity,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Whether every segment fits the graduated path's largest warp tile
 * (`STRINGS_GRAD_WARP_CAP`)
 *
 * Synchronizing device probe; the caller runs the cheap scalar gates first and guarantees at
 * least two offsets, so `segment_offsets.size() - 1` is a valid segment count.
 */
bool strings_grad_all_segments_fit(column_view const& segment_offsets,
                                   rmm::cuda_stream_view stream);

/**
 * @brief Segmented sorted-order for a STRING column via graduated in-warp sorts
 *
 * One virtual warp per segment with `cub::WarpMergeSort` under a string comparator; the warp width
 * follows the segment-size band (W8/W16/W32, two items per lane except the (0,8] slice at one) so
 * a tiny segment never occupies a full warp. Every band launches over the full segment list and
 * self-filters to its size slice; the bands partition [1, 64], so every output slot is written
 * exactly once. The caller guarantees every segment fits `STRINGS_GRAD_WARP_CAP`
 * (`strings_grad_all_segments_fit`) and offsets spanning all rows; `polarity` folds order and
 * null placement into the keys as in `fast_segmented_sorted_order_strings_prefix`.
 *
 * The cap precondition is re-verified in debug builds only, so a release-build violation is an
 * unchecked out-of-bounds hazard.
 */
[[nodiscard]] std::unique_ptr<column> fast_segmented_sorted_order_strings_grad(
  column_view const& input,
  column_view const& segment_offsets,
  sort_polarity polarity,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace cudf
