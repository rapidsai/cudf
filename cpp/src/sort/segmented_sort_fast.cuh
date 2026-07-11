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

}  // namespace detail
}  // namespace cudf
