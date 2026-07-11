/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cub/device/device_select.cuh>
#include <cuda/std/cstdint>
#include <cuda/std/tuple>

namespace cudf {
namespace detail {

/**
 * @brief Runtime key polarity realizing one explicit (order, null_order) on the segmented-sort
 * fast paths
 *
 * Every engine orders by an unsigned comparison of a packed key, so the requested configuration is
 * folded into the key bits -- one XOR/class operation per element -- rather than into per-engine
 * comparators, which would multiply kernel instantiations. `descending` complements the encoded
 * value field: an order-reversing bijection confined to the field's exact width, leaving the
 * class/segment bits above it untouched. `nulls_first` picks the null class bit. The only current
 * caller constructs the default `{false, false}` state, which reproduces the shipped ascending /
 * nulls-last keys bit for bit; the other states engage once the gate widens to the full matrix.
 */
struct sort_polarity {
  bool descending  = false;
  bool nulls_first = false;

  /// Class bit for a valid (0/1) or null (1/0) element: unsigned key order then places nulls on
  /// the requested side of every valid element.
  __host__ __device__ cuda::std::uint32_t element_class(bool is_null) const
  {
    return static_cast<cuda::std::uint32_t>(is_null != nulls_first);
  }
  /// XOR mask reversing a 32-bit encoded value's unsigned order when descending.
  __host__ __device__ cuda::std::uint32_t value_mask32() const
  {
    return descending ? ~cuda::std::uint32_t{0} : cuda::std::uint32_t{0};
  }
  /// XOR mask reversing a 64-bit encoded value's unsigned order when descending.
  __host__ __device__ cuda::std::uint64_t value_mask64() const
  {
    return descending ? ~cuda::std::uint64_t{0} : cuda::std::uint64_t{0};
  }
};

/**
 * @brief Fixed-width radix key ordering runs still tied after the single-`uint64` first pass
 *
 * Twelve bytes with no padding decompose to exactly 96 radix bits (12 passes) while carrying a
 * full eight-byte window; key bytes drive this sort's cost, so shedding a 16-byte key's four pad
 * bytes is a direct data-movement win. Fields are most- to least-significant. `seg_null` holds
 * `(rank << 1) | is_null` -- the dense run rank dominates, so a pass preserves every order an
 * earlier pass resolved and reorders only within a tied run, and within a rank a null sorts after
 * every non-null. `prefix_hi`/`prefix_lo` hold the element's next eight window bytes packed
 * big-endian (byte 0 most significant, exhausted bytes zero-filled), so comparing them in order
 * reproduces unsigned-byte lexicographic order; a null's window words are zero, never its string
 * bytes.
 */
struct prefix_key96 {
  cuda::std::uint32_t seg_null;
  cuda::std::uint32_t prefix_hi;
  cuda::std::uint32_t prefix_lo;
};

static_assert(sizeof(prefix_key96) == 12 and alignof(prefix_key96) == 4,
              "prefix_key96 must stay 12 bytes with 4-byte alignment");

/// Decomposes a `prefix_key96` for `cub::DeviceRadixSort`; the leftmost tuple element is the most
/// significant, ordering by run rank and null flag, then the window words.
struct prefix_decomposer {
  __device__ cuda::std::tuple<cuda::std::uint32_t&, cuda::std::uint32_t&, cuda::std::uint32_t&>
  operator()(prefix_key96& key) const
  {
    return {key.seg_null, key.prefix_hi, key.prefix_lo};
  }
};

/// True when two keys are bit-for-bit equal, i.e. in the same run of an unresolved tie.
__device__ inline bool keys_equal(prefix_key96 const& a, prefix_key96 const& b)
{
  return a.seg_null == b.seg_null && a.prefix_hi == b.prefix_hi && a.prefix_lo == b.prefix_lo;
}

/**
 * @brief Packs a dense run rank (bits 1..31) and a null flag (bit 0) into one 32-bit field
 *
 * The rank comes from an inclusive scan over run-head flags, so it is bounded by the element count
 * (a `size_type`, <= 2^31 - 1) and `(rank << 1) | flag` never overflows; a `static_assert` in the
 * caller proves it.
 */
__device__ inline cuda::std::uint32_t pack_seg_null(cuda::std::uint32_t label,
                                                    cuda::std::uint32_t flag)
{
  return (label << 1) | flag;
}

/// Splits a big-endian-packed eight-byte window into a `prefix_key96`'s two window words.
__device__ inline void split_prefix(cuda::std::uint64_t packed,
                                    cuda::std::uint32_t& prefix_hi,
                                    cuda::std::uint32_t& prefix_lo)
{
  prefix_hi = static_cast<cuda::std::uint32_t>(packed >> 32);
  prefix_lo = static_cast<cuda::std::uint32_t>(packed & 0xFFFF'FFFFu);
}

/**
 * @brief Flags the first position of each maximal run of equal keys (1 = head, 0 = continuation)
 *
 * An inclusive sum over these flags yields a dense one-based run rank shared exactly by equal
 * keys; window keys carry the prior pass's rank in their leading field, so each new rank refines
 * the old grouping without merging elements an earlier pass separated.
 */
struct key_head_flag {
  prefix_key96 const* d_keys;
  __device__ cuda::std::uint32_t operator()(size_type i) const
  {
    if (i == 0) { return 1u; }
    return keys_equal(d_keys[i - 1], d_keys[i]) ? 0u : 1u;
  }
};

/**
 * @brief Runs `cub::DeviceSelect::Flagged` to completion: the temp-storage query pass, then the
 * execute pass
 *
 * Shared by every call site that compacts positions down to their `tied_flags`-true subset -- the
 * strings prefix-radix setup and its per-pass loop repeat this exact two-call CUB idiom (size the
 * scratch buffer, then execute) with only the iterator types and the element count differing.
 */
template <typename InputIteratorT, typename OutputIteratorT>
void cub_select_flagged(InputIteratorT d_in,
                        bool const* tied_flags,
                        OutputIteratorT d_out,
                        size_type* d_num_selected,
                        size_type count,
                        rmm::cuda_stream_view stream)
{
  rmm::device_buffer d_temp_storage;
  size_t temp_storage_bytes = 0;
  cub::DeviceSelect::Flagged(d_temp_storage.data(),
                             temp_storage_bytes,
                             d_in,
                             tied_flags,
                             d_out,
                             d_num_selected,
                             count,
                             stream.value());
  d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
  cub::DeviceSelect::Flagged(d_temp_storage.data(),
                             temp_storage_bytes,
                             d_in,
                             tied_flags,
                             d_out,
                             d_num_selected,
                             count,
                             stream.value());
}

}  // namespace detail
}  // namespace cudf
