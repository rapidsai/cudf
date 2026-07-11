/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "segmented_sort_fast.cuh"
#include "segmented_sort_keys.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/labeling/label_segments.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <cuda/std/bit>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <thrust/sequence.h>
#include <thrust/transform.h>

namespace cudf {
namespace detail {
namespace {

/**
 * @brief Maps a fixed-width value to the unsigned key whose ascending order equals value order
 *
 * Signed integrals flip the sign bit; chrono types encode their integer rep; floating point takes
 * the order-preserving IEEE-754 flip with every NaN mapped to all-ones (NaNs tie, sorting after
 * +Inf). The transform runs at the input width; the caller's zero-extension adds equal high bits
 * that cannot reorder. Non-integral branches resolve first: `make_unsigned_t` is ill-formed for
 * floating-point and chrono types.
 */
template <typename T>
__device__ inline cuda::std::uint32_t radix_encode_u32(T value)
{
  if constexpr (cuda::std::is_same_v<T, bool>) {
    return value ? cuda::std::uint32_t{1} : cuda::std::uint32_t{0};
  } else if constexpr (cudf::is_timestamp<T>()) {
    return radix_encode_u32(value.time_since_epoch().count());
  } else if constexpr (cudf::is_duration<T>()) {
    return radix_encode_u32(value.count());
  } else if constexpr (cuda::std::is_floating_point_v<T>) {
    if (cuda::std::isnan(value)) { return ~cuda::std::uint32_t{0}; }
    auto const bits          = cuda::std::bit_cast<cuda::std::uint32_t>(value);
    auto constexpr sign_mask = cuda::std::uint32_t{1} << (sizeof(cuda::std::uint32_t) * 8 - 1);
    return (bits & sign_mask) ? ~bits : (bits | sign_mask);
  } else {
    using U      = cuda::std::make_unsigned_t<T>;
    auto encoded = static_cast<U>(value);
    if constexpr (cuda::std::is_signed_v<T>) {
      encoded ^= static_cast<U>(U{1} << (sizeof(U) * 8 - 1));
    }
    return static_cast<cuda::std::uint32_t>(encoded);
  }
}

/**
 * @brief Eight-byte analogue of `radix_encode_u32`: the same transforms at 64 bits
 */
template <typename T>
__device__ inline cuda::std::uint64_t radix_encode_u64(T value)
{
  if constexpr (cudf::is_timestamp<T>()) {
    return radix_encode_u64(value.time_since_epoch().count());
  } else if constexpr (cudf::is_duration<T>()) {
    return radix_encode_u64(value.count());
  } else if constexpr (cuda::std::is_floating_point_v<T>) {
    if (cuda::std::isnan(value)) { return ~cuda::std::uint64_t{0}; }
    auto const bits          = cuda::std::bit_cast<cuda::std::uint64_t>(value);
    auto constexpr sign_mask = cuda::std::uint64_t{1} << (sizeof(cuda::std::uint64_t) * 8 - 1);
    return (bits & sign_mask) ? ~bits : (bits | sign_mask);
  } else {
    using U      = cuda::std::make_unsigned_t<T>;
    auto encoded = static_cast<U>(value);
    if constexpr (cuda::std::is_signed_v<T>) {
      encoded ^= static_cast<U>(U{1} << (sizeof(U) * 8 - 1));
    }
    return static_cast<cuda::std::uint64_t>(encoded);
  }
}

/**
 * @brief Builds the single-`uint64` packed key for a fixed-width element of width four bytes or
 * less
 *
 * Layout, most- to least-significant: `[segment : S bits][class : 1 bit][value : 32 bits]` with
 * `S = bit_width(num_segments)`, so an unsigned compare orders by segment, then the polarity's
 * null placement, then the radix-encoded value (complemented within its field when descending).
 * The class bit sits above the value field and (since `S <= 31`) below the segment field, so a
 * null -- value zero, unread -- sorts on its configured side of every valid element; the gap bits
 * are zero for all elements.
 */
template <typename T>
struct numeric_packed_key_builder {
  size_type const* d_segment_ids;
  column_device_view const d_input;
  bool const has_nulls;
  int const segment_bits;
  sort_polarity const polarity;
  __device__ cuda::std::uint64_t operator()(size_type idx) const
  {
    auto const segment_part = static_cast<cuda::std::uint64_t>(d_segment_ids[idx])
                              << (64 - segment_bits);
    if (has_nulls && d_input.is_null(idx)) {
      return segment_part | (static_cast<cuda::std::uint64_t>(polarity.element_class(true))
                             << (sizeof(cuda::std::uint32_t) * 8));
    }
    return segment_part |
           (static_cast<cuda::std::uint64_t>(polarity.element_class(false))
            << (sizeof(cuda::std::uint32_t) * 8)) |
           static_cast<cuda::std::uint64_t>(radix_encode_u32<T>(d_input.element<T>(idx)) ^
                                            polarity.value_mask32());
  }
};

/**
 * @brief Builds the `prefix_key96` packed key for an eight-byte fixed-width element
 *
 * Reuses the strings path's key: `seg_null` packs the segment ordinal and the polarity's class bit
 * (via `pack_seg_null`), so the segment dominates and a null sorts on its configured side; the
 * window words carry the encoded 64-bit value split hi/lo -- complemented when descending, which
 * distributes over the split. A null leaves the value words zero.
 */
template <typename T>
struct numeric_packed_key_builder64 {
  size_type const* d_segment_ids;
  column_device_view const d_input;
  bool const has_nulls;
  sort_polarity const polarity;
  __device__ prefix_key96 operator()(size_type idx) const
  {
    auto const segment = static_cast<cuda::std::uint32_t>(d_segment_ids[idx]);
    if (has_nulls && d_input.is_null(idx)) {
      return prefix_key96{pack_seg_null(segment, polarity.element_class(true)), 0u, 0u};
    }
    prefix_key96 key{pack_seg_null(segment, polarity.element_class(false)), 0u, 0u};
    split_prefix(radix_encode_u64<T>(d_input.element<T>(idx)) ^ polarity.value_mask64(),
                 key.prefix_hi,
                 key.prefix_lo);
    return key;
  }
};

/**
 * @brief Builds a radix key for each input index via `build_key`, then two-call
 * `cub::DeviceRadixSort::SortPairs` orders the paired index array by it
 *
 * Shared by every packed-radix branch that builds a key column then radix-sorts an index array by
 * it: only the key type, key builder, optional decomposer, and end bit differ per call site. This
 * overload covers a directly radix-sortable unsigned integer key (no decomposer); the sibling
 * overload below covers a struct key (`prefix_key96`) CUB needs help decomposing.
 */
template <typename KeyT, typename IndexIteratorT, typename KeyBuilder>
void build_and_radix_sort_packed_keys(IndexIteratorT d_indices_in,
                                      size_type count,
                                      KeyBuilder const& build_key,
                                      int end_bit,
                                      size_type const* d_values_in,
                                      size_type* d_values_out,
                                      rmm::cuda_stream_view stream)
{
  auto const alloc = cudf::get_current_device_resource_ref();
  rmm::device_uvector<KeyT> keys_in(count, stream);
  rmm::device_uvector<KeyT> keys_out(count, stream);
  thrust::transform(rmm::exec_policy_nosync(stream, alloc),
                    d_indices_in,
                    d_indices_in + count,
                    keys_in.begin(),
                    build_key);
  rmm::device_buffer d_temp_storage;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                  temp_storage_bytes,
                                  keys_in.data(),
                                  keys_out.data(),
                                  d_values_in,
                                  d_values_out,
                                  count,
                                  0,
                                  end_bit,
                                  stream.value());
  d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
  cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                  temp_storage_bytes,
                                  keys_in.data(),
                                  keys_out.data(),
                                  d_values_in,
                                  d_values_out,
                                  count,
                                  0,
                                  end_bit,
                                  stream.value());
}

/// Decomposer overload of `build_and_radix_sort_packed_keys` for a struct key (e.g.
/// `prefix_key96`).
template <typename KeyT, typename IndexIteratorT, typename KeyBuilder, typename DecomposerT>
void build_and_radix_sort_packed_keys(IndexIteratorT d_indices_in,
                                      size_type count,
                                      KeyBuilder const& build_key,
                                      DecomposerT const& decomposer,
                                      int end_bit,
                                      size_type const* d_values_in,
                                      size_type* d_values_out,
                                      rmm::cuda_stream_view stream)
{
  auto const alloc = cudf::get_current_device_resource_ref();
  rmm::device_uvector<KeyT> keys_in(count, stream);
  rmm::device_uvector<KeyT> keys_out(count, stream);
  thrust::transform(rmm::exec_policy_nosync(stream, alloc),
                    d_indices_in,
                    d_indices_in + count,
                    keys_in.begin(),
                    build_key);
  rmm::device_buffer d_temp_storage;
  size_t temp_storage_bytes = 0;
  cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                  temp_storage_bytes,
                                  keys_in.data(),
                                  keys_out.data(),
                                  d_values_in,
                                  d_values_out,
                                  count,
                                  decomposer,
                                  0,
                                  end_bit,
                                  stream.value());
  d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
  cub::DeviceRadixSort::SortPairs(d_temp_storage.data(),
                                  temp_storage_bytes,
                                  keys_in.data(),
                                  keys_out.data(),
                                  d_values_in,
                                  d_values_out,
                                  count,
                                  decomposer,
                                  0,
                                  end_bit,
                                  stream.value());
}

/**
 * @brief Sorts a single fixed-width key column within its segments by one global radix sort
 *
 * The whole value is encoded into the key, so the radix order is final: elements of four bytes or
 * less use one `uint64` key, eight-byte elements the `prefix_key96`. The paired sort value is the
 * element index, so the sorted values are the segmented sorted order. The failing overload covers
 * only the non-fixed-width types, which the caller's gate never dispatches.
 */
struct numeric_packed_sort_fn {
  // Compile-time counterpart of `is_numeric_packed_radix_supported` after `dispatch_storage_type`
  // mapping: widen the two together. The DECIMAL128 rep is enabled here but fenced off at run time.
  template <typename T>
  static constexpr bool is_supported()
  {
    return cudf::is_integral<T>() or cudf::is_floating_point<T>() or cudf::is_chrono<T>();
  }

  template <typename T, CUDF_ENABLE_IF(is_supported<T>())>
  void operator()(column_device_view const& d_input,
                  size_type const* d_segment_ids,
                  bool has_nulls,
                  sort_polarity polarity,
                  int segment_bits,
                  size_type num_elements,
                  size_type* d_indices_out,
                  rmm::cuda_stream_view stream) const
  {
    auto const counting = cuda::counting_iterator<size_type>{0};
    rmm::device_uvector<size_type> indices_in(num_elements, stream);
    thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     indices_in.begin(),
                     indices_in.end(),
                     0);

    if constexpr (sizeof(T) <= 4) {
      // S = bit_width(num_segments) <= 31, so S + 1 + 32 <= 64: the three fields always fit.
      static_assert(cuda::std::bit_width(static_cast<cuda::std::uint64_t>(
                      cuda::std::numeric_limits<size_type>::max())) +
                        1 + 32 <=
                      64,
                    "packed numeric key fields exceed the 64-bit key for some segment count");
      // The segment rides the key's high bits, so the end bit cannot be tightened.
      build_and_radix_sort_packed_keys<cuda::std::uint64_t>(
        counting,
        num_elements,
        numeric_packed_key_builder<T>{d_segment_ids, d_input, has_nulls, segment_bits, polarity},
        static_cast<int>(sizeof(cuda::std::uint64_t) * 8),
        indices_in.data(),
        d_indices_out,
        stream);
    } else if constexpr (sizeof(T) == 8) {
      // seg_null packs (segment << 1) | flag, so the shift must fit a uint32.
      static_assert(2ull * cuda::std::numeric_limits<size_type>::max() + 1ull <=
                      cuda::std::numeric_limits<cuda::std::uint32_t>::max(),
                    "size_type segment label does not fit prefix_key96::seg_null after the shift");
      // seg_null needs only segment_bits + 1 bits; the constant-zero high bits skip radix passes.
      auto const key_bits = cuda::std::min(static_cast<int>(sizeof(prefix_key96) * 8),
                                           64 + cuda::std::min(32, segment_bits + 1));
      build_and_radix_sort_packed_keys<prefix_key96>(
        counting,
        num_elements,
        numeric_packed_key_builder64<T>{d_segment_ids, d_input, has_nulls, polarity},
        prefix_decomposer{},
        key_bits,
        indices_in.data(),
        d_indices_out,
        stream);
    } else {
      // Unreached fail-safe: the run-time gate excludes the DECIMAL128 (`__int128`) rep.
      CUDF_FAIL("Column type cannot be used with the numeric packed-radix segmented sort");
    }
  }

  template <typename T, CUDF_ENABLE_IF(not is_supported<T>())>
  void operator()(column_device_view const&,
                  size_type const*,
                  bool,
                  sort_polarity,
                  int,
                  size_type,
                  size_type*,
                  rmm::cuda_stream_view) const
  {
    CUDF_FAIL("Column type cannot be used with the numeric packed-radix segmented sort");
  }
};

/**
 * @brief Run-time predicate for the numeric packed-radix fast path: every fixed-width type the
 * packed key encodes losslessly
 *
 * @param type The key column's data type
 * @return true if the type is eligible for the packed-radix fast path
 */
inline bool is_numeric_packed_radix_supported(data_type type)
{
  return cudf::is_integral(type) or
         (cudf::is_fixed_point(type) and type.id() != type_id::DECIMAL128) or
         cudf::is_floating_point(type) or cudf::is_chrono(type);
}

}  // namespace

[[nodiscard]] std::unique_ptr<column> fast_segmented_sorted_order_numeric_packed(
  column_view const& input,
  column_view const& segment_offsets,
  sort_polarity polarity,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const num_elements = input.size();
  auto const num_segments = segment_offsets.size() - 1;
  CUDF_EXPECTS(num_segments >= 1, "the packed key layout requires at least one segment");

  // Dense segment ordinals: the key's segment field needs only bit_width(num_segments) bits; an
  // empty segment skips a label, preserving cross-segment order.
  rmm::device_uvector<size_type> segment_ids(num_elements, stream);
  label_segments(segment_offsets.begin<size_type>(),
                 segment_offsets.end<size_type>(),
                 segment_ids.begin(),
                 segment_ids.end(),
                 stream);

  auto const d_input   = column_device_view::create(input, stream);
  auto const has_nulls = input.has_nulls();

  // The packed key reserves a null class only when nullable; pin the caller-side coupling the
  // strings path also asserts, so a null-free column can never arrive with `nulls_first` set.
  CUDF_EXPECTS(has_nulls or not polarity.nulls_first,
               "nulls_first requires a nullable column for the packed key layout");

  auto const segment_bits =
    static_cast<int>(cuda::std::bit_width(static_cast<cuda::std::uint64_t>(num_segments)));

  auto sorted_indices = cudf::make_numeric_column(
    data_type{type_to_id<size_type>()}, num_elements, mask_state::UNALLOCATED, stream, mr);

  // Storage-type dispatch: DECIMAL32/64 sort by their int32/int64 rep.
  cudf::type_dispatcher<dispatch_storage_type>(input.type(),
                                               numeric_packed_sort_fn{},
                                               *d_input,
                                               segment_ids.data(),
                                               has_nulls,
                                               polarity,
                                               segment_bits,
                                               num_elements,
                                               sorted_indices->mutable_view().begin<size_type>(),
                                               stream);
  return sorted_indices;
}

fixed_width_sort_path choose_fixed_width_sort_path(column_view const& key,
                                                   size_type num_rows,
                                                   column_view const& segment_offsets,
                                                   [[maybe_unused]] rmm::cuda_stream_view stream)
{
  auto const type = key.type();
  // String / nested keys have no fixed-width fast path.
  if (not is_numeric_packed_radix_supported(type)) { return fixed_width_sort_path::comparison; }

  // avg_list_size uses num_rows / num_offsets, the heuristic prefer_cub_segmented_sort also uses.
  auto const num_offsets   = segment_offsets.size();
  auto const avg_list_size = num_rows / num_offsets;

  // Packed radix folds validity into the key, so nulls need no separate pass, at any list size.
  if (key.has_nulls()) { return fixed_width_sort_path::packed_radix; }
  // Long lists amortize the global packed-key radix's bandwidth-bound pass.
  if (avg_list_size >= MAX_AVG_LIST_SIZE_FOR_FAST_SORT) {
    return fixed_width_sort_path::packed_radix;
  }
  // The short no-null remainder keeps main's CUB-or-comparison decision.
  return fixed_width_sort_path::comparison;
}

}  // namespace detail
}  // namespace cudf
