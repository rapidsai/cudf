/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/concatenate.hpp>
#include <cudf/detail/null_mask.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/utilities/batched_memcpy.hpp>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/dictionary/detail/concatenate.hpp>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/aligned.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>

#include <cub/block/block_reduce.cuh>
#include <cub/device/dispatch/dispatch_batch_memcpy.cuh>
#include <cub/device/dispatch/tuning/tuning_batch_memcpy.cuh>
#include <cuda/atomic>
#include <cuda/bit>
#include <cuda/memory_resource>
#include <cuda/std/bit>
#include <cuda/stream>
#include <thrust/binary_search.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace cudf {
namespace detail {
namespace {


// ============================================================================
// 1. Utilities
// Shared helpers, memcpy batching, pinned staging.
// ============================================================================

/**
 * @brief Launches `launch` once per `gridDim.y`-sized chunk of `total` rows/segments/levels.
 *
 * gridDim.y is capped at 65535, so any launch whose grid.y indexes an unbounded item count must
 * chunk. Every launch site shares the same math: a fixed x-dim, y-dim of `min(65535, remaining)`,
 * and a per-chunk item offset of `first` for the kernel's pointer arithmetic. `launch` receives
 * `(first, grid)` and issues the kernel launch itself, keeping each site's kernel name, arguments,
 * and stream untouched.
 */
template <typename LaunchFn>
void for_each_grid_y_chunk(size_type total, unsigned int num_blocks_x, LaunchFn&& launch)
{
  constexpr size_type max_grid_y = 65535;
  for (size_type first = 0; first < total; first += max_grid_y) {
    dim3 const grid(num_blocks_x, static_cast<unsigned int>(std::min(max_grid_y, total - first)));
    launch(first, grid);
  }
}

/// CUB's stock batched-memcpy tuning with a 16 KiB block-level tile (256 threads x 64 bytes) in
/// place of its 8 KiB one, which suits the buffer-size mix this file copies better.
struct memcpy_tile_policy {
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const
    -> cub::BatchedCopyPolicy
  {
    auto policy = cub::detail::batch_memcpy::policy_selector{}(cc);
    policy.lookback.large_buffer.threads_per_block     = 256;
    policy.lookback.large_buffer.bytes_per_thread      = 64;
    policy.lookback.small_buffer.block_level_tile_size = 256 * 64;
    return policy;
  }
};

/// Runs CUB's batched memcpy under `memcpy_tile_policy`. A null @p d_temp only stores the scratch
/// size the copy needs in @p temp_bytes and enqueues nothing.
void batched_memcpy(void* d_temp,
                    std::size_t& temp_bytes,
                    void const** src,
                    void** dst,
                    std::size_t* sizes,
                    std::size_t num_buffs,
                    cuda::stream_ref stream)
{
  cub::detail::batch_memcpy::dispatch<cub::CopyAlg::Memcpy, uint32_t>(
    d_temp,
    temp_bytes,
    src,
    dst,
    sizes,
    static_cast<::cuda::std::int64_t>(num_buffs),
    stream.get(),
    memcpy_tile_policy{});
}

/**
 * @brief A pinned host block that metadata arrays are copied into on their way to the device.
 *
 * An async upload may read a pageable host source when the stream reaches the copy rather than
 * during the call, after a short-lived `std::vector` source is already freed. A staged array is
 * pinned, and the block returns to the pool on the stream its uploads are issued on, so it stays
 * valid until they finish. One block, sized up front, serves the whole call, because the pinned
 * allocator synchronizes the stream it allocates on.
 */
class pinned_metadata {
 public:
  /// Takes no block at all when `bytes` is zero, which is how the pageable arm is selected.
  pinned_metadata(std::size_t bytes, cuda::stream_ref stream)
  {
    if (bytes == 0) { return; }
    block_.emplace(cudf::detail::make_pinned_vector_async<std::byte>(bytes, stream));
    device_accessible_ = host_span<std::byte>{*block_}.is_device_accessible();
  }

  /**
   * @brief Returns a device-accessible copy of `src`, or `src` itself on the pageable arm.
   *
   * Running out of block is a sizing bug and not a fallback: it would put that one array back on
   * the pageable path and so back under the lifetime contract above, where it is unsafe rather than
   * merely slow. `pinned_metadata_budget` sizes the block to an upper bound so this cannot happen,
   * and it fails loudly rather than silently degrading if that bound is ever wrong.
   */
  template <typename T>
  host_span<T const> stage(host_span<T const> src)
  {
    static_assert(std::is_trivially_copyable_v<T>, "staged metadata is copied bytewise");
    if (!device_accessible_) { return src; }

    // The pool aligns the block itself to at least this much, so an aligned offset is an aligned
    // address, and one allowance of it per call is what the budget below reserves
    auto const start = cudf::util::round_up_safe(used_, alignof(std::max_align_t));
    auto const bytes = src.size() * sizeof(T);
    CUDF_EXPECTS(start + bytes <= block_->size(),
                 "batch_concatenate pinned metadata block is undersized");

    auto* const dst = block_->data() + start;
    std::memcpy(dst, src.data(), bytes);
    used_ = start + bytes;
    return host_span<T const>{reinterpret_cast<T const*>(dst), src.size(), true};
  }

  /// Stages `src` into the block and uploads it to device memory in one step.
  template <typename T>
  auto upload(std::vector<T> const& src, cuda::stream_ref stream, rmm::device_async_resource_ref mr)
  {
    return cudf::detail::make_device_uvector_async(stage(host_span<T const>{src}), stream, mr);
  }

 private:
  std::optional<cudf::detail::host_vector<std::byte>> block_;
  bool device_accessible_{false};
  std::size_t used_{0};
};

/**
 * @brief Divides `sizes` into `ways` contiguous runs carrying as close to equal bytes as possible.
 *
 * Balancing by descriptor count instead would leave one stream with the chars buffers and the other
 * with the fixed-width ones, which on this schema differ by more than an order of magnitude.
 *
 * @return `ways + 1` boundaries, so run `i` is `[out[i], out[i + 1])`
 */
std::vector<std::size_t> partition_copy_by_bytes(host_span<std::size_t const> sizes, int ways)
{
  auto const total = std::accumulate(sizes.begin(), sizes.end(), std::size_t{0});
  std::vector<std::size_t> bounds{0};
  bounds.reserve(ways + 1);

  std::size_t running = 0;
  for (std::size_t i = 0; i < sizes.size(); ++i) {
    running += sizes[i];
    // Close a run once it holds its share, leaving enough descriptors for the runs still to come
    auto const share =
      (total * static_cast<std::size_t>(bounds.size())) / static_cast<std::size_t>(ways);
    if (static_cast<int>(bounds.size()) < ways && running >= share &&
        sizes.size() - (i + 1) >= static_cast<std::size_t>(ways) - bounds.size()) {
      bounds.push_back(i + 1);
    }
  }
  while (static_cast<int>(bounds.size()) <= ways) {
    bounds.push_back(sizes.size());
  }
  return bounds;
}

// ============================================================================
// 2. Data model
// Level state, deferred-read bookkeeping, slicing-plan data types.
// ============================================================================

/**
 * @brief Consolidated data structure for a single nesting level.
 *
 * Contains all information needed for both data copy and mask concatenation.
 *
 * Per-source arrays are small `std::vector`s: each holds 2–3 elements
 * (`chars_partition_offsets` holds one more). Flattened / cross-column scratch stays on
 * `std::vector`.
 */
struct level_info {
  // Data copy info
  std::vector<void const*> src_data_ptrs;  // Source data pointers (empty for struct levels)
  std::vector<std::size_t> data_sizes;     // Size of each data buffer in bytes

  // Mask concatenation info
  std::vector<bitmask_type const*> mask_ptrs;  // Source mask pointers (can be null)
  std::vector<size_type> mask_offsets;         // Bit offset for each mask
  std::vector<size_type> col_sizes;            // Number of rows in each column

  // String/List offset concatenation info
  std::vector<void const*> src_offsets_ptrs;         // Source offsets pointers
  std::vector<size_type> src_offsets_offsets;        // Offset within each source offsets column
  std::vector<std::size_t> chars_partition_offsets;  // Cumulative child offsets for each column
  std::vector<int64_t> first_src_offsets;  // First offset value for each column (for adjustment)
  data_type offsets_dtype;                 // Type of offsets column (INT32 or INT64)
  std::size_t total_child_size;            // Total chars bytes (STRING) or child rows (LIST)

  // LIST-specific info
  data_type child_dtype;  // Type of child column (for empty list reconstruction)
  std::optional<column_view> empty_child_like;
  // True when fill/collect emitted a list child level (preorder next level). Needed when
  // empty_child_like is unset but total_child_size later resolves to 0.
  bool list_child_collected{false};

  // DICTIONARY delegation info
  bool delegate_to_specialized;               // If true, skip batch processing and delegate
  std::vector<column_view> original_columns;  // Store column views for delegation

  // Level metadata
  data_type dtype;             // Data type at this level
  size_type total_rows;        // Total number of rows
  size_type total_null_count;  // Sum of null counts from all columns
  size_type num_children;      // Number of children (for struct types)
  bool has_nulls;              // Whether any column has nulls
  bool is_struct;              // Whether this level is a struct type
  bool is_string;              // Whether this level is a string type
  bool is_list;                // Whether this level is a list type
  bool is_dictionary;          // Whether this level is a dictionary type
};

/**
 * @brief A deferred read of one element of an offsets column.
 */
struct pending_offset_read {
  void const* offsets_data;  ///< Offsets column data pointer
  size_type index;           ///< Element to read, relative to `offsets_data`
  size_type type_size;       ///< Element width in bytes, 4 or 8
};

/**
 * @brief Where to store the child bounds of one source column once its reads are resolved.
 */
struct pending_child_bounds {
  std::vector<level_info>* levels;  ///< Level list holding the target level
  size_type level_idx;              ///< Index of the target level
  size_type col_pos;                ///< Position of the source column within that level
  int32_t first_slot;               ///< Read slot holding the first offset, or -1 when it is zero
  int32_t last_slot;                ///< Read slot holding the last offset
};

/**
 * @brief Child bounds whose device reads are batched into a single round trip.
 *
 * The first and last offset of every string and list source column live on the device. Reading
 * them one at a time costs a device-to-host copy plus a synchronization each, which dominates
 * metadata collection for wide, deeply nested schemas. `collect_level_info_recursive` records a
 * request per source column instead, leaving the dependent sizes as placeholders, and
 * `resolve_deferred_bounds` gathers every value with one kernel and one synchronization.
 *
 * The slicing-plan path does not use this; it batches its reads in a `slicing_plan` instead,
 * with device-chained gathers and a single D2H (see `collect_levels_bfs`).
 */
struct deferred_bounds {
  std::vector<pending_offset_read> reads;
  std::vector<pending_child_bounds> targets;

  /**
   * @brief Records a read of `offsets[element_index]` and returns its slot in the batch.
   *
   * @param offsets Offsets column to read from
   * @param element_index Element to read, in the offsets column's own index space
   * @return Slot holding the value once the batch is resolved
   */
  int32_t add_read(column_view const& offsets, size_type element_index)
  {
    reads.push_back({offsets.head(),
                     offsets.offset() + element_index,
                     static_cast<size_type>(cudf::size_of(offsets.type()))});
    return static_cast<int32_t>(reads.size() - 1);
  }
};

/**
 * @brief A null count that could not be taken from a source view during the walk.
 *
 * Synthesized sliced struct/list child views use placeholder `null_count == 0`. The real range
 * popcount is deferred to one batched kernel after the walk (before phase-1 slab sizing).
 */
struct pending_null_count {
  bitmask_type const* mask;  ///< Never null; maskless views record nothing
  size_type begin_bit;       ///< Inclusive, in the mask's bit space
  size_type end_bit;         ///< Exclusive
  std::vector<level_info>* levels;
  size_type level_idx;
};

struct deferred_null_counts {
  std::vector<pending_null_count> requests;

  void add(bitmask_type const* mask,
           size_type begin_bit,
           size_type end_bit,
           std::vector<level_info>* levels,
           size_type level_idx)
  {
    if (mask == nullptr || begin_bit >= end_bit) { return; }
    requests.push_back({mask, begin_bit, end_bit, levels, level_idx});
  }

  void maybe_add(column_view const& col,
                 size_type begin_bit,
                 size_type end_bit,
                 std::vector<level_info>* levels,
                 size_type level_idx)
  {
    if (col.null_mask() == nullptr || begin_bit >= end_bit) { return; }
    add(col.null_mask(), begin_bit, end_bit, levels, level_idx);
  }
};

void resolve_deferred_null_counts(deferred_null_counts& deferred, cuda::stream_ref stream);

/**
 * @brief One offset-buffer read in the device-chained slicing plan.
 *
 * Slot index equals the position in the device `d_values` buffer. The element index is
 * `absolute_index + (add_slot0>=0 ? d_values[add_slot0] : 0) + (add_slot1>=0 ? d_values[add_slot1]
 * : 0)
 *  - (sub_slot>=0 ? d_values[sub_slot] : 0)`. Generation-0 reads have all deps at -1
 * (host-absolute). Later generations depend only on earlier slots; kernels chain on the GPU with no
 * host sync.
 */
struct planned_offset_read {
  void const* offsets_data;
  size_type type_size;
  size_type absolute_index;
  int32_t add_slot0;  ///< -1 if unused
  int32_t add_slot1;  ///< -1 if unused
  int32_t sub_slot;   ///< -1 if unused
  size_type generation;
};

/**
 * @brief A list child to materialize on the host after the single slicing D2H.
 */
struct planned_list_child {
  std::vector<level_info>* levels;
  size_type child_level_idx;
  size_type child_source;
  size_type run_base;            ///< Index of child_source==0 entry for this (levels, slot)
  column_view const* raw_child;  ///< Borrowed storage; never null
  int32_t first_slot;
  int32_t last_slot;
  bool verbatim;
};

/**
 * @brief Host-side description of a source view whose offset/size may depend on earlier gather
 * slots.
 */
struct planned_source {
  column_view const* storage;  ///< Type tree / buffers; borrowed, never null
  size_type offset_base;       ///< Host part of the view offset
  int32_t offset_add_slot;     ///< -1, or add d_values[slot] to form offset
  bool size_is_span;           ///< size = values[size_last] - values[size_first]
  size_type size_const;
  int32_t size_first_slot;  ///< -1 means 0 when size_is_span
  int32_t size_last_slot;
  size_type generation;  ///< List-nesting generation of this view's offset dependency
  bool exact;            ///< offset and size fully known on the host
};

struct slicing_plan {
  std::vector<planned_offset_read> reads;
  std::vector<size_type> generation_ends;  ///< Exclusive end slot per generation
  std::vector<pending_child_bounds> targets;
  std::vector<planned_list_child> list_children;

  int32_t add_read(void const* offsets_data,
                   size_type type_size,
                   size_type absolute_index,
                   int32_t add_slot0,
                   int32_t add_slot1,
                   int32_t sub_slot,
                   size_type generation)
  {
    auto const slot = static_cast<int32_t>(reads.size());
    reads.push_back(
      {offsets_data, type_size, absolute_index, add_slot0, add_slot1, sub_slot, generation});
    return slot;
  }
};

/**
 * @brief One (levels vector, preorder slot) whose planned sources are ready to plan or fill.
 */
struct collect_frontier_entry {
  std::vector<level_info>* levels;
  size_type level_idx;
  size_type col_idx;
  size_type first_source;
  size_type num_sources;
};

// Uniform accessors for the level checks: value spans hold `column_view`, pointer spans hold
// borrowed `column_view const*`, and `fill_view` keeps its row count as a data member.

size_type level_col_size(column_view const& c) { return c.size(); }
size_type level_col_size(column_view const* c) { return c->size(); }

column_view const& level_col_view(column_view const& c) { return c; }
column_view const& level_col_view(column_view const* c) { return *c; }

/// Whether a position's roots are EMPTY-typed or all zero-row.
template <typename ColRef>
bool is_empty_position(host_span<ColRef const> roots, ColRef const& first)
{
  auto const& view = level_col_view(first);
  return view.type().id() == type_id::EMPTY ||
         std::all_of(roots.begin(), roots.end(), [](ColRef const& c) {
           return level_col_view(c).is_empty();
         });
}

/**
 * @brief Pinned bytes to reserve so every metadata upload of one call can be staged.
 *
 * An upper bound over every array the phases stage per (level, source) pair and per level; falling
 * short would put an array back on the pageable path, which `stage` refuses outright.
 */
struct offsets_segment {
  void const* src_ptr;  ///< Source offsets buffer for this run
  void* out_ptr;        ///< Output offsets buffer of the column this run belongs to
  int64_t delta;        ///< Added to every source value in the run
  int64_t tail_value;   ///< Total child size, written past the run when `write_tail`
  size_type src_base;   ///< Index of the run's first element within the source
  size_type out_base;   ///< Index of the run's first element within the output
  size_type count;      ///< Elements in the run
  bool src_is_int64;    ///< Width of the source offsets
  bool write_tail;      ///< Set on a column's last run, which owns the trailing total
};

std::size_t pinned_metadata_budget(std::vector<level_info> const& levels)
{
  // Upper bound on the bytes the phases stage per (level, source) pair
  constexpr std::size_t bytes_per_pair = 4 * sizeof(void*) + 3 * sizeof(std::size_t) +
                                         4 * sizeof(size_type) + sizeof(int64_t) +
                                         sizeof(offsets_segment);
  // Upper bound on the bytes the phases stage per level
  constexpr std::size_t bytes_per_level = 2 * sizeof(void*) + 2 * sizeof(std::size_t) +
                                          9 * sizeof(size_type) +
                                          sizeof(cudf::detail::output_offsetalator);

  std::size_t pairs = 0;
  for (auto const& level : levels) {
    // Doubled so a phase that gains an array stays covered, plus a fixed allowance for the
    // alignment gap ahead of each staged array
    pairs += std::max({level.col_sizes.size(),
                       level.src_data_ptrs.size(),
                       level.mask_ptrs.size(),
                       level.src_offsets_ptrs.size()}) +
             1;
  }

  // Doubled, and with a fixed allowance for the alignment gap each staged array may lose ahead of
  // it. The counted arrays come to a few tens of bytes per pair, so the block is kilobytes either
  // way, and the factor is what stops a phase that gains an array from pushing a later one off the
  // end before anyone revisits the counts above
  return 2 * (pairs * bytes_per_pair + levels.size() * bytes_per_level) +
         128 * alignof(std::max_align_t);
}

/// Sums `pinned_metadata_budget` over every column position of a table concatenation.
std::size_t pinned_metadata_budget(std::vector<std::vector<level_info>> const& levels_by_column)
{
  std::size_t bytes = 0;
  for (auto const& levels : levels_by_column) {
    bytes += pinned_metadata_budget(levels);
  }
  return bytes;
}


// ============================================================================
// 3. Validation
// Bounds and type checks shared by every walk.
// ============================================================================

/**
 * @brief Checks if the given data type is supported by batch_concatenate.
 *
 * Supported types are: fixed-width types (plain types), struct types, string types,
 * list types, and dictionary types.
 */
bool is_batch_concat_supported_type(data_type type)
{
  return cudf::is_fixed_width(type) || type.id() == type_id::STRUCT ||
         type.id() == type_id::STRING || type.id() == type_id::LIST ||
         type.id() == type_id::DICTIONARY32;
}

/**
 * @brief Borrows a child out of its parent's own child vector, copying neither.
 *
 * `child_begin()` returns an iterator into that vector, where `child(i)` and the typed views'
 * accessors all hand back a copy of the subtree. The reference lives as long as the parent, which
 * at every call site here outlives the walk that reads it.
 *
 * @param parent The column holding the child
 * @param index Index of the child
 * @return A reference to the child, valid while @p parent is
 */
column_view const& borrowed_child(column_view const& parent, size_type index)
{
  return *(parent.child_begin() + index);
}

/// Payload below which the batched path's fixed setup — the metadata uploads and kernel launches
/// it needs before copying anything — costs more than the per-column launches it removes.
///
/// Measured on the synthetic flat benchmark (interleaved on/off, neon-sandbox): forcing the
// batched engine on below-threshold input costs ~3x at 32-128 KB payloads (21.7 -> 63.7 us
// at 8 columns x 1024 rows) and ~2.3x at 2 MB (29.7 -> 69.1 us); the two arms converge at the
// 8 MiB boundary itself (84.1 vs 85.4 us). The delegation is why the fixed setup must stay
// paid only when it pays for itself.
constexpr std::size_t min_batch_payload_bytes = std::size_t{8} << 20;

/// Block size shared by the batched-metadata and copy kernels in this file.
constexpr size_type batch_block_size{256};

/// @brief An EMPTY-typed column of @p length rows: no data, no mask, null count = length.
std::unique_ptr<column> make_empty_type_column(size_type length,
                                               cuda::stream_ref stream,
                                               rmm::device_async_resource_ref mr)
{
  return std::make_unique<column>(
    data_type(type_id::EMPTY),
    length,
    rmm::device_buffer{},
    cudf::detail::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr),
    length);
}

/**
 * @brief Bytes a flat run of columns will copy, or nullopt if any of them is nested.
 *
 * Only flat fixed-width input has a payload that can be measured without touching the device, and
 * only flat input is at risk of being too small to be worth batching: a nested column costs the
 * unbatched path one launch per level per column, which the batched path wins at any size.
 */
std::optional<std::size_t> flat_payload_bytes(host_span<column_view const> cols)
{
  std::size_t bytes = 0;
  for (auto const& col : cols) {
    // EMPTY is excluded because the batched path constructs those columns itself
    if (col.type().id() == type_id::EMPTY || !cudf::is_fixed_width(col.type())) {
      return std::nullopt;
    }
    bytes += static_cast<std::size_t>(col.size()) * cudf::size_of(col.type());
  }
  return bytes;
}

/// @brief Same payload rule across every column of every table; nullopt if any column is nested.
std::optional<std::size_t> flat_payload_bytes(host_span<table_view const> tables)
{
  std::size_t bytes = 0;
  for (auto const& table : tables) {
    for (auto const& col : table) {
      if (col.type().id() == type_id::EMPTY || !cudf::is_fixed_width(col.type())) {
        return std::nullopt;
      }
      bytes += static_cast<std::size_t>(col.size()) * cudf::size_of(col.type());
    }
  }
  return bytes;
}

template <typename T>
host_span<T const> span_of(std::vector<T> const& v)
{
  return host_span<T const>(v.data(), v.size());
}

/**
 * @brief Verifies bounds and type compatibility for batch concatenation.
 */
void batch_bounds_and_type_check(host_span<column_view const> cols,
                                 cuda::stream_ref stream,
                                 bool check_types = true);
void bounds_check_ptrs(host_span<column_view const* const> ptrs, cuda::stream_ref stream)
{
  std::vector<column_view> materialized;
  materialized.reserve(ptrs.size());
  for (auto const* p : ptrs) {
    materialized.push_back(*p);
  }
  batch_bounds_and_type_check(
    host_span<column_view const>{materialized.data(), materialized.size()}, stream);
}

/// @brief Throws when the tables do not all have the same column count.
void validate_table_shapes(host_span<table_view const> tables)
{
  CUDF_EXPECTS(std::all_of(tables.begin(),
                           tables.end(),
                           [num_columns = tables.front().num_columns()](auto const& t) {
                             return t.num_columns() == num_columns;
                           }),
               "Mismatch in table columns to concatenate.");
}

/// Fails when columns to concatenate disagree on type; the shared message of every type check.
inline void expect_type_match(bool cond)
{
  CUDF_EXPECTS(cond, "Type mismatch in columns to concatenate.", cudf::data_type_error);
}

/**
 * @brief Compare list-child type trees for unvisited peers (control deep-copies via `child()`).
 */

void expect_matching_list_child_type_trees(column_view const& want, column_view const& col)
{
  auto const ci = lists_column_view::child_column_index;
  expect_type_match(cudf::have_same_types(want.child(ci), col.child(ci)));
}

/// @brief Typed head() through a dependent walker column (edg rejects `c.template head<T>()`).
template <typename T, typename C>
T const* col_head(C const& c)
{
  return c.template head<T>();
}

/**
 * @brief Per-level type match used by the fused metadata walk, templated over the three source
 *        shapes (`column_view`, `column_view const*`, `fill_view`).
 *
 * The walk visits every nesting level, so a same-level comparison is enough for structs and
 * leaves: child mismatches are raised when the walk reaches that child. (The recursive root
 * checker additionally compares each root's whole nested type tree once at the outermost level.)
 *
 * Lists and dictionaries need extra care. Collect only recurses into non-empty list sources, so a
 * nested mismatch on an empty peer would never be visited — when any source would be skipped, the
 * list child type trees are compared here. Dictionaries are never recursed into by collect, so
 * they always go through `have_same_types`.
 *
 * Same exception type and message as the recursive root checker. The fill walk borrows storage for
 * the type tree and asserts both list children exist before comparing, while the view walkers treat
 * both-missing as a degenerate match — `RequireBothListChildren` selects the fill semantics.
 */
template <typename ColRef, bool RequireBothListChildren = false>
void expect_same_types_at_level_impl(host_span<ColRef const> cols)
{
  if (cols.size() <= 1) { return; }

  auto const& want = level_col_view(cols.front());
  auto const tid   = want.type().id();

  // List child trees must be compared here whenever collect will not visit every source's child
  auto const list_child_unvisited =
    tid == type_id::LIST && std::any_of(cols.begin(), cols.end(), [](ColRef const& c) {
      return level_col_size(c) == 0 || level_col_view(c).num_children() == 0;
    });

  for (auto it = std::next(cols.begin()); it != cols.end(); ++it) {
    auto const& col = level_col_view(*it);
    if (tid == type_id::DICTIONARY32) {
      expect_type_match(cudf::have_same_types(want, col));
      continue;
    }
    expect_type_match(col.type() == want.type());
    if (tid == type_id::STRUCT) {
      expect_type_match(col.num_children() == want.num_children());
    } else if (list_child_unvisited) {
      auto const ci = lists_column_view::child_column_index;
      if constexpr (RequireBothListChildren) {
        expect_type_match(col.num_children() > ci && want.num_children() > ci);
      } else if (want.num_children() > ci || col.num_children() > ci) {
        // Both missing children is a match (degenerate empty lists); otherwise compare
        expect_type_match(want.num_children() > ci && col.num_children() > ci);
      }
      expect_matching_list_child_type_trees(want, col);
    }
  }
}

template <typename ColRef>

void check_level_offset_overflow(host_span<ColRef const> cols)
{
  size_t const total_offset_count =
    std::accumulate(
      cols.begin(),
      cols.end(),
      std::size_t{},
      [](size_t a, ColRef const& b) { return a + static_cast<size_t>(level_col_size(b)); }) +
    1;
  CUDF_EXPECTS(total_offset_count <= static_cast<size_t>(std::numeric_limits<size_type>::max()),
               "Total number of concatenated offsets exceeds the column size limit",
               std::overflow_error);
}

/**
 * @brief Shared body of the per-level bounds and type checks: rows overflow, EMPTY
 *        all-or-nothing, support set, then per-level type match and offsets overflow.
 *
 * Templated over the three source shapes (`column_view`, `column_view const*`, `fill_view`),
 * including the same overflow and support messages as the non-recursive body of
 * `batch_bounds_and_type_check`. `FullChecks=false` runs only the first three phases — the
 * recursive roots use it as their prologue and keep their own root-only type tree compare and
 * per-branch recursion.
 *
 * @param cols Columns of one nesting level
 * @return true if this level is EMPTY (caller should not collect further)
 */
template <typename ColRef, bool RequireBothListChildren = false, bool FullChecks = true>

bool check_bounds_and_types_at_level_impl(host_span<ColRef const> cols)
{
  size_t const total_row_count =
    std::accumulate(cols.begin(), cols.end(), std::size_t{}, [](size_t a, ColRef const& b) {
      return a + static_cast<size_t>(level_col_size(b));
    });
  CUDF_EXPECTS(total_row_count <= static_cast<size_t>(std::numeric_limits<size_type>::max()),
               "Total number of concatenated rows exceeds the column size limit",
               std::overflow_error);

  if (std::any_of(cols.begin(), cols.end(), [](ColRef const& c) {
        return level_col_view(c).type().id() == cudf::type_id::EMPTY;
      })) {
    CUDF_EXPECTS(std::all_of(cols.begin(),
                             cols.end(),
                             [](ColRef const& c) {
                               return level_col_view(c).type().id() == cudf::type_id::EMPTY;
                             }),
                 "Mismatch in columns to concatenate.",
                 cudf::data_type_error);
    return true;
  }

  CUDF_EXPECTS(std::all_of(cols.begin(),
                           cols.end(),
                           [](ColRef const& c) {
                             return is_batch_concat_supported_type(level_col_view(c).type());
                           }),
               "batch_concatenate only supports fixed-width, struct, string, list, and dictionary "
               "types.",
               cudf::logic_error);

  if constexpr (FullChecks) {
    expect_same_types_at_level_impl<ColRef, RequireBothListChildren>(cols);

    auto const tid = level_col_view(cols.front()).type().id();
    if (tid == type_id::STRING || tid == type_id::LIST) { check_level_offset_overflow(cols); }
  }

  return false;
}


// ============================================================================
// 4. Slicing plan
// Host-side plan construction and the single-sync resolve path.
// ============================================================================

/**
 * @brief Preorder child-slot tree for one output column, assigned host-side before BFS fill.
 *
 * Reconstruct and superimpose walk `levels` with a preorder cursor; BFS visit order must not
 * change emission order. The shape pass numbers every slot and records each node's children.
 */
struct level_tree_shape {
  std::vector<std::vector<size_type>> children_of;
  size_type num_levels{0};
};

/**
 * @brief Shape-only list-child existence over borrowed pointers (no view copies).
 */
bool list_child_level_exists_ptrs(host_span<column_view const* const> cols)
{
  return std::any_of(cols.begin(), cols.end(), [](column_view const* c) {
    return c->size() > 0 && c->num_children() > 0;
  });
}

/**
 * @brief Host-only preorder numbering using borrowed child pointers (no deep view copies).
 *
 * Shape needs the type tree and list-child existence only. Sliced offset/size are irrelevant
 * here; unsliced child storage has the same schema as any host slice of the parent.
 */
size_type assign_preorder_slots(host_span<column_view const* const> cols,
                                level_tree_shape& shape,
                                size_type slot)
{
  if (cols.empty()) { return slot; }

  auto const* first = cols.front();
  if (first->type().id() == type_id::EMPTY) { return slot; }

  if (static_cast<size_type>(shape.children_of.size()) <= slot) {
    shape.children_of.resize(static_cast<std::size_t>(slot) + 1);
  }
  shape.children_of[slot].clear();
  auto next = slot + 1;

  if (first->type().id() == type_id::DICTIONARY32) { return next; }

  if (first->type().id() == type_id::STRUCT) {
    auto const num_children = first->num_children();
    std::vector<column_view const*> child_ptrs;
    child_ptrs.reserve(cols.size());
    for (size_type child_idx = 0; child_idx < num_children; ++child_idx) {
      child_ptrs.clear();
      for (auto const* col : cols) {
        child_ptrs.push_back(&borrowed_child(*col, child_idx));
      }
      shape.children_of[slot].push_back(next);
      next = assign_preorder_slots(
        host_span<column_view const* const>{child_ptrs.data(), child_ptrs.size()}, shape, next);
    }
  } else if (first->type().id() == type_id::LIST && list_child_level_exists_ptrs(cols)) {
    std::vector<column_view const*> child_ptrs;
    child_ptrs.reserve(cols.size());
    for (auto const* col : cols) {
      if (col->size() > 0 && col->num_children() > 0) {
        child_ptrs.push_back(&borrowed_child(*col, lists_column_view::child_column_index));
      }
    }
    shape.children_of[slot].push_back(next);
    next = assign_preorder_slots(
      host_span<column_view const* const>{child_ptrs.data(), child_ptrs.size()}, shape, next);
  }
  return next;
}

level_tree_shape build_level_tree_shape(host_span<column_view const* const> cols)
{
  level_tree_shape shape;
  shape.num_levels = assign_preorder_slots(cols, shape, 0);
  shape.children_of.resize(static_cast<std::size_t>(shape.num_levels));
  return shape;
}

planned_source make_exact_planned(column_view const& col, size_type generation)
{
  return planned_source{&col,
                        static_cast<size_type>(col.offset()),
                        -1,
                        false,
                        static_cast<size_type>(col.size()),
                        -1,
                        -1,
                        generation,
                        true};
}

planned_source make_span_planned(column_view const& raw_child,
                                 int32_t first_slot,
                                 int32_t last_slot,
                                 size_type parent_generation)
{
  return planned_source{&raw_child,
                        static_cast<size_type>(raw_child.offset()),
                        first_slot,
                        true,
                        0,
                        first_slot,
                        last_slot,
                        parent_generation + 1,
                        false};
}

std::pair<int32_t, int32_t> plan_endpoint_reads(slicing_plan& plan,
                                                planned_source const& src,
                                                column_view const& offsets_col)
{
  auto const type_size = static_cast<size_type>(cudf::size_of(offsets_col.type()));
  auto const off_base  = static_cast<size_type>(offsets_col.offset());

  auto const offset_known_zero = src.offset_add_slot < 0 && src.offset_base == 0;
  int32_t first_slot           = -1;
  if (!offset_known_zero) {
    first_slot = plan.add_read(offsets_col.head(),
                               type_size,
                               off_base + src.offset_base,
                               src.offset_add_slot,
                               -1,
                               -1,
                               src.generation);
  }

  int32_t last_slot;
  if (src.size_is_span) {
    last_slot = plan.add_read(offsets_col.head(),
                              type_size,
                              off_base + src.offset_base,
                              src.offset_add_slot,
                              src.size_last_slot,
                              src.size_first_slot,
                              src.generation);
  } else {
    last_slot = plan.add_read(offsets_col.head(),
                              type_size,
                              off_base + src.offset_base + src.size_const,
                              src.offset_add_slot,
                              -1,
                              -1,
                              src.generation);
  }
  return {first_slot, last_slot};
}

struct plan_frontier_item {
  std::vector<level_info>* levels;
  size_type level_idx;
  size_type col_idx;
  std::vector<planned_source> sources;  ///< Borrowed storage pointers (no deep view copies)
};

/**
 * @brief Shared cumulative sweep over contiguous same-level bounds targets.
 *
 * Fills `chars_partition_offsets` gaps with the running total, records each target's first
 * offset, advances string data pointers/sizes, and finalizes each level's child size.
 */
template <typename Target>
void apply_bounds_targets(host_span<Target const> targets, host_span<int64_t const> values)
{
  std::size_t next = 0;
  while (next < targets.size()) {
    auto* const owner     = targets[next].levels;
    auto const level_idx  = targets[next].level_idx;
    auto& level           = (*owner)[level_idx];
    std::size_t running   = 0;
    std::size_t col_pos   = 0;
    auto const same_level = [&](Target const& t) {
      return t.levels == owner && t.level_idx == level_idx;
    };

    for (; next < targets.size() && same_level(targets[next]); ++next) {
      auto const& target = targets[next];

      for (; col_pos < static_cast<std::size_t>(target.col_pos); ++col_pos) {
        level.chars_partition_offsets[col_pos + 1] = running;
      }

      auto const first = target.first_slot < 0 ? int64_t{0} : values[target.first_slot];
      auto const span  = static_cast<std::size_t>(values[target.last_slot] - first);

      level.first_src_offsets[col_pos] = first;
      if (level.is_string) {
        level.src_data_ptrs[col_pos] =
          static_cast<char const*>(level.src_data_ptrs[col_pos]) + first;
        level.data_sizes[col_pos] = span;
      }

      running += span;
      level.chars_partition_offsets[++col_pos] = running;
    }

    for (; col_pos < level.col_sizes.size(); ++col_pos) {
      level.chars_partition_offsets[col_pos + 1] = running;
    }

    level.total_child_size = running;
  }
}

__device__ __forceinline__ int64_t read_offset_value(void const* ptr,
                                                     size_type idx,
                                                     size_type type_size)
{
  return (type_size == 8) ? static_cast<int64_t const*>(ptr)[idx]
                          : static_cast<int64_t>(static_cast<int32_t const*>(ptr)[idx]);
}

CUDF_KERNEL void gather_offset_values_indirect_kernel(
  void const* const* __restrict__ offsets_ptrs,
  size_type const* __restrict__ absolute_indices,
  int32_t const* __restrict__ add_slot0,
  int32_t const* __restrict__ add_slot1,
  int32_t const* __restrict__ sub_slot,
  size_type const* __restrict__ type_sizes,
  int64_t const* __restrict__ d_values,
  int64_t* __restrict__ out,
  size_type num_reads)
{
  auto const read_idx = cudf::detail::grid_1d::global_thread_id();
  if (read_idx >= static_cast<thread_index_type>(num_reads)) { return; }

  int64_t index = absolute_indices[read_idx];
  if (add_slot0[read_idx] >= 0) { index += d_values[add_slot0[read_idx]]; }
  if (add_slot1[read_idx] >= 0) { index += d_values[add_slot1[read_idx]]; }
  if (sub_slot[read_idx] >= 0) { index -= d_values[sub_slot[read_idx]]; }

  auto const idx = static_cast<size_type>(index);
  out[read_idx]  = read_offset_value(offsets_ptrs[read_idx], idx, type_sizes[read_idx]);
}

/**
 * @brief Upload plan descriptors, device-chain gather gens, one pageable D2H sync.
 *
 * Pin staging of these tiny descriptors measured worse — keep pageable.
 */
std::vector<int64_t> execute_slicing_plan(slicing_plan const& plan, cuda::stream_ref stream)
{
  auto const num_reads = static_cast<size_type>(plan.reads.size());
  if (num_reads == 0) { return {}; }

  std::vector<void const*> offsets_ptrs(num_reads);
  std::vector<size_type> absolute_indices(num_reads);
  std::vector<int32_t> add0(num_reads);
  std::vector<int32_t> add1(num_reads);
  std::vector<int32_t> sub(num_reads);
  std::vector<size_type> type_sizes(num_reads);
  for (size_type i = 0; i < num_reads; ++i) {
    offsets_ptrs[i]     = plan.reads[i].offsets_data;
    absolute_indices[i] = plan.reads[i].absolute_index;
    add0[i]             = plan.reads[i].add_slot0;
    add1[i]             = plan.reads[i].add_slot1;
    sub[i]              = plan.reads[i].sub_slot;
    type_sizes[i]       = plan.reads[i].type_size;
  }

  auto const scratch_mr = cudf::get_current_device_resource_ref();
  auto d_offsets_ptrs   = cudf::detail::make_device_uvector_async(offsets_ptrs, stream, scratch_mr);
  auto d_absolute   = cudf::detail::make_device_uvector_async(absolute_indices, stream, scratch_mr);
  auto d_add0       = cudf::detail::make_device_uvector_async(add0, stream, scratch_mr);
  auto d_add1       = cudf::detail::make_device_uvector_async(add1, stream, scratch_mr);
  auto d_sub        = cudf::detail::make_device_uvector_async(sub, stream, scratch_mr);
  auto d_type_sizes = cudf::detail::make_device_uvector_async(type_sizes, stream, scratch_mr);
  rmm::device_uvector<int64_t> d_values(num_reads, stream, scratch_mr);

  size_type gen_begin = 0;
  for (size_type gen_end : plan.generation_ends) {
    auto const n = gen_end - gen_begin;
    if (n > 0) {
      cudf::detail::grid_1d const config(n, batch_block_size);
      gather_offset_values_indirect_kernel<<<config.num_blocks,
                                             config.num_threads_per_block,
                                             0,
                                             stream.get()>>>(d_offsets_ptrs.data() + gen_begin,
                                                             d_absolute.data() + gen_begin,
                                                             d_add0.data() + gen_begin,
                                                             d_add1.data() + gen_begin,
                                                             d_sub.data() + gen_begin,
                                                             d_type_sizes.data() + gen_begin,
                                                             d_values.data(),
                                                             d_values.data() + gen_begin,
                                                             n);
    }
    gen_begin = gen_end;
  }
  CUDF_CUDA_TRY(cudaGetLastError());

  return cudf::detail::make_std_vector(device_span<int64_t const>{d_values.data(), d_values.size()},
                                       stream);
}

bool planned_source_contributes_child(planned_source const& src)
{
  if (src.storage->num_children() == 0) { return false; }
  if (src.exact) { return src.size_const > 0; }
  // Non-exact sources may still be empty after resolve; plan conservatively when storage has a
  // child. Matching fill's skip of known-empty exact sources is enough for shape agreement when
  // the shape pass used the same storage-side existence check.
  return true;
}

/// Whether one root forces the slicing plan; the per-column test of `roots_need_slicing_plan`.
bool root_needs_slicing_plan(column_view const& col)
{
  if (col.offset() != 0) { return true; }
  auto const tid = col.type().id();
  if (tid == type_id::LIST && col.size() > 0 && col.num_children() > 0) {
    return col.size() < borrowed_child(col, lists_column_view::offsets_column_index).size() - 1;
  }
  return tid == type_id::STRUCT &&
         std::any_of(col.child_begin(), col.child_end(), [&](column_view const& child) {
           return child.size() != col.size();
         });
}

/**
 * @brief True when any root is sliced: a nonzero offset, a list whose offsets reach past its rows,
 * or a struct whose children differ from it in size.
 *
 * Slicing nested below an unsliced root does not count; the depth-first walk handles it, possibly
 * with a synchronization.
 */
template <typename ColRef>
bool roots_need_slicing_plan_impl(host_span<host_span<ColRef const> const> roots)
{
  for (auto const& cols : roots) {
    for (auto const& c : cols) {
      auto const& col = level_col_view(c);
      if (col.offset() != 0) { return true; }
      auto const tid = col.type().id();
      if (tid == type_id::LIST && col.size() > 0 && col.num_children() > 0) {
        column_view offsets_col = borrowed_child(col, lists_column_view::offsets_column_index);
        if (col.size() < offsets_col.size() - 1) { return true; }
      } else if (tid == type_id::STRUCT) {
        for (size_type child_idx = 0; child_idx < col.num_children(); ++child_idx) {
          auto const& child = *(col.child_begin() + child_idx);
          if (col.size() != child.size()) { return true; }
        }
      }
    }
  }
  return false;
}

bool roots_need_slicing_plan(host_span<host_span<column_view const* const> const> roots)
{
  return roots_need_slicing_plan_impl<column_view const*>(roots);
}

/**
 * @brief Reads one element from each of many offsets columns.
 *
 * @param offsets_ptrs Data pointer of each offsets column
 * @param indices Element to read from each offsets column
 * @param type_sizes Element width of each offsets column, 4 or 8 bytes
 * @param out Gathered values, widened to 64 bits
 * @param num_reads Number of reads
 */
CUDF_KERNEL void gather_offset_values_kernel(void const* const* __restrict__ offsets_ptrs,
                                             size_type const* __restrict__ indices,
                                             size_type const* __restrict__ type_sizes,
                                             int64_t* __restrict__ out,
                                             size_type num_reads)
{
  auto const read_idx = cudf::detail::grid_1d::global_thread_id();
  if (read_idx >= static_cast<thread_index_type>(num_reads)) { return; }

  auto const index = indices[read_idx];
  out[read_idx]    = read_offset_value(offsets_ptrs[read_idx], index, type_sizes[read_idx]);
}

/**
 * @brief Reads every batched offset value in one kernel and one synchronization.
 *
 * This is the device round trip that the fast paths in `collect_level_info_recursive` exist to
 * avoid; it runs only for the sources that could not be resolved on the host.
 */
std::vector<int64_t> gather_deferred_reads(deferred_bounds const& deferred, cuda::stream_ref stream)
{
  auto const num_reads = static_cast<size_type>(deferred.reads.size());

  std::vector<void const*> offsets_ptrs(num_reads);
  std::vector<size_type> indices(num_reads);
  std::vector<size_type> type_sizes(num_reads);
  for (size_type i = 0; i < num_reads; ++i) {
    offsets_ptrs[i] = deferred.reads[i].offsets_data;
    indices[i]      = deferred.reads[i].index;
    type_sizes[i]   = deferred.reads[i].type_size;
  }

  auto const scratch_mr = cudf::get_current_device_resource_ref();

  // This read is the first device work the call issues and it ends in a synchronize, so the block
  // is taken and released against a stream with nothing outstanding either side of it
  pinned_metadata staging{
    3 * alignof(std::max_align_t) +
      static_cast<std::size_t>(num_reads) * (sizeof(void const*) + 2 * sizeof(size_type)),
    stream};

  auto d_offsets_ptrs = staging.upload(offsets_ptrs, stream, scratch_mr);
  auto d_indices      = staging.upload(indices, stream, scratch_mr);
  auto d_type_sizes   = staging.upload(type_sizes, stream, scratch_mr);
  rmm::device_uvector<int64_t> d_values(num_reads, stream, scratch_mr);

  cudf::detail::grid_1d const config(num_reads, batch_block_size);
  gather_offset_values_kernel<<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
    d_offsets_ptrs.data(), d_indices.data(), d_type_sizes.data(), d_values.data(), num_reads);

  return cudf::detail::make_std_vector(device_span<int64_t const>{d_values.data(), d_values.size()},
                                       stream);
}

/**
 * @brief Fills in every size that depends on a source column's child bounds.
 *
 * Every span comes from the device, gathered in one batched round trip rather than one per source.
 *
 * @param deferred Bounds recorded during metadata collection
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return Gathered offset values (indexed by read slot); empty when there were no reads
 */
std::vector<int64_t> resolve_deferred_bounds(deferred_bounds& deferred, cuda::stream_ref stream)
{
  if (deferred.targets.empty() && deferred.reads.empty()) { return {}; }

  auto const values =
    deferred.reads.empty() ? std::vector<int64_t>{} : gather_deferred_reads(deferred, stream);

  // A level's targets are recorded contiguously and in column order, so one sweep per level
  // rebuilds its cumulative partition offsets. Columns without a target are empty and add nothing.
  apply_bounds_targets(
    host_span<pending_child_bounds const>{deferred.targets.data(), deferred.targets.size()},
    host_span<int64_t const>{values.data(), values.size()});

  return values;
}

/**
 * @brief Builds the per-request null-count launch arrays and the widest word count.
 */
void fill_null_count_arrays(deferred_null_counts const& deferred,
                            std::vector<bitmask_type const*>& masks,
                            std::vector<size_type>& begins,
                            std::vector<size_type>& ends,
                            size_type& max_words)
{
  auto const num = static_cast<size_type>(deferred.requests.size());
  max_words      = 0;
  for (size_type i = 0; i < num; ++i) {
    auto const& r = deferred.requests[i];
    masks[i]      = r.mask;
    begins[i]     = r.begin_bit;
    ends[i]       = r.end_bit;
    max_words = std::max(max_words, cudf::util::div_rounding_up_safe(r.end_bit - r.begin_bit, 32));
  }
}

/**
 * @brief Adds each readback popcount back into its level's totals and clears the requests.
 */
void accumulate_null_counts(deferred_null_counts& deferred, device_span<size_type const> set_counts)
{
  auto const num = static_cast<size_type>(deferred.requests.size());
  for (size_type i = 0; i < num; ++i) {
    auto const nulls =
      (deferred.requests[i].end_bit - deferred.requests[i].begin_bit) - set_counts[i];
    auto& level = (*deferred.requests[i].levels)[deferred.requests[i].level_idx];
    level.total_null_count += nulls;
    if (nulls > 0) { level.has_nulls = true; }
  }
  deferred.requests.clear();
}

// ============================================================================
// 5. Walkers
// DFS and BFS metadata collection plus the recursive root checker.
// ============================================================================

/**
 * @brief Returns the `index`-th child of a struct column restricted to the parent's row range.
 *
 * `structs_column_view::get_sliced_child` recomputes the child's null count with a device
 * reduction whenever the child is nullable, and reads `child(index)` repeatedly -- each read
 * deep-copying that child's whole subtree of views. When the parent covers the child exactly --
 * the common case, since only slicing makes them differ -- the child view is already the answer
 * and its stored null count is exact.
 *
 * Takes the parent as a `column_view` rather than a `structs_column_view` on purpose: converting
 * copies the parent, and with it every descendant view, once per child visited.
 *
 * When @p nulls is non-null (sliced-sync reduction), the slow path synthesizes the child view with
 * placeholder null_count 0 and records a range-popcount request instead of synchronizing.
 *
 * @param parent The struct column to take a child from
 * @param index Index of the child
 * @param stream CUDA stream used for the fallback null count
 * @param nulls Optional deferred null-count batch; null keeps the syncing fallback
 * @param child_levels Level vector the deferred count should update (required when @p nulls set)
 * @param child_level_idx Preorder slot of the child level (required when @p nulls set)
 * @return The child restricted to the parent's rows
 */
column_view sliced_struct_child(column_view const& parent,
                                size_type index,
                                cuda::stream_ref stream,
                                deferred_null_counts* nulls           = nullptr,
                                std::vector<level_info>* child_levels = nullptr,
                                size_type child_level_idx             = 0)
{
  auto const& child = *(parent.child_begin() + index);
  if (parent.offset() == 0 && parent.size() == child.size()) { return child; }

  if (nulls == nullptr) { return structs_column_view{parent}.get_sliced_child(index, stream); }

  // Mirrors structs_column_view::get_sliced_child exactly, minus the count: the parent's offset
  // replaces the child's (struct invariant), and grandchildren are left unsliced for the next
  // level to re-slice.
  auto view = column_view{child.type(),
                          parent.size(),
                          child.head(),
                          child.null_mask(),
                          0,
                          parent.offset(),
                          std::vector<column_view>(child.child_begin(), child.child_end())};
  nulls->add(child.null_mask(),
             parent.offset(),
             parent.offset() + parent.size(),
             child_levels,
             child_level_idx);
  return view;
}

/**
 * @brief Returns a list column's child restricted to the parent's row range.
 *
 * The same shape as `sliced_struct_child` one level up: `lists_column_view::get_sliced_child` only
 * needs the device when the parent does not cover its child exactly, and building the typed view
 * to ask costs a copy of the parent's whole subtree before the question is even put. Reads the
 * offsets and the child straight out of the parent instead, and falls back to the typed route on
 * the sliced path, where the device round trip dominates the copy anyway.
 *
 * @param parent The list column to take the child from
 * @param stream CUDA stream used for the sliced path's offset reads
 * @return The child restricted to the parent's rows
 */
column_view sliced_list_child(column_view const& parent, cuda::stream_ref stream)
{
  CUDF_EXPECTS(parent.num_children() == lists_column_view::child_column_index + 1,
               "lists column has an incorrect number of children");
  auto const& offsets = borrowed_child(parent, lists_column_view::offsets_column_index);
  auto const& child   = borrowed_child(parent, lists_column_view::child_column_index);
  // `get_sliced_child`'s own fast path, with the same two conditions in the same order, plus a
  // structural one: a child longer than the offsets' reach (make_lists_column permits it) is not
  // exactly covered by the parent even at offset 0 — the unreachable tail must not leak into the
  // output. child.size() == offsets.size() - 1 is the strongest host-only proof of exactness:
  // every offset entry consumed, no tail.
  if (parent.offset() == 0 && parent.size() >= offsets.size() - 1 &&
      child.size() == offsets.size() - 1) {
    return child;
  }
  return lists_column_view{parent}.get_sliced_child(stream);
}

/**
 * @brief Whether a list parent span already covers its child exactly (host-only; same test as the
 * `sliced_list_child` fast path).
 */
bool list_span_covers_child_exactly(column_view const& storage, size_type offset, size_type size)
{
  if (storage.num_children() <= lists_column_view::child_column_index) { return false; }
  auto const& offsets = borrowed_child(storage, lists_column_view::offsets_column_index);
  auto const& child   = borrowed_child(storage, lists_column_view::child_column_index);
  // `size >= offsets.size() - 1` alone is only necessary: make_lists_column permits a child
  // longer than the offsets reach, and the unreachable tail must not leak into the output.
  // child.size() == offsets.size() - 1 is the strongest host-only proof of exact cover.
  return offset == 0 && size >= offsets.size() - 1 && child.size() == offsets.size() - 1;
}

/**
 * @brief Whether a list parent already covers its child exactly (host-only; same test as the
 * `sliced_list_child` fast path).
 */
bool list_parent_covers_child_exactly(column_view const& parent)
{
  return list_span_covers_child_exactly(parent, parent.offset(), parent.size());
}

/**
 * @brief Host-build a list child span from already-gathered offset endpoints.
 */
column_view host_sliced_list_child(column_view const& parent,
                                   size_type child_begin,
                                   size_type child_end,
                                   deferred_null_counts* nulls,
                                   std::vector<level_info>* child_levels,
                                   size_type child_level_idx)
{
  auto const& child = borrowed_child(parent, lists_column_view::child_column_index);
  auto const size   = child_end - child_begin;
  if (child_begin == 0 && size == child.size()) { return child; }

  auto view = column_view{child.type(),
                          size,
                          child.head(),
                          child.null_mask(),
                          0,
                          child_begin,
                          std::vector<column_view>(child.child_begin(), child.child_end())};
  nulls->add(child.null_mask(), child_begin, child_end, child_levels, child_level_idx);
  return view;
}

/**
 * @brief Materialize list children for DFS recurse with one deferred-bounds offset batch.
 */
std::vector<column_view> materialize_list_children_dfs(host_span<column_view const* const> parents,
                                                       cuda::stream_ref stream,
                                                       deferred_null_counts* nulls,
                                                       std::vector<level_info>* levels,
                                                       size_type child_level_idx)
{
  std::vector<column_view> child_cols;
  child_cols.reserve(parents.size());

  deferred_bounds local;
  struct item {
    column_view const* parent;
    int32_t first_slot;
    int32_t last_slot;
    bool exact;
  };
  std::vector<item> items;
  items.reserve(parents.size());
  for (auto const* p : parents) {
    if (list_parent_covers_child_exactly(*p)) {
      items.push_back({p, -1, -1, true});
    } else {
      auto const& offsets   = borrowed_child(*p, lists_column_view::offsets_column_index);
      auto const first_slot = p->offset() > 0 ? local.add_read(offsets, p->offset()) : -1;
      auto const last_slot  = local.add_read(offsets, p->offset() + p->size());
      items.push_back({p, first_slot, last_slot, false});
    }
  }

  auto const values =
    local.reads.empty() ? std::vector<int64_t>{} : gather_deferred_reads(local, stream);
  for (auto const& it : items) {
    if (it.exact) {
      child_cols.push_back(borrowed_child(*it.parent, lists_column_view::child_column_index));
    } else {
      auto const first =
        it.first_slot < 0 ? size_type{0} : static_cast<size_type>(values[it.first_slot]);
      auto const last = static_cast<size_type>(values[it.last_slot]);
      child_cols.push_back(
        host_sliced_list_child(*it.parent, first, last, nulls, levels, child_level_idx));
    }
  }
  return child_cols;
}

/**
 * @brief Host proof that an exact-cover list parent already has a nonempty child.
 */
bool list_source_proves_nonempty_child(column_view const& storage,
                                       size_type offset,
                                       size_type size,
                                       column_view const& child)
{
  return list_span_covers_child_exactly(storage, offset, size) && child.size() > 0;
}

/**
 * @brief Record list child dtype; omit unused empty_like schema when the source proves the child
 *        level nonempty.
 *
 * Reconstruct only reads empty-child schema when `total_child_size == 0`: for an exact-cover list
 * parent (`offset==0` and full offsets span) a nonzero `child.size()` proves the concatenated
 * child level is nonempty, so the nested `column_view` copy into `empty_child_like` is dead.
 * Sliced / non-exact parents still capture schema.
 */
void store_list_empty_child_schema(level_info& info,
                                   column_view const& storage,
                                   size_type offset,
                                   size_type size,
                                   column_view const& child)
{
  info.child_dtype = child.type();
  if (list_source_proves_nonempty_child(storage, offset, size, child)) { return; }
  info.empty_child_like = child;
}

/**
 * @brief Host bounds and type checks for one nesting level (no recursion).
 *
 * Called from the fused collect walk at every level. Mirrors the non-recursive body of
 * `batch_bounds_and_type_check`, including the same overflow and support messages.
 *
 * @param cols Columns of one nesting level
 * @return true if this level is EMPTY (caller should not collect further)
 */

/**
 * @brief Recursively collects all level information in a single pass, running the bounds and type
 *        checks at each level so the schema is not walked twice and each child view is
 *        materialised once.
 *
 * Level emission order is preorder (this level, then struct children left-to-right, then the list
 * child). Reconstruct and the superimpose walk depend on that order; do not change it.
 */
/**
 * @brief Zero the per-level counters and record the level type flags.
 *
 * Shared by the two walkers (`collect_level_info_recursive`, `fill_level_info_at_slot`);
 * callers keep their own dictionary delegation loop and level emission.
 */
void initialize_level(level_info& info, data_type dtype, size_type num_children)
{
  info.dtype                   = dtype;
  info.is_struct               = (dtype.id() == type_id::STRUCT);
  info.is_string               = (dtype.id() == type_id::STRING);
  info.is_list                 = (dtype.id() == type_id::LIST);
  info.is_dictionary           = (dtype.id() == type_id::DICTIONARY32);
  info.delegate_to_specialized = info.is_dictionary;
  info.num_children            = num_children;
  info.has_nulls               = false;
  info.total_rows              = 0;
  info.total_null_count        = 0;
  info.total_child_size        = 0;
}

/**
 * @brief Reserve per-level buffers from the source count and compute the element size.
 *
 * Fixed-width levels get their element size; struct/string/list/dictionary levels get 0. The
 * string and list branches reserve their offsets buffers; non-nested levels reserve the data
 * buffers. Returns the element size for the per-column scan.
 */
std::size_t reserve_level_buffers(level_info& info, size_type num_cols)
{
  std::size_t const element_size =
    (info.is_struct || info.is_string || info.is_list || info.is_dictionary)
      ? 0
      : cudf::size_of(info.dtype);

  info.mask_ptrs.reserve(num_cols);
  info.mask_offsets.reserve(num_cols);
  info.col_sizes.reserve(num_cols);

  if (!info.is_struct && !info.is_string && !info.is_list) {
    info.src_data_ptrs.reserve(num_cols);
    info.data_sizes.reserve(num_cols);
  }
  if (info.is_string || info.is_list) {
    info.src_offsets_ptrs.reserve(num_cols);
    info.src_offsets_offsets.reserve(num_cols);
    info.chars_partition_offsets.reserve(num_cols + 1);
    info.chars_partition_offsets.push_back(0);
    info.first_src_offsets.reserve(num_cols);
  }
  if (info.is_string) {
    info.src_data_ptrs.reserve(num_cols);
    info.data_sizes.reserve(num_cols);
    info.offsets_dtype = data_type{type_id::INT32};
  }
  if (info.is_list) { info.offsets_dtype = data_type{type_id::INT32}; }
  return element_size;
}

/**
 * @brief Shared body of the two `collect_level_info_recursive` walkers, templated over the
 *        source-column reference shape: `column_view` (value roots) or `column_view const*`
 *        (borrowed pointer roots). Per-column access goes through `level_col_view`, so a
 *        single body serves both shapes; `ColRef` affects only how child pointers are taken
 *        (uniform through `&level_col_view(...)`).
 *
 * @param cols Columns of one nesting level
 */

template <typename ColRef>
void collect_level_info_recursive(host_span<ColRef const> cols,
                                  std::vector<level_info>& levels,
                                  deferred_bounds& deferred,
                                  cuda::stream_ref stream,
                                  deferred_null_counts* nulls)
{
  if (cols.empty()) return;

  if (check_bounds_and_types_at_level_impl<ColRef>(cols)) { return; }

  auto const this_level_idx = static_cast<size_type>(levels.size());

  level_info info;
  auto const& first_col = level_col_view(cols.front());
  initialize_level(info, first_col.type(), first_col.num_children());

  auto const num_cols            = cols.size();
  std::size_t const element_size = reserve_level_buffers(info, num_cols);

  // Handle DICTIONARY delegation - collect columns and return early
  if (info.delegate_to_specialized) {
    // Collect basic mask info for all columns first
    for (auto const& c : cols) {
      auto const& col     = level_col_view(c);
      auto const col_size = col.size();
      info.total_rows += col_size;
      info.col_sizes.push_back(col_size);
      info.mask_ptrs.push_back(col.null_mask());
      info.mask_offsets.push_back(col.offset());
      info.total_null_count += col.null_count();
      if (col.has_nulls()) { info.has_nulls = true; }
      info.original_columns.push_back(col);
    }
    levels.push_back(std::move(info));
    return;  // Don't recurse - specialized implementation handles everything
  }

  if (info.is_string) {
    // Get offsets type from the first non-empty column's offsets child; empty string columns
    // have no children
    for (auto const& c : cols) {
      auto const& col = level_col_view(c);
      if (col.size() > 0 && col.num_children() > 0) {
        info.offsets_dtype = borrowed_child(col, strings_column_view::offsets_column_index).type();
        break;
      }
    }
  }
  if (info.is_list) {
    // Store child info for empty list reconstruction
    for (auto const& c : cols) {
      auto const& col = level_col_view(c);
      // Check num_children() > 0 to avoid crash on malformed columns
      if (col.size() > 0 && col.num_children() > 0) {
        auto const& child = borrowed_child(col, lists_column_view::child_column_index);
        store_list_empty_child_schema(info, col, col.offset(), col.size(), child);
        break;
      }
    }
  }

  // Single pass over columns to collect all info
  for (auto const& c : cols) {
    auto const& col     = level_col_view(c);
    auto const col_size = col.size();
    info.total_rows += col_size;
    info.col_sizes.push_back(col_size);

    // Mask info
    info.mask_ptrs.push_back(col.null_mask());
    info.mask_offsets.push_back(col.offset());
    info.total_null_count += col.null_count();
    if (col.has_nulls()) { info.has_nulls = true; }

    // Data info based on type
    if (info.is_string) {
      // String column: collect offsets and chars info
      // Note: empty string columns have no children, so we must check size AND children
      // Check num_children() > 0 to avoid crash on malformed columns
      if (col_size > 0 && col.num_children() > 0) {
        auto const& offsets_col = borrowed_child(col, strings_column_view::offsets_column_index);
        // Offsets info
        info.src_offsets_ptrs.push_back(offsets_col.head());
        info.src_offsets_offsets.push_back(col.offset());

        // The chars bounds live on the device; batch the reads and fill the sizes in later
        {
          auto const first_slot =
            col.offset() > 0 ? deferred.add_read(offsets_col, col.offset()) : -1;
          auto const last_slot = deferred.add_read(offsets_col, col.offset() + col_size);
          deferred.targets.push_back({&levels,
                                      this_level_idx,
                                      static_cast<size_type>(info.col_sizes.size() - 1),
                                      first_slot,
                                      last_slot});
        }
        // Placeholders; the chars pointer is advanced by the first offset once it is known
        info.src_data_ptrs.push_back(col_head<char>(col));
        info.data_sizes.push_back(0);
        info.chars_partition_offsets.push_back(0);
        info.first_src_offsets.push_back(0);
      } else {
        // Empty column - push placeholders
        info.src_offsets_ptrs.push_back(nullptr);
        info.src_offsets_offsets.push_back(0);
        info.src_data_ptrs.push_back(nullptr);
        info.data_sizes.push_back(0);
        info.chars_partition_offsets.push_back(info.total_child_size);
        info.first_src_offsets.push_back(0);
      }
    } else if (info.is_list) {
      // List column: collect offsets info (child data handled via recursion)
      // Check num_children() > 0 to avoid crash on malformed columns
      if (col_size > 0 && col.num_children() > 0) {
        auto const& offsets_col = borrowed_child(col, lists_column_view::offsets_column_index);
        // Offsets info
        info.src_offsets_ptrs.push_back(offsets_col.head());
        info.src_offsets_offsets.push_back(col.offset());

        // The child boundaries live on the device, so batch the reads and fill the sizes in
        // later. They cannot be established host-side: `child().size()` is only equal to
        // `offsets[size()]` when the child is exactly as long as the offsets reach, and
        // nothing in a column_view records that. A child may legally run past its last offset
        // — `make_lists_column` neither requires nor checks otherwise — and consumers that
        // bound their reads by the offsets, `cudf::concatenate` among them, handle that
        // correctly while a copy length taken from `child().size()` would pull the extra rows
        // into the output. See BatchConcatenateTest.ListChildLongerThanOffsetsSpan.
        {
          auto const col_pos = static_cast<size_type>(info.col_sizes.size() - 1);
          auto const first_slot =
            col.offset() > 0 ? deferred.add_read(offsets_col, col.offset()) : -1;
          auto const last_slot = deferred.add_read(offsets_col, col.offset() + col_size);
          deferred.targets.push_back({&levels, this_level_idx, col_pos, first_slot, last_slot});
        }
        info.first_src_offsets.push_back(0);
        info.chars_partition_offsets.push_back(0);
      } else {
        // Empty column - push placeholders
        info.src_offsets_ptrs.push_back(nullptr);
        info.src_offsets_offsets.push_back(0);
        info.first_src_offsets.push_back(0);
        info.chars_partition_offsets.push_back(info.total_child_size);
      }
    } else if (!info.is_struct) {
      // Fixed-width type
      info.src_data_ptrs.push_back(col_head<char>(col) + (col.offset() * element_size));
      info.data_sizes.push_back(col_size * element_size);
    }
  }

  levels.push_back(std::move(info));

  // Recurse into struct children
  if (first_col.type().id() == type_id::STRUCT) {
    auto const num_children = first_col.num_children();
    std::vector<column_view> child_cols;
    std::vector<column_view const*> child_ptrs;
    child_cols.reserve(num_cols);
    child_ptrs.reserve(num_cols);

    for (size_type child_idx = 0; child_idx < num_children; ++child_idx) {
      auto const child_level_idx = static_cast<size_type>(levels.size());
      bool all_exact             = true;
      for (auto const& c : cols) {
        auto const& col   = level_col_view(c);
        auto const& child = borrowed_child(col, child_idx);
        if (!(col.offset() == 0 && col.size() == child.size())) {
          all_exact = false;
          break;
        }
      }
      if (all_exact) {
        child_ptrs.clear();
        for (auto const& c : cols) {
          auto const& col = level_col_view(c);
          child_ptrs.push_back(&borrowed_child(col, child_idx));
        }
        collect_level_info_recursive(
          host_span<column_view const* const>{child_ptrs.data(), child_ptrs.size()},
          levels,
          deferred,
          stream,
          nulls);
        continue;
      }
      child_cols.clear();
      for (auto const& c : cols) {
        auto const& col = level_col_view(c);
        child_cols.push_back(
          sliced_struct_child(col, child_idx, stream, nulls, &levels, child_level_idx));
      }
      collect_level_info_recursive(
        host_span<column_view const>{child_cols.data(), child_cols.size()},
        levels,
        deferred,
        stream,
        nulls);
    }
  }

  // Recurse into list children
  if (first_col.type().id() == type_id::LIST) {
    // Probe + materialize non-exact list children, then recurse into the surviving child span
    // (the fallback for a level with any non-exact parent).
    bool all_exact = true;
    for (auto const& c : cols) {
      auto const& col = level_col_view(c);
      if (!(col.size() > 0 && col.num_children() > 0)) { continue; }
      if (!list_parent_covers_child_exactly(col)) {
        all_exact = false;
        break;
      }
    }
    if (all_exact) {
      std::vector<column_view const*> child_ptrs;
      child_ptrs.reserve(num_cols);
      for (auto const& c : cols) {
        auto const& col = level_col_view(c);
        if (col.size() > 0 && col.num_children() > 0) {
          child_ptrs.push_back(&borrowed_child(col, lists_column_view::child_column_index));
        }
      }
      if (!child_ptrs.empty()) {
        collect_level_info_recursive(
          host_span<column_view const* const>{child_ptrs.data(), child_ptrs.size()},
          levels,
          deferred,
          stream,
          nulls);
      }
    } else {
      // Some parent does not cover its child exactly: materialize the sliced children and recurse
      std::vector<column_view> child_cols;
      child_cols.reserve(num_cols);
      // Only recurse if there are non-empty columns; an all-empty level contributes no child level.
      {
        std::vector<column_view const*> nonempty;
        nonempty.reserve(cols.size());
        for (auto const& c : cols) {
          auto const& col = level_col_view(c);
          if (col.size() > 0 && col.num_children() > 0) { nonempty.push_back(&col); }
        }
        if (!nonempty.empty()) {
          child_cols = materialize_list_children_dfs(
            span_of(nonempty), stream, nulls, &levels, static_cast<size_type>(levels.size()));
        }
      }
      if (!child_cols.empty()) {
        collect_level_info_recursive(
          host_span<column_view const>{child_cols.data(), child_cols.size()},
          levels,
          deferred,
          stream,
          nulls);
      }
    }
  }
}

/**
 * @brief Verifies bounds and type compatibility for batch concatenation.
 *
 * @param cols Source columns of one level
 * @param stream CUDA stream used for device memory operations
 * @param views Whether to borrow nested columns' children or copy them through the typed views
 * @param check_types Whether to compare types; `cudf::all_have_same_types` already compares the
 *        whole nested type tree, so only the outermost call needs to, and repeating it per level
 *        would rewalk each subtree once per ancestor
 * @return Number of levels visited, which `collect_level_info_recursive` will produce for the same
 *         columns. Used only to reserve, so an inexact count stays correct.
 */
void batch_bounds_and_type_check(host_span<column_view const> cols,
                                 cuda::stream_ref stream,
                                 bool check_types)
{
  // Check total row count doesn't exceed size_type limit, then EMPTY all-or-nothing and the
  // support set; the root type tree compare and recursion below stay root-specific.
  check_bounds_and_types_at_level_impl<column_view, false, false>(cols);

  // Check all types match
  if (check_types) {
    // `cudf::all_have_same_types` captures its reference column BY VALUE -- `[want = *first]` --
    // which deep-copies that column's whole subtree of views before a single comparison is made.
    // `have_same_types` only reads the type tree, so a reference answers the same question.
    auto const same_types = [&] {
      if (cols.empty()) { return true; }
      auto const& want = cols.front();
      return std::all_of(std::next(cols.begin()), cols.end(), [&want](auto const& c) {
        return cudf::have_same_types(want, c);
      });
    }();
    expect_type_match(same_types);
  }

  // Recursively check struct children
  if (cols.front().type().id() == type_id::STRUCT) {
    auto const num_children = cols.front().num_children();
    std::vector<column_view> nth_children;
    nth_children.reserve(cols.size());

    for (size_type child_idx = 0; child_idx < num_children; ++child_idx) {
      {
        nth_children.clear();
        for (auto const& col : cols) {
          nth_children.push_back(sliced_struct_child(col, child_idx, stream));
        }
      }
      batch_bounds_and_type_check(span_of(nth_children), stream, false);
    }
  }

  // Check string offset overflow
  if (cols.front().type().id() == type_id::STRING) { check_level_offset_overflow(cols); }

  // Recursively check list children
  if (cols.front().type().id() == type_id::LIST) {
    // Check offset overflow
    check_level_offset_overflow(cols);

    // Recursively check children
    std::vector<column_view> child_cols;
    child_cols.reserve(cols.size());
    for (auto const& col : cols) {
      if (col.size() > 0) { child_cols.push_back(sliced_list_child(col, stream)); }
    }
    if (!child_cols.empty()) {
      batch_bounds_and_type_check(
        host_span<column_view const>{child_cols.data(), child_cols.size()}, stream, false);
    }
  }
}

/**
 * @brief Shared root walk of the two table-entry collect paths for one column position.
 *
 * Gathers per-table roots as borrowed pointers into the live table columns (`ptr_storage` backs
 * the returned span, so slicing-plan callers use it as their root store), skips EMPTY-typed
 * positions, and for all-empty positions runs the bounds check and skips collection. Returns
 * false when the position was skipped. `on_root` receives
 * `host_span<column_view const* const>` only when the position survives.
 */
template <typename RootFn>
bool collect_table_entry_roots(host_span<table_view const> tables,
                               size_type col_idx,
                               size_type num_tables,
                               std::vector<column_view const*>& ptr_storage,
                               cuda::stream_ref stream,
                               RootFn&& on_root)
{
  ptr_storage.clear();
  ptr_storage.reserve(num_tables);
  for (size_type t = 0; t < num_tables; ++t) {
    ptr_storage.push_back(&tables[t].column(col_idx));
  }
  host_span<column_view const* const> roots{ptr_storage.data(), ptr_storage.size()};
  if (is_empty_position(roots, roots.front())) {
    bounds_check_ptrs(roots, stream);
    return false;
  }
  on_root(roots);
  return true;
}

/**
 * @brief Collects level_info with deferred struct null counts and a single slicing sync.
 *
 * Unsliced roots take the depth-first walk; sliced roots take the BFS plan/fill path below.
 *
 * Plan path: host shape + plan records every list/string endpoint read with device indirection
 * across list-nesting generations. Kernels chain on the GPU; exactly one D2H/`make_std_vector`
 * sync brings all values back. List children and level bounds are then filled on the host with no
 * further slicing synchronizations.
 *
 * @param resolve_nulls When true (default), drains deferred null counts before returning. Table
 * concat passes false so it can share one stream sync with the superimpose decision readback.
 */
/**
 * @brief Lightweight host view for the slicing-plan fill walk.
 *
 * Points at borrowed storage for the type tree / buffers and carries the row range separately.
 * Avoids deep-copying nested `column_view` children on every sliced struct/list expansion.
 * Null counts are taken from storage only when the range matches exactly; otherwise the plan or
 * fill records a deferred range popcount and leaves `null_count == 0` here.
 */
struct fill_view {
  column_view const* storage;  ///< Never null; owns the child vector we borrow
  size_type offset;
  size_type size;

  [[nodiscard]] bool counts_known() const
  {
    return offset == storage->offset() && size == storage->size();
  }
  [[nodiscard]] data_type type() const { return storage->type(); }
  [[nodiscard]] size_type num_children() const { return storage->num_children(); }
  [[nodiscard]] bitmask_type const* null_mask() const { return storage->null_mask(); }
  [[nodiscard]] size_type null_count() const { return counts_known() ? storage->null_count() : 0; }
  [[nodiscard]] bool has_nulls() const { return counts_known() && storage->has_nulls(); }
  template <typename T = char>
  [[nodiscard]] T const* head() const
  {
    return storage->head<T>();
  }
};

size_type level_col_size(fill_view const& c) { return c.size; }

column_view const& level_col_view(fill_view const& c) { return *c.storage; }

/**
 * @brief Fills one preassigned level from borrowed fill_views (no deep column_view copies).
 *
 * Uses known null counts only when the fill range matches storage exactly; sliced ranges rely on
 * deferred nulls already recorded by the plan (exact parents) or by list-child materialization.
 */
void fill_level_info_at_slot(host_span<fill_view const> cols,
                             std::vector<level_info>& levels,
                             size_type slot)
{
  if (cols.empty()) return;

  if (check_bounds_and_types_at_level_impl<fill_view, true>(cols)) { return; }

  level_info info;
  auto const& first_col = cols.front();
  initialize_level(info, first_col.type(), first_col.num_children());

  auto const num_cols            = cols.size();
  std::size_t const element_size = reserve_level_buffers(info, num_cols);

  if (info.delegate_to_specialized) {
    for (auto const& col : cols) {
      info.total_rows += col.size;
      info.col_sizes.push_back(col.size);
      info.mask_ptrs.push_back(col.null_mask());
      info.mask_offsets.push_back(col.offset);
      info.total_null_count += col.null_count();
      if (col.has_nulls()) { info.has_nulls = true; }
      // Dictionary delegates need a real column_view; exact-range only in practice.
      info.original_columns.push_back(column_view{
        col.storage->type(),
        col.size,
        col.storage->head(),
        col.storage->null_mask(),
        col.null_count(),
        col.offset,
        std::vector<column_view>(col.storage->child_begin(), col.storage->child_end())});
    }
    levels[slot] = std::move(info);
    return;
  }

  if (info.is_string) {
    for (auto const& col : cols) {
      if (col.size > 0 && col.num_children() > 0) {
        info.offsets_dtype =
          borrowed_child(*col.storage, strings_column_view::offsets_column_index).type();
        break;
      }
    }
  }
  if (info.is_list) {
    // Plan/fill-only: when fill will emit a child level, dtype-only (no nested empty_child copy).
    if (std::any_of(cols.begin(), cols.end(), [](fill_view const& c) {
          return c.size > 0 && c.num_children() > 0;
        })) {
      for (auto const& col : cols) {
        if (col.size > 0 && col.num_children() > 0) {
          info.child_dtype =
            borrowed_child(*col.storage, lists_column_view::child_column_index).type();
          break;
        }
      }
    }
  }

  for (auto const& col : cols) {
    info.total_rows += col.size;
    info.col_sizes.push_back(col.size);

    info.mask_ptrs.push_back(col.null_mask());
    info.mask_offsets.push_back(col.offset);
    // Known counts only — never call syncing range null_count on sliced placeholders.
    info.total_null_count += col.null_count();
    if (col.has_nulls()) { info.has_nulls = true; }

    if (info.is_string) {
      if (col.size > 0 && col.num_children() > 0) {
        auto const& offsets_col =
          borrowed_child(*col.storage, strings_column_view::offsets_column_index);
        info.src_offsets_ptrs.push_back(offsets_col.head());
        info.src_offsets_offsets.push_back(col.offset);
        // Bounds come from the slicing plan after D2H.
        info.src_data_ptrs.push_back(col.storage->head<char>());
        info.data_sizes.push_back(0);
        info.chars_partition_offsets.push_back(0);
        info.first_src_offsets.push_back(0);
      } else {
        info.src_offsets_ptrs.push_back(nullptr);
        info.src_offsets_offsets.push_back(0);
        info.src_data_ptrs.push_back(nullptr);
        info.data_sizes.push_back(0);
        info.chars_partition_offsets.push_back(info.total_child_size);
        info.first_src_offsets.push_back(0);
      }
    } else if (info.is_list) {
      if (col.size > 0 && col.num_children() > 0) {
        auto const& offsets_col =
          borrowed_child(*col.storage, lists_column_view::offsets_column_index);
        info.src_offsets_ptrs.push_back(offsets_col.head());
        info.src_offsets_offsets.push_back(col.offset);
        info.first_src_offsets.push_back(0);
        info.chars_partition_offsets.push_back(0);
      } else {
        info.src_offsets_ptrs.push_back(nullptr);
        info.src_offsets_offsets.push_back(0);
        info.first_src_offsets.push_back(0);
        info.chars_partition_offsets.push_back(info.total_child_size);
      }
    } else if (!info.is_struct) {
      info.src_data_ptrs.push_back(col.head<char>() + (col.offset * element_size));
      info.data_sizes.push_back(col.size * element_size);
    }
  }

  levels[slot] = std::move(info);
}

void collect_levels_bfs(host_span<host_span<column_view const* const> const> roots,
                        host_span<std::vector<level_info>> levels_by_column,
                        deferred_bounds& deferred,
                        deferred_null_counts& nulls,
                        cuda::stream_ref stream,
                        bool resolve_nulls = true)
{
  auto const num_columns = static_cast<size_type>(roots.size());

  // Unsliced roots: depth-first walk with batched bound reads, no slicing plan.
  if (!roots_need_slicing_plan(roots)) {
    for (size_type col_idx = 0; col_idx < num_columns; ++col_idx) {
      auto const& cols = roots[col_idx];
      if (cols.empty() || is_empty_position(cols, cols.front())) { continue; }
      std::vector<column_view> materialized;
      materialized.reserve(cols.size());
      for (auto const* p : cols) {
        materialized.push_back(*p);
      }
      collect_level_info_recursive(
        span_of(materialized), levels_by_column[col_idx], deferred, stream, &nulls);
    }
    resolve_deferred_bounds(deferred, stream);
    resolve_deferred_null_counts(nulls, stream);
    return;
  }

  (void)deferred;  // Old-path deferred_bounds unused on the slicing-plan path.

  std::vector<level_tree_shape> shapes(num_columns);
  slicing_plan plan;

  std::vector<plan_frontier_item> frontier;
  frontier.reserve(static_cast<std::size_t>(num_columns));

  for (size_type col_idx = 0; col_idx < num_columns; ++col_idx) {
    auto const& cols = roots[col_idx];
    if (cols.empty() || is_empty_position(cols, cols.front())) { continue; }
    shapes[col_idx] = build_level_tree_shape(cols);
    levels_by_column[col_idx].resize(static_cast<std::size_t>(shapes[col_idx].num_levels));

    std::vector<planned_source> sources;
    sources.reserve(cols.size());
    for (auto const* p : cols) {
      sources.push_back(make_exact_planned(*p, 0));
    }
    frontier.push_back({&levels_by_column[col_idx], 0, col_idx, std::move(sources)});
  }

  // Reserve up front so the plan does not regrow while it expands.
  std::size_t total_sources = 0;
  for (size_type col_idx = 0; col_idx < num_columns; ++col_idx) {
    total_sources += roots[col_idx].size();
  }
  plan.reads.reserve(total_sources * 2);
  plan.targets.reserve(total_sources);
  plan.list_children.reserve(total_sources);
  plan.generation_ends.reserve(8);

  // ---- Host plan (no sync): expand structs in-generation; queue list children for next gen ----
  std::vector<plan_frontier_item> worklist;
  std::vector<plan_frontier_item> next_gen;
  size_type generation = 0;

  while (!frontier.empty()) {
    worklist = std::move(frontier);
    frontier.clear();
    next_gen.clear();

    for (std::size_t wi = 0; wi < worklist.size(); ++wi) {
      auto item         = std::move(worklist[wi]);
      auto* levels      = item.levels;
      auto const slot   = item.level_idx;
      auto const& shape = shapes[item.col_idx];
      auto const& kids  = shape.children_of[slot];
      auto& sources     = item.sources;
      if (sources.empty()) { continue; }

      auto const tid = sources.front().storage->type().id();

      if (tid == type_id::STRING || tid == type_id::LIST) {
        size_type col_pos           = 0;
        size_type list_child_source = 0;
        std::vector<planned_source> list_child_sources;

        for (auto const& src : sources) {
          // Known-empty (exact and zero rows) or childless sources record no bounds
          if ((src.exact && src.size_const == 0) || src.storage->num_children() == 0) {
            ++col_pos;
            continue;
          }

          column_view const& offsets_col =
            (tid == type_id::STRING)
              ? borrowed_child(*src.storage, strings_column_view::offsets_column_index)
              : borrowed_child(*src.storage, lists_column_view::offsets_column_index);

          auto const [first_slot, last_slot] = plan_endpoint_reads(plan, src, offsets_col);
          plan.targets.push_back({levels, slot, col_pos, first_slot, last_slot});

          if (tid == type_id::LIST && !kids.empty() && planned_source_contributes_child(src)) {
            auto const& raw_child =
              borrowed_child(*src.storage, lists_column_view::child_column_index);
            auto const verbatim =
              src.exact && src.offset_base == 0 && src.size_const >= offsets_col.size() - 1;
            // First contributor of this list level starts a contiguous run in list_children.
            auto const run_base =
              static_cast<size_type>(plan.list_children.size()) - list_child_source;
            plan.list_children.push_back({levels,
                                          kids.front(),
                                          list_child_source,
                                          run_base,
                                          &raw_child,
                                          first_slot,
                                          last_slot,
                                          verbatim});
            if (verbatim) {
              list_child_sources.push_back(make_exact_planned(raw_child, src.generation));
            } else {
              list_child_sources.push_back(
                make_span_planned(raw_child, first_slot, last_slot, src.generation));
            }
            ++list_child_source;
          }
          ++col_pos;
        }

        if (!list_child_sources.empty()) {
          next_gen.push_back({levels, kids.front(), item.col_idx, std::move(list_child_sources)});
        }
      }

      if (tid == type_id::STRUCT) {
        auto const num_children = sources.front().storage->num_children();
        for (size_type child_idx = 0; child_idx < num_children; ++child_idx) {
          std::vector<planned_source> child_sources;
          child_sources.reserve(sources.size());
          for (auto const& src : sources) {
            auto const& child_storage = borrowed_child(*src.storage, child_idx);
            if (src.exact && src.offset_base == 0 && src.size_const == child_storage.size()) {
              child_sources.push_back(make_exact_planned(child_storage, src.generation));
              continue;
            }
            if (src.exact) {
              nulls.maybe_add(child_storage,
                              src.offset_base,
                              src.offset_base + src.size_const,
                              levels,
                              kids[child_idx]);
            }
            auto child    = src;
            child.storage = &child_storage;
            child_sources.push_back(child);
          }
          worklist.push_back({levels, kids[child_idx], item.col_idx, std::move(child_sources)});
        }
      }
    }

    plan.generation_ends.push_back(static_cast<size_type>(plan.reads.size()));
    frontier = std::move(next_gen);
    ++generation;
  }

  // ---- One slicing sync (pageable D2H); plan-time nulls stay in @p nulls ----
  auto const values = execute_slicing_plan(plan, stream);

  // ---- Materialize list children, then fill ----
  std::vector<fill_view> list_child_views(plan.list_children.size());
  for (std::size_t i = 0; i < plan.list_children.size(); ++i) {
    auto const& lc = plan.list_children[i];
    if (lc.verbatim) {
      list_child_views[i] = fill_view{lc.raw_child, lc.raw_child->offset(), lc.raw_child->size()};
      continue;
    }
    auto const first =
      lc.first_slot < 0 ? int64_t{0} : values[static_cast<std::size_t>(lc.first_slot)];
    auto const last       = values[static_cast<std::size_t>(lc.last_slot)];
    auto const child_size = static_cast<size_type>(last - first);
    auto const begin      = lc.raw_child->offset() + static_cast<size_type>(first);
    nulls.maybe_add(*lc.raw_child, begin, begin + child_size, lc.levels, lc.child_level_idx);
    list_child_views[i] = fill_view{lc.raw_child, begin, child_size};
  }

  struct list_run_entry {
    std::vector<level_info>* levels;
    size_type child_level_idx;
    size_type run_base;
  };

  std::vector<list_run_entry> list_runs;
  list_runs.reserve(plan.list_children.size());
  for (auto const& lc : plan.list_children) {
    if (lc.child_source == 0) { list_runs.push_back({lc.levels, lc.child_level_idx, lc.run_base}); }
  }

  std::vector<collect_frontier_entry> fill_frontier;
  std::vector<fill_view> view_pool;
  fill_frontier.reserve(static_cast<std::size_t>(num_columns));
  std::size_t root_sources = 0;
  for (size_type col_idx = 0; col_idx < num_columns; ++col_idx) {
    root_sources += roots[col_idx].size();
  }
  view_pool.reserve(root_sources * 4 + plan.list_children.size());

  for (size_type col_idx = 0; col_idx < num_columns; ++col_idx) {
    auto const& cols = roots[col_idx];
    if (cols.empty() || is_empty_position(cols, cols.front())) { continue; }
    auto const first = static_cast<size_type>(view_pool.size());
    for (auto const* p : cols) {
      view_pool.push_back(fill_view{p, p->offset(), p->size()});
    }
    fill_frontier.push_back(
      {&levels_by_column[col_idx], 0, col_idx, first, static_cast<size_type>(cols.size())});
  }

  std::vector<collect_frontier_entry> fill_worklist;
  while (!fill_frontier.empty()) {
    fill_worklist = std::move(fill_frontier);
    fill_frontier.clear();

    for (std::size_t wi = 0; wi < fill_worklist.size(); ++wi) {
      auto const entry        = fill_worklist[wi];
      auto& levels            = *entry.levels;
      auto const parent_first = entry.first_source;
      auto const parent_n     = entry.num_sources;
      auto const cols         = host_span<fill_view const>{view_pool.data() + parent_first,
                                                           static_cast<std::size_t>(parent_n)};

      auto const& kids = shapes[entry.col_idx].children_of[entry.level_idx];

      fill_level_info_at_slot(cols, levels, entry.level_idx);

      auto const tid          = cols.front().type().id();
      auto const num_children = cols.front().num_children();
      if (tid == type_id::STRUCT) {
        for (size_type child_idx = 0; child_idx < num_children; ++child_idx) {
          auto const child_slot   = kids[child_idx];
          auto const first_source = static_cast<size_type>(view_pool.size());
          for (size_type i = 0; i < parent_n; ++i) {
            auto const& parent        = view_pool[static_cast<std::size_t>(parent_first + i)];
            auto const& child_storage = borrowed_child(*parent.storage, child_idx);
            // Exact parents already drained with the slicing sync. Span parents: late.
            if (!parent.counts_known()) {
              nulls.maybe_add(
                child_storage, parent.offset, parent.offset + parent.size, &levels, child_slot);
            }
            view_pool.push_back(fill_view{&child_storage, parent.offset, parent.size});
          }
          fill_worklist.push_back({&levels, child_slot, entry.col_idx, first_source, parent_n});
        }
      } else if (tid == type_id::LIST && !kids.empty()) {
        auto const child_slot   = kids.front();
        auto const first_source = static_cast<size_type>(view_pool.size());
        size_type run_base      = 0;
        {
          auto const run_it =
            std::find_if(list_runs.begin(), list_runs.end(), [&](list_run_entry const& e) {
              return e.levels == &levels && e.child_level_idx == child_slot;
            });
          CUDF_EXPECTS(run_it != list_runs.end(),
                       "batch_concatenate slicing plan missing list child run");
          run_base = run_it->run_base;
        }
        size_type child_source = 0;
        for (size_type i = 0; i < parent_n; ++i) {
          auto const& col = view_pool[static_cast<std::size_t>(parent_first + i)];
          if (!(col.size > 0 && col.num_children() > 0)) { continue; }
          auto const idx = static_cast<std::size_t>(run_base + child_source);
          CUDF_EXPECTS(idx < list_child_views.size(),
                       "batch_concatenate slicing plan list child index OOB");
          view_pool.push_back(list_child_views[idx]);
          ++child_source;
        }
        if (child_source > 0) {
          levels[entry.level_idx].list_child_collected = true;
          fill_frontier.push_back({&levels, child_slot, entry.col_idx, first_source, child_source});
        }
      }
    }
  }

  apply_bounds_targets(
    host_span<pending_child_bounds const>{plan.targets.data(), plan.targets.size()},
    host_span<int64_t const>{values.data(), values.size()});

  // Plan-time + span-dependent nulls remain in @p nulls; counts must land before phase-1 slab
  // sizing (`has_nulls`).
  if (resolve_nulls) { resolve_deferred_null_counts(nulls, stream); }
}

// ============================================================================
// 6. GPU kernels and launches
// Every kernel paired with its launch site.
// ============================================================================

// Largest per-level source column count whose offsets are staged through shared memory. The staging
// buffer is requested dynamically at launch and sized for the widest level in the batch, so a
// launch reserves only what it needs; this is the point past which the request would approach the
// 48 KB a kernel may ask for without opting in to a larger carveout.
constexpr size_type max_smem_columns = 1024;

/**
 * @brief Bytes of dynamic shared memory to request for staging `count + 1` offsets of type `T`.
 *
 * @tparam T Type of the offsets being staged
 * @param max_num_columns Widest source column count over the levels in the launch
 */
template <typename T>
std::size_t smem_staging_bytes(size_type max_num_columns)
{
  return (static_cast<std::size_t>(std::min(max_num_columns, max_smem_columns)) + 1) * sizeof(T);
}

/**
 * @brief Concatenates bitmasks for a single level using word-level operations.
 *
 * Each thread processes one output word (32 bits). The algorithm:
 * 1. Find which column(s) contribute to this output word
 * 2. Read one or two source words into a uint64_t
 * 3. Shift the bits to align with the output position
 * 4. Write the output word (with special handling for column boundaries)
 *
 * For boundary handling: when an output word spans two columns, the thread processing
 * the first word of the later column will merge the trailing bits from the previous
 * column with its own bits.
 *
 * @param mask_ptrs Source mask pointers for columns in this level
 * @param mask_offsets Bit offset for each source mask (from column offset)
 * @param offsets_ptr Prefix sum of column sizes (in bits) [num_columns + 1], possibly in shared
 * memory
 * @param col_sizes Number of rows in each source column
 * @param num_columns Number of columns being concatenated
 * @param dest_mask Output mask buffer
 * @param num_output_words Total number of words in the output mask
 * @param first_word_idx Index of the first output word this thread handles
 * @param stride Distance between consecutive output words handled by this thread
 */
__device__ void concatenate_mask_words(bitmask_type const* const* __restrict__ mask_ptrs,
                                       size_type const* __restrict__ mask_offsets,
                                       size_type const* __restrict__ col_sizes,
                                       size_type const* __restrict__ offsets_ptr,
                                       size_type num_columns,
                                       bitmask_type* __restrict__ dest_mask,
                                       size_type num_output_words,
                                       thread_index_type first_word_idx,
                                       thread_index_type stride)
{
  constexpr size_type bits_per_word = detail::size_in_bits<bitmask_type>();

  auto word_idx = first_word_idx;

  while (word_idx < static_cast<thread_index_type>(num_output_words)) {
    // Bit range for this output word
    size_type const out_bit_start = static_cast<size_type>(word_idx) * bits_per_word;

    // Find which column contains the start bit of this word
    size_type col_idx =
      thrust::upper_bound(thrust::seq, offsets_ptr, offsets_ptr + num_columns + 1, out_bit_start) -
      offsets_ptr - 1;

    bitmask_type output_word = 0;
    size_type bits_filled    = 0;

    // Process columns that contribute to this output word
    while (bits_filled < bits_per_word && col_idx < num_columns) {
      size_type const col_start_bit = offsets_ptr[col_idx];
      size_type const col_end_bit   = offsets_ptr[col_idx + 1];
      size_type const col_size      = col_end_bit - col_start_bit;

      // Calculate bit positions within this column
      size_type const current_out_bit = out_bit_start + bits_filled;
      size_type const bit_in_col      = current_out_bit - col_start_bit;

      // How many bits can we take from this column?
      size_type const bits_remaining_in_col = col_size - bit_in_col;
      size_type const bits_needed           = bits_per_word - bits_filled;
      size_type const bits_to_copy          = min(bits_remaining_in_col, bits_needed);

      bitmask_type const* src_mask = mask_ptrs[col_idx];

      if (src_mask == nullptr) {
        // No mask means all valid (all 1s)
        bitmask_type const all_ones_mask = (bits_to_copy == bits_per_word)
                                             ? ~bitmask_type{0}
                                             : ((bitmask_type{1} << bits_to_copy) - 1);
        output_word |= (all_ones_mask << bits_filled);
      } else {
        // Calculate source bit position
        size_type const src_bit_idx = mask_offsets[col_idx] + bit_in_col;

        // Read source bits using the shared funnel-shift helper (end-word clamp is safe:
        // bits_to_copy <= bits_remaining_in_col keeps end_bit within the column's last word)
        bitmask_type src_bits =
          cudf::detail::get_mask_offset_word(src_mask, 0, src_bit_idx, src_bit_idx + bits_to_copy);

        // Mask to get only the bits we need
        bitmask_type const bits_mask      = (bits_to_copy == bits_per_word)
                                              ? ~bitmask_type{0}
                                              : ((bitmask_type{1} << bits_to_copy) - 1);
        bitmask_type const extracted_bits = src_bits & bits_mask;

        // Place the extracted bits at the correct position in the output word
        output_word |= (extracted_bits << bits_filled);
      }

      bits_filled += bits_to_copy;
      col_idx++;
    }

    // Write the output word
    dest_mask[word_idx] = output_word;

    word_idx += stride;
  }
}

/**
 * @brief Concatenates the null masks of many output levels in a single launch.
 *
 * `blockIdx.y` selects the level; the x dimension is a grid-stride loop over that level's output
 * words. Every level's source metadata is flattened into the `all_*` arrays and located through
 * the per-level boundaries, so the whole table costs one upload and one launch instead of one
 * per level.
 *
 * @param all_mask_ptrs Flattened source mask pointers for every level
 * @param all_mask_offsets Flattened bit offset of each source mask
 * @param all_col_sizes Flattened row count of each source column
 * @param all_output_offsets Flattened prefix sums of column sizes, `num_columns + 1` per level
 * @param level_src_offsets Boundaries into the `all_mask_*` / `all_col_sizes` arrays [num_levels+1]
 * @param level_output_offsets_offsets Boundaries into `all_output_offsets` [num_levels+1]
 * @param level_num_src_columns Number of source columns per level [num_levels]
 * @param level_dest_masks Output mask buffer per level [num_levels]
 * @param level_num_output_words Number of output words per level [num_levels]
 */

CUDF_KERNEL void batch_concatenate_masks_kernel_2d(
  bitmask_type const* const* __restrict__ all_mask_ptrs,
  size_type const* __restrict__ all_mask_offsets,
  size_type const* __restrict__ all_col_sizes,
  size_type const* __restrict__ all_output_offsets,
  size_type const* __restrict__ level_src_offsets,
  size_type const* __restrict__ level_output_offsets_offsets,
  size_type const* __restrict__ level_num_src_columns,
  bitmask_type* const* __restrict__ level_dest_masks,
  size_type const* __restrict__ level_num_output_words)
{
  auto const level = blockIdx.y;

  auto const src_start        = level_src_offsets[level];
  auto const num_columns      = level_num_src_columns[level];
  auto const num_output_words = level_num_output_words[level];
  auto const output_offsets   = all_output_offsets + level_output_offsets_offsets[level];

  // Cache output_offsets in shared memory for the binary search, when it fits
  extern __shared__ char smem_raw[];
  auto* const smem_output_offsets = reinterpret_cast<size_type*>(smem_raw);
  bool const use_smem             = (num_columns <= max_smem_columns);

  if (use_smem) {
    for (size_type i = threadIdx.x; i <= num_columns; i += batch_block_size) {
      smem_output_offsets[i] = output_offsets[i];
    }
    __syncthreads();
  }

  concatenate_mask_words(all_mask_ptrs + src_start,
                         all_mask_offsets + src_start,
                         all_col_sizes + src_start,
                         use_smem ? smem_output_offsets : output_offsets,
                         num_columns,
                         level_dest_masks[level],
                         num_output_words,
                         cudf::detail::grid_1d::global_thread_id<batch_block_size>(),
                         cudf::detail::grid_1d::grid_stride<batch_block_size>());
}

/**
 * @brief One output null mask to build, as handed to `launch_batched_mask_concat`.
 */
struct mask_concat_task {
  level_info const* level;  ///< Level whose source masks are concatenated
  bitmask_type* dest_mask;  ///< Pre-allocated output mask for that level
};

/**
 * @brief Concatenates every requested null mask with one metadata upload and one launch.
 *
 * Uploading per level instead would cost four device allocations, four host-to-device copies and
 * one launch for each of them, which dominates the runtime for wide, deeply nested schemas.
 *
 * @param tasks Output masks to build
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param staging Pinned block the metadata is uploaded from
 */
/// Blocks needed for @p n work items, clamped to at least one and at most @p cap.
inline size_type blocks_capped(size_type n, size_type cap)
{
  return std::max(size_type{1},
                  std::min(cap, cudf::util::div_rounding_up_safe(n, batch_block_size)));
}

/// Maximum active blocks of @p Kernel per SM.
template <auto Kernel>
inline int32_t max_active_blocks_per_sm()
{
  int32_t max_active_blocks{-1};
  CUDF_CUDA_TRY(
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, Kernel, batch_block_size, 0));
  return max_active_blocks;
}

void launch_batched_mask_concat(host_span<mask_concat_task const> tasks,
                                cuda::stream_ref stream,
                                pinned_metadata& staging)
{
  if (tasks.empty()) { return; }

  auto const num_levels = static_cast<size_type>(tasks.size());

  std::size_t total_src = 0;
  for (auto const& task : tasks) {
    total_src += task.level->mask_ptrs.size();
  }

  std::vector<bitmask_type const*> all_mask_ptrs;
  std::vector<size_type> all_mask_offsets;
  std::vector<size_type> all_col_sizes;
  std::vector<size_type> all_output_offsets;
  all_mask_ptrs.reserve(total_src);
  all_mask_offsets.reserve(total_src);
  all_col_sizes.reserve(total_src);
  all_output_offsets.reserve(total_src + num_levels);

  std::vector<size_type> level_src_offsets;
  std::vector<size_type> level_output_offsets_offsets;
  std::vector<size_type> level_num_src_columns;
  std::vector<bitmask_type*> level_dest_masks;
  std::vector<size_type> level_num_output_words;
  level_src_offsets.reserve(num_levels + 1);
  level_output_offsets_offsets.reserve(num_levels + 1);
  level_num_src_columns.reserve(num_levels);
  level_dest_masks.reserve(num_levels);
  level_num_output_words.reserve(num_levels);

  size_type max_output_words    = 0;
  size_type max_num_src_columns = 0;

  for (auto const& task : tasks) {
    auto const& level = *task.level;

    level_src_offsets.push_back(static_cast<size_type>(all_mask_ptrs.size()));
    level_output_offsets_offsets.push_back(static_cast<size_type>(all_output_offsets.size()));
    level_num_src_columns.push_back(static_cast<size_type>(level.mask_ptrs.size()));
    level_dest_masks.push_back(task.dest_mask);
    max_num_src_columns =
      std::max(max_num_src_columns, static_cast<size_type>(level.mask_ptrs.size()));

    all_mask_ptrs.insert(all_mask_ptrs.end(), level.mask_ptrs.begin(), level.mask_ptrs.end());
    all_mask_offsets.insert(
      all_mask_offsets.end(), level.mask_offsets.begin(), level.mask_offsets.end());
    all_col_sizes.insert(all_col_sizes.end(), level.col_sizes.begin(), level.col_sizes.end());

    // Output offsets are the prefix sum of the source column sizes, in bits
    size_type running = 0;
    all_output_offsets.push_back(running);
    for (auto const col_size : level.col_sizes) {
      running += col_size;
      all_output_offsets.push_back(running);
    }

    auto const num_output_words = cudf::util::div_rounding_up_safe(level.total_rows, 32);
    level_num_output_words.push_back(num_output_words);
    max_output_words = std::max(max_output_words, num_output_words);
  }
  level_src_offsets.push_back(static_cast<size_type>(all_mask_ptrs.size()));
  level_output_offsets_offsets.push_back(static_cast<size_type>(all_output_offsets.size()));

  auto const scratch_mr = cudf::get_current_device_resource_ref();

  auto d_all_mask_ptrs      = staging.upload(all_mask_ptrs, stream, scratch_mr);
  auto d_all_mask_offsets   = staging.upload(all_mask_offsets, stream, scratch_mr);
  auto d_all_col_sizes      = staging.upload(all_col_sizes, stream, scratch_mr);
  auto d_all_output_offsets = staging.upload(all_output_offsets, stream, scratch_mr);
  auto d_level_src_offsets  = staging.upload(level_src_offsets, stream, scratch_mr);
  auto d_level_output_offsets_offsets =
    staging.upload(level_output_offsets_offsets, stream, scratch_mr);
  auto d_level_num_src_columns  = staging.upload(level_num_src_columns, stream, scratch_mr);
  auto d_level_dest_masks       = staging.upload(level_dest_masks, stream, scratch_mr);
  auto d_level_num_output_words = staging.upload(level_num_output_words, stream, scratch_mr);



  auto const occupancy_grid = max_active_blocks_per_sm<batch_concatenate_masks_kernel_2d>() *
                              cudf::detail::num_multiprocessors();
  auto const num_blocks_x = blocks_capped(max_output_words, occupancy_grid);
  auto const smem_bytes   = smem_staging_bytes<size_type>(max_num_src_columns);

  for_each_grid_y_chunk(
    num_levels, static_cast<unsigned int>(num_blocks_x), [&](size_type first, dim3 grid) {
      batch_concatenate_masks_kernel_2d<<<grid, batch_block_size, smem_bytes, stream.get()>>>(
        d_all_mask_ptrs.data(),
        d_all_mask_offsets.data(),
        d_all_col_sizes.data(),
        d_all_output_offsets.data(),
        d_level_src_offsets.data() + first,
        d_level_output_offsets_offsets.data() + first,
        d_level_num_src_columns.data() + first,
        d_level_dest_masks.data() + first,
        d_level_num_output_words.data() + first);
    });
}

/**
 * @brief One (output column, source) pair: a run of offsets that copies with one constant delta.
 *
 * Describing the run once lets the kernel hoist everything per-element work would reload out of
 * the loop: no per-element search, the corrections collapse into a single `delta`, and the source
 * width stops being a per-element branch.
 */

/**

 * @brief Writes one run of offsets per block row, with no per-element search.
 *
 * Templated on the output width so the offsetalator's per-element type dispatch is gone too; the
 * segments are split by output width on the host and each width gets its own launch.
 */
template <size_type block_size, typename OutT>
CUDF_KERNEL void batch_concatenate_offsets_segments_kernel(
  offsets_segment const* __restrict__ segments)
{
  auto const seg   = segments[blockIdx.y];
  auto* const out  = static_cast<OutT*>(seg.out_ptr) + seg.out_base;
  auto const count = seg.count;
  auto const delta = seg.delta;

  auto const start     = static_cast<size_type>(threadIdx.x + blockIdx.x * block_size);
  auto const stride    = static_cast<size_type>(block_size * gridDim.x);
  auto const type_size = seg.src_is_int64 ? size_type{8} : size_type{4};

  for (auto i = start; i < count; i += stride) {
    out[i] = static_cast<OutT>(read_offset_value(seg.src_ptr, seg.src_base + i, type_size) + delta);
  }

  // The trailing element belongs to the column, not to any run, so its last run writes it
  if (seg.write_tail && start == 0) { out[count] = static_cast<OutT>(seg.tail_value); }
}

/**
 * @brief Phase 2's copy runs on its own stream, concurrently with phases 3 and 4.
 *
 * The copy descriptors are independent work items writing disjoint destinations, so any partition
 * of them is safe; splitting by input row range instead would share boundary mask words between
 * streams. Nothing here touches the null masks or the offsets.
 */

/// Streams the batched copy is split across. The copy descriptors write disjoint destinations, so
/// any split is safe; past four ways the gain levels off.
constexpr int copy_ways = 4;

/**
 * @brief Builds one descriptor per (level, source) and launches the segment-directed offsets
 * kernel.
 *
 * Needs no grouping by row count: a run's length lives in its own descriptor, so runs of every
 * length share a launch and the host-side bucketing the element-directed path performs disappears
 * along with the search it existed to feed. One launch per output width, because the width is a
 * template parameter rather than a per-element branch.
 *
 * @param levels_by_column Levels of every output column
 * @param offsets_columns_by_column Allocated offsets column of each level, indexed to match
 * @param stream Stream the launches and the descriptor upload are issued on
 * @param staging Pinned block the descriptors are uploaded from
 */




void launch_offsets_segments(
  std::vector<std::vector<level_info>> const& levels_by_column,
  std::vector<std::vector<std::unique_ptr<column>>>& offsets_columns_by_column,
  cuda::stream_ref stream,
  pinned_metadata& staging)
{
  std::vector<offsets_segment> segments_32;
  std::vector<offsets_segment> segments_64;

  auto const num_columns = static_cast<size_type>(levels_by_column.size());
  for (size_type col_idx = 0; col_idx < num_columns; ++col_idx) {
    auto const& levels = levels_by_column[col_idx];
    for (size_type level_idx = 0; level_idx < static_cast<size_type>(levels.size()); ++level_idx) {
      auto const& level = levels[level_idx];
      if (!level.is_string && !level.is_list) { continue; }

      auto out_view           = offsets_columns_by_column[col_idx][level_idx]->mutable_view();
      auto const out_is_int64 = out_view.type().id() == type_id::INT64;
      auto& dst               = out_is_int64 ? segments_64 : segments_32;
      auto const src_is_int64 = cudf::size_of(level.offsets_dtype) == 8;
      auto const num_src      = static_cast<size_type>(level.src_offsets_ptrs.size());

      size_type out_base = 0;
      for (size_type s = 0; s < num_src; ++s) {
        auto const count   = level.col_sizes[s];
        auto const is_last = (s + 1 == num_src);

        // An empty source contributes nothing, unless it is the run that owns the trailing total
        if (count > 0 || is_last) {
          dst.push_back(offsets_segment{
            level.src_offsets_ptrs[s],
            out_view.head<void>(),
            static_cast<int64_t>(level.chars_partition_offsets[s]) - level.first_src_offsets[s],
            static_cast<int64_t>(level.total_child_size),
            level.src_offsets_offsets[s],
            out_base,
            count,
            src_is_int64,
            is_last});
        }
        out_base += count;
      }
    }
  }

  auto const launch = [&](auto out_type_tag, std::vector<offsets_segment> const& segments) {
    using OutT = decltype(out_type_tag);
    if (segments.empty()) { return; }

    auto d_segments = staging.upload(segments, stream, cudf::get_current_device_resource_ref());

    auto const occupancy_grid =
      max_active_blocks_per_sm<
        batch_concatenate_offsets_segments_kernel<batch_block_size, OutT>>() *
      cudf::detail::num_multiprocessors();

    auto const longest =
      std::max_element(segments.begin(), segments.end(), [](auto const& a, auto const& b) {
        return a.count < b.count;
      })->count;
    auto const num_blocks_x = blocks_capped(longest + 1, occupancy_grid);

    auto const num_segments = static_cast<size_type>(segments.size());
    for_each_grid_y_chunk(
      num_segments, static_cast<unsigned int>(num_blocks_x), [&](size_type first, dim3 grid) {
        batch_concatenate_offsets_segments_kernel<batch_block_size, OutT>
          <<<grid, batch_block_size, 0, stream.get()>>>(d_segments.data() + first);
      });
  };

  launch(int32_t{}, segments_32);
  launch(int64_t{}, segments_64);
}

/**
 * @brief Per-request range popcount for deferred sliced null counts.
 *
 * One block-row per request; grid.y chunked like `find_unsanitized_structs_kernel`.
 */
CUDF_KERNEL void batched_range_null_count_kernel(bitmask_type const* const* __restrict__ masks,
                                                 size_type const* __restrict__ begins,
                                                 size_type const* __restrict__ ends,
                                                 size_type* __restrict__ out_counts)
{
  auto const req   = static_cast<size_type>(blockIdx.y);
  auto const* mask = masks[req];
  auto const begin = begins[req];
  auto const end   = ends[req];
  if (mask == nullptr || begin >= end) { return; }

  constexpr auto word_size = detail::size_in_bits<bitmask_type>();
  auto const first_word    = word_index(begin);
  auto const last_word     = word_index(end - 1);

  auto const tid    = static_cast<size_type>(threadIdx.x + blockIdx.x * batch_block_size);
  auto const stride = static_cast<size_type>(batch_block_size * gridDim.x);
  size_type thread_set{0};

  for (auto word = first_word + tid; word <= last_word; word += stride) {
    thread_set += cuda::std::popcount(mask[word]);
  }

  // Subtract slack bits outside [begin, end) on the first/last word (same as count_set_bits_kernel)
  if (tid < 2) {
    bool const first          = tid == 0;
    size_type const bit_index = first ? begin : (end - 1);
    size_type const word_idx  = first ? first_word : last_word;
    size_type num_slack_bits  = bit_index % word_size;
    if (!first) { num_slack_bits = word_size - num_slack_bits - 1; }
    if (num_slack_bits > 0) {
      auto const word = mask[word_idx];
      auto const slack_mask =
        first ? cuda::bitmask<bitmask_type>(0, num_slack_bits)
              : cuda::bitmask<bitmask_type>(word_size - num_slack_bits, num_slack_bits);
      thread_set -= cuda::std::popcount(word & slack_mask);
    }
  }

  using BlockReduce = cub::BlockReduce<size_type, batch_block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  auto const block_set = BlockReduce(temp_storage).Sum(thread_set);
  if (threadIdx.x == 0) {
    cuda::atomic_ref<size_type, cuda::thread_scope_device>{out_counts[req]}.fetch_add(
      block_set, cuda::std::memory_order_relaxed);
  }
}

/**
 * @brief One struct-parent / child null-mask pair to test for agreement.
 */
struct superimpose_check {
  std::uint8_t* flag;               ///< Set when the pair disagrees
  bitmask_type const* parent_mask;  ///< Null mask of the struct level
  bitmask_type const* child_mask;   ///< Null mask of one of its children
  size_type num_rows;               ///< Rows both masks span
  size_type parent_offset;          ///< Bit offset of the first row in the parent mask
  size_type child_offset;           ///< Bit offset of the first row in the child mask
};

/**
 * @brief Flags pairs where a child is valid on a row its struct parent calls null.
 *
 * @param parent_masks Struct null mask of each pair
 * @param child_masks Child null mask of each pair
 * @param num_rows Row count of each pair
 * @param flags Set to 1 for each disagreeing pair; untouched otherwise
 */
/**
 * @brief Reads word @p word of the @p rows bits starting at bit @p offset of @p mask.
 *
 * A sliced source starts partway into a word, so the word wanted spans two stored words. The last
 * read is clamped to the word holding the final row so a slice never reads past its own mask.
 */
CUDF_KERNEL void find_unsanitized_structs_kernel(
  bitmask_type const* const* __restrict__ parent_masks,
  bitmask_type const* const* __restrict__ child_masks,
  size_type const* __restrict__ num_rows,
  size_type const* __restrict__ parent_offsets,
  size_type const* __restrict__ child_offsets,
  std::uint8_t* __restrict__ flags)
{
  auto const check_idx = static_cast<size_type>(blockIdx.y);
  auto const rows      = num_rows[check_idx];
  auto const num_words = cudf::util::div_rounding_up_safe(rows, static_cast<size_type>(32));

  auto const* const parent = parent_masks[check_idx];
  auto const* const child  = child_masks[check_idx];
  auto const parent_offset = parent_offsets[check_idx];
  auto const child_offset  = child_offsets[check_idx];

  auto const stride = static_cast<size_type>(batch_block_size * gridDim.x);
  for (auto word = static_cast<size_type>(threadIdx.x + blockIdx.x * batch_block_size);
       word < num_words;
       word += stride) {
    // Bits past the last row are padding and must not decide the outcome
    auto const trailing_bits = rows - word * 32;
    auto const in_range =
      trailing_bits >= 32 ? ~bitmask_type{0} : ((bitmask_type{1} << trailing_bits) - 1);

    auto const parent_word =
      cudf::detail::get_mask_offset_word(parent, word, parent_offset, parent_offset + rows);
    auto const child_word =
      cudf::detail::get_mask_offset_word(child, word, child_offset, child_offset + rows);
    if ((~parent_word & child_word & in_range) != 0) {
      flags[check_idx] = 1;
      return;
    }
  }
}

/**
 * @brief Records the superimpose decision from the *source* masks, before any output exists.
 *
 * The output mask of a struct level is its sources' masks concatenated, and a struct child always
 * has its parent's row count, so output row i of the parent and of the child both come from the
 * same row of the same source. A child that is valid where its parent is null therefore exists in
 * the output if and only if it exists in some source, which makes this decision available from the
 * inputs alone. Taking it here instead of from the finished output masks moves the readback ahead
 * of every allocation and launch.
 *
 * @param levels Levels of one output column
 * @param level_idx Preorder cursor, advanced in step with reconstruction
 * @param needs_superimpose One entry per level, set for struct levels that cannot take the fast
 * path
 * @param checks Accumulates the pairs to test
 */
void collect_superimpose_checks_from_inputs(std::vector<level_info> const& levels,
                                            size_type& level_idx,
                                            std::vector<std::uint8_t>& needs_superimpose,
                                            std::vector<superimpose_check>& checks)
{
  auto const current = level_idx;
  auto const& level  = levels[current];
  ++level_idx;

  if (level.is_dictionary) { return; }

  if (level.is_struct) {
    for (size_type i = 0; i < level.num_children; ++i) {
      auto const child_level = level_idx;
      collect_superimpose_checks_from_inputs(levels, level_idx, needs_superimpose, checks);

      auto const& child = levels[child_level];

      // A dictionary child builds its own mask elsewhere and cannot be settled here
      if (child.is_dictionary) {
        needs_superimpose[current] = 1;
        continue;
      }

      auto const num_sources = static_cast<size_type>(level.mask_ptrs.size());
      for (size_type s = 0; s < num_sources; ++s) {
        auto const* parent_mask = level.mask_ptrs[s];

        // A source whose struct has no mask calls no row null, so it can push nothing down
        if (parent_mask == nullptr) { continue; }

        // A source whose child has no mask calls every row valid, including the rows its parent
        // calls null, and that cannot be ruled out without reading the parent
        if (s >= static_cast<size_type>(child.mask_ptrs.size()) || child.mask_ptrs[s] == nullptr) {
          needs_superimpose[current] = 1;
          continue;
        }

        if (level.col_sizes[s] == 0) { continue; }

        checks.push_back({&needs_superimpose[current],
                          parent_mask,
                          child.mask_ptrs[s],
                          level.col_sizes[s],
                          level.mask_offsets[s],
                          child.mask_offsets[s]});
      }
    }
  } else if (level.is_list) {
    if (level.total_child_size != 0) {
      collect_superimpose_checks_from_inputs(levels, level_idx, needs_superimpose, checks);
    }
  }
}

/**
 * @brief Builds the per-pair superimpose launch arrays and the widest word count.
 */
void fill_superimpose_arrays(host_span<superimpose_check const> checks,
                             std::vector<bitmask_type const*>& parent_masks,
                             std::vector<bitmask_type const*>& child_masks,
                             std::vector<size_type>& num_rows,
                             std::vector<size_type>& parent_offsets,
                             std::vector<size_type>& child_offsets,
                             size_type& max_words)
{
  auto const num_checks = static_cast<size_type>(checks.size());
  max_words             = 0;
  for (size_type i = 0; i < num_checks; ++i) {
    parent_masks[i]   = checks[i].parent_mask;
    child_masks[i]    = checks[i].child_mask;
    num_rows[i]       = checks[i].num_rows;
    parent_offsets[i] = checks[i].parent_offset;
    child_offsets[i]  = checks[i].child_offset;
    max_words         = std::max(max_words, cudf::util::div_rounding_up_safe(num_rows[i], 32));
  }
}

/**
 * @brief Copies each nonzero readback flag into the pair's out-flag.
 */
void apply_superimpose_flags(host_span<superimpose_check const> checks,
                             host_span<std::uint8_t const> flags)
{
  auto const num_checks = static_cast<size_type>(checks.size());
  for (size_type i = 0; i < num_checks; ++i) {
    if (flags[i] != 0) { *checks[i].flag = 1; }
  }
}

/**
 * @brief One stream sync for deferred null-count popcounts and struct superimpose checks.
 *
 * Both kernels are independent of each other and only need filled `level_info` / pending ranges.
 * Launching both then draining with a single `synchronize` saves a host round trip between them.
 */
void resolve_deferred_null_counts_and_superimpose(deferred_null_counts& deferred,
                                                  host_span<superimpose_check const> checks,
                                                  cuda::stream_ref stream)
{
  if (deferred.requests.empty() && checks.empty()) { return; }

  auto const scratch_mr = cudf::get_current_device_resource_ref();
  auto const num_checks = static_cast<size_type>(checks.size());
  auto const num_nulls  = static_cast<size_type>(deferred.requests.size());

  // ---- Null-count launch (pageable uploads; drained below while arrays live) ----
  std::vector<bitmask_type const*> null_masks(num_nulls);
  std::vector<size_type> begins(num_nulls);
  std::vector<size_type> ends(num_nulls);
  size_type max_null_words = 0;
  fill_null_count_arrays(deferred, null_masks, begins, ends, max_null_words);
  auto d_null_masks = cudf::detail::make_device_uvector_async(null_masks, stream, scratch_mr);
  auto d_begins     = cudf::detail::make_device_uvector_async(begins, stream, scratch_mr);
  auto d_ends       = cudf::detail::make_device_uvector_async(ends, stream, scratch_mr);
  rmm::device_uvector<size_type> d_set_counts(num_nulls, stream, scratch_mr);
  // cudaMemsetAsync's behavior on a null pointer at count 0 is undocumented; skip the zeroing.
  if (num_nulls > 0) {
    CUDF_CUDA_TRY(cudaMemsetAsync(
      d_set_counts.data(), 0, d_set_counts.size() * sizeof(size_type), stream.get()));
  }

  auto const num_blocks_x = blocks_capped(max_null_words, 32);
  for_each_grid_y_chunk(
    num_nulls, static_cast<unsigned int>(num_blocks_x), [&](size_type first, dim3 grid) {
      batched_range_null_count_kernel<<<grid, batch_block_size, 0, stream.get()>>>(
        d_null_masks.data() + first,
        d_begins.data() + first,
        d_ends.data() + first,
        d_set_counts.data() + first);
    });

  // ---- Superimpose launch ----
  // Pageable uploads, drained below while the five arrays are still in scope — the same lifetime
  // arrangement `pinned_metadata` exists to buy explicitly everywhere else, free here from the
  // readback happening inside this function.
  std::vector<bitmask_type const*> parent_masks(num_checks);
  std::vector<bitmask_type const*> child_masks(num_checks);
  std::vector<size_type> num_rows(num_checks);
  std::vector<size_type> parent_offsets(num_checks);
  std::vector<size_type> child_offsets(num_checks);
  size_type max_check_words = 0;
  fill_superimpose_arrays(
    checks, parent_masks, child_masks, num_rows, parent_offsets, child_offsets, max_check_words);
  auto d_parent_masks = cudf::detail::make_device_uvector_async(parent_masks, stream, scratch_mr);
  auto d_child_masks  = cudf::detail::make_device_uvector_async(child_masks, stream, scratch_mr);
  auto d_num_rows     = cudf::detail::make_device_uvector_async(num_rows, stream, scratch_mr);
  auto d_parent_offsets =
    cudf::detail::make_device_uvector_async(parent_offsets, stream, scratch_mr);
  auto d_child_offsets = cudf::detail::make_device_uvector_async(child_offsets, stream, scratch_mr);
  rmm::device_uvector<std::uint8_t> d_flags(num_checks, stream, scratch_mr);
  if (num_checks > 0) {
    CUDF_CUDA_TRY(
      cudaMemsetAsync(d_flags.data(), 0, d_flags.size() * sizeof(std::uint8_t), stream.get()));
  }

  {
    auto const num_blocks_x = blocks_capped(max_check_words, 32);
    for_each_grid_y_chunk(
      num_checks, static_cast<unsigned int>(num_blocks_x), [&](size_type first, dim3 grid) {
        find_unsanitized_structs_kernel<<<grid, batch_block_size, 0, stream.get()>>>(
          d_parent_masks.data() + first,
          d_child_masks.data() + first,
          d_num_rows.data() + first,
          d_parent_offsets.data() + first,
          d_child_offsets.data() + first,
          d_flags.data() + first);
      });
  }
  CUDF_CUDA_TRY(cudaGetLastError());

  // ---- One drain for both D2Hs ----
  auto set_counts = cudf::detail::make_std_vector_async(
    device_span<size_type const>{d_set_counts.data(), d_set_counts.size()}, stream);
  auto flags = cudf::detail::make_std_vector_async(
    device_span<std::uint8_t const>{d_flags.data(), d_flags.size()}, stream);
  cudf::detail::sync_stream(stream);

  accumulate_null_counts(deferred,
                         device_span<size_type const>{set_counts.data(), set_counts.size()});

  apply_superimpose_flags(checks, flags);
}

/**
 * @brief Resolves every deferred sliced null count into its level's totals before slab sizing.
 */
void resolve_deferred_null_counts(deferred_null_counts& deferred, cuda::stream_ref stream)
{
  resolve_deferred_null_counts_and_superimpose(
    deferred, host_span<superimpose_check const>{}, stream);
}

}  // anonymous namespace

// ============================================================================
// 7. Reconstruction and entry points
// Column rebuild, slab resources, and the public API.
// ============================================================================

/**
 * @brief Storage backing every output buffer of one top-level column position.
 *
 * A column position produces a buffer per nesting level — leaf data, an offsets child, a null mask
 * — and the metadata walk has already sized all of them exactly, so they can be served from a
 * single upstream allocation by bumping a cursor.
 */
struct output_slab {
  rmm::device_buffer storage;
  std::size_t used{0};
  /// One reference for the code building the columns, plus one per live allocation
  std::atomic<int> refs{1};
};

/// Drops one reference to @p slab, freeing the upstream block once the last one goes.
void release_slab(output_slab* slab) noexcept
{
  if (slab != nullptr && slab->refs.fetch_sub(1, std::memory_order_acq_rel) == 1) { delete slab; }
}

/// Owns the builder's reference to a slab, so an exception part way through a column still releases
/// it.
struct slab_releaser {
  void operator()(output_slab* slab) const noexcept { release_slab(slab); }
};
using slab_ref = std::unique_ptr<output_slab, slab_releaser>;

/// Rounds @p bytes up to the next multiple of @p alignment.
constexpr std::size_t round_up_to(std::size_t bytes, std::size_t alignment) noexcept
{
  return (bytes + alignment - 1) / alignment * alignment;
}

/**
 * @brief Stream-ordered memory resource handing out disjoint subranges of one `output_slab`.
 *
 * `rmm::device_buffer` stores its resource in an owning `cuda::mr::any_resource`, so a buffer keeps
 * a copy of this object alive for as long as the buffer itself lives. The slab is reference counted
 * by its live *allocations* rather than by those copies: `allocate` takes a reference and
 * `deallocate` drops one, and `device_buffer` calls them exactly once each per buffer because it
 * skips both when the buffer is empty. So the upstream block is released when the last column
 * carved out of the slab is destroyed, in whatever order that happens, and no caller has to hold
 * anything.
 *
 * The state is a bare pointer to keep the type trivially copyable, which it has to be to be copied
 * around inside `any_resource`'s host-device machinery.
 *
 * Nothing goes back upstream until every buffer in a slab is gone. That is the trade: 5,981
 * allocations and as many frees per call become one of each per column.
 */
class slab_resource {
 public:
  // Default-constructible so the type models cuda::std::semiregular, which the type-erased
  // any_resource wrapper requires; such an instance owns no slab and never serves an allocation.
  slab_resource() = default;
  explicit slab_resource(output_slab* slab) : slab_{slab} {}

  void* allocate_sync(std::size_t bytes, std::size_t alignment) { return bump(bytes, alignment); }
  void deallocate_sync(void*, std::size_t, std::size_t) noexcept { release_slab(slab_); }

  void* allocate(cuda::stream_ref, std::size_t bytes, std::size_t alignment)
  {
    return bump(bytes, alignment);
  }
  void deallocate(cuda::stream_ref, void*, std::size_t, std::size_t) noexcept
  {
    release_slab(slab_);
  }

  [[nodiscard]] bool operator==(slab_resource const& other) const noexcept
  {
    return slab_ == other.slab_;
  }

  friend void get_property(slab_resource const&, cuda::mr::device_accessible) noexcept {}

 private:
  void* bump(std::size_t bytes, std::size_t alignment)
  {
    CUDF_EXPECTS(slab_ != nullptr, "batch_concatenate slab_resource has no slab");
    auto const offset =
      round_up_to(slab_->used, std::max(alignment, rmm::CUDA_ALLOCATION_ALIGNMENT));
    CUDF_EXPECTS(offset + bytes <= slab_->storage.size(),
                 "batch_concatenate output slab exhausted");
    slab_->used = offset + bytes;
    slab_->refs.fetch_add(1, std::memory_order_relaxed);
    return static_cast<char*>(slab_->storage.data()) + offset;
  }

  output_slab* slab_{nullptr};
};

static_assert(cuda::mr::resource_with<slab_resource, cuda::mr::device_accessible>,
              "slab_resource must model a device-accessible stream-ordered resource so that "
              "rmm::device_buffer can own a reference to it");

/**
 * @brief Element type of the offsets child this level will produce.
 *
 * `level.offsets_dtype` describes the *sources*, which the kernels read; the output type depends on
 * whether the concatenated child exceeds the 32-bit offset limit. Mirrors
 * `cudf::strings::detail::create_offsets_child_column`, including its overflow check, because the
 * slab has to know the size before the column is built.
 */
/**
 * @brief The two environment-derived settings that decide a string level's offsets width.
 *
 * Neither `get_offset64_threshold` nor `is_large_strings_enabled` caches, and each is a `getenv`,
 * which is a linear scan of `environ`. `output_offsets_type` runs once per level in both the
 * capacity pass and the allocation pass, so on a wide schema the pair is read thousands of times
 * per call, all before the first enqueue. Reading them once per `batch_concatenate` keeps the
 * per-call lookup while dropping the repetition. Held per call rather than in a static so that a
 * test which sets either variable at runtime still takes effect.
 */
struct large_strings_policy {
  int64_t threshold;
  bool enabled;
};

large_strings_policy read_large_strings_policy()
{
  return {cudf::strings::detail::get_offset64_threshold(),
          cudf::strings::detail::is_large_strings_enabled()};
}

data_type output_offsets_type(level_info const& level, large_strings_policy const& policy)
{
  if (!level.is_string) { return data_type{type_id::INT32}; }
  auto const chars_bytes = static_cast<int64_t>(level.total_child_size);
  if (!policy.enabled) {
    CUDF_EXPECTS(chars_bytes < policy.threshold,
                 "Size of output exceeds the column size limit",
                 std::overflow_error);
  }
  return chars_bytes < policy.threshold ? data_type{type_id::INT32} : data_type{type_id::INT64};
}

/**
 * @brief Bytes one level contributes to its column's slab, each buffer padded to the allocation
 * alignment so that the subranges are as aligned as separate allocations would have been.
 */
std::size_t level_slab_bytes(level_info const& level, large_strings_policy const& policy)
{
  if (level.delegate_to_specialized) { return 0; }

  std::size_t bytes = 0;
  auto const add    = [&bytes](std::size_t n) {
    bytes += round_up_to(n, rmm::CUDA_ALLOCATION_ALIGNMENT);
  };

  if (!level.is_struct && !level.is_list) {
    add(level.is_string
          ? level.total_child_size
          : std::accumulate(level.data_sizes.begin(), level.data_sizes.end(), std::size_t{0}));
  }
  if (level.is_string || level.is_list) {
    add(static_cast<std::size_t>(level.total_rows + 1) *
        cudf::size_of(output_offsets_type(level, policy)));
  }
  if (level.has_nulls) { add(cudf::bitmask_allocation_size_bytes(level.total_rows)); }

  return bytes;
}

/**
 * @brief Result structure for batch_process_table_levels.
 */
struct table_batch_process_result {
  std::vector<std::vector<rmm::device_buffer>> data_buffers_by_column;
  std::vector<std::vector<std::pair<cuda::device_buffer<cudf::bitmask_type>, size_type>>>
    mask_results_by_column;
  std::vector<std::vector<std::unique_ptr<column>>> offsets_columns_by_column;
};

/**
 * @brief Performs all batch operations for table concatenation.
 *
 * Allocates every output buffer, then runs the data copy, offsets and null-mask phases for all
 * column positions together.
 */
table_batch_process_result batch_process_table_levels(
  std::vector<std::vector<level_info>> const& levels_by_column,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  size_type const num_columns = static_cast<size_type>(levels_by_column.size());

  // Phase 1: one slab per column. The metadata walk sized every buffer, so a column's data,
  // offsets and null masks at every level are carved out of one upstream allocation. Slabs are
  // per column so no single huge block is needed and releasing part of the result frees memory.
  std::vector<std::vector<rmm::device_buffer>> data_buffers_by_column(num_columns);
  std::vector<std::vector<cuda::device_buffer<cudf::bitmask_type>>> null_masks_by_column(
    num_columns);
  std::vector<std::vector<std::unique_ptr<column>>> offsets_columns_by_column(num_columns);

  // Forked copy streams, plus the copy's metadata and scratch, which free on the caller's stream
  // and so must outlive the join below.
  struct copy_part {
    rmm::device_uvector<void const*> d_src;
    rmm::device_uvector<void*> d_dst;
    rmm::device_uvector<std::size_t> d_sizes;
    rmm::device_buffer d_temp;
    std::size_t temp_bytes;
  };
  std::vector<cuda::stream_ref> copy_streams;
  std::vector<copy_part> parts;

  // Pinned staging for every phase's metadata uploads, taken before anything is queued so the
  // allocator's synchronize falls on an idle stream.
  pinned_metadata metadata_staging{pinned_metadata_budget(levels_by_column), stream};

  // One batched memcpy covers every copy in the table: cub balances the copies better when it
  // can see all of them at once.
  std::size_t total_copy_ops = 0;
  for (auto const& levels : levels_by_column) {
    for (auto const& level : levels) {
      if (level.delegate_to_specialized || level.is_struct || level.is_list) continue;
      total_copy_ops += level.src_data_ptrs.size();
    }
  }

  std::vector<void const*> src_ptrs;
  std::vector<void*> dst_ptrs;
  std::vector<std::size_t> sizes;
  src_ptrs.reserve(total_copy_ops);
  dst_ptrs.reserve(total_copy_ops);
  sizes.reserve(total_copy_ops);

  auto const strings_policy = read_large_strings_policy();

  for (size_type col_idx = 0; col_idx < num_columns; ++col_idx) {
    auto const& levels    = levels_by_column[col_idx];
    auto const num_levels = static_cast<size_type>(levels.size());

    auto& data_buffers    = data_buffers_by_column[col_idx];
    auto& null_masks      = null_masks_by_column[col_idx];
    auto& offsets_columns = offsets_columns_by_column[col_idx];

    // Entries left default-constructed are empty and own nothing, which is what a struct level or
    // a delegated level needs
    data_buffers.resize(num_levels);
    null_masks.reserve(num_levels);
    for (size_type lvl = 0; lvl < num_levels; ++lvl) {
      null_masks.emplace_back(stream, mr, std::size_t{0}, cuda::no_init);
    }
    offsets_columns.resize(num_levels);

    std::size_t capacity = 0;
    for (auto const& level : levels) {
      capacity += level_slab_bytes(level, strings_policy);
    }
    if (capacity == 0) continue;

    slab_ref const slab{new output_slab{}};
    slab->storage = rmm::device_buffer{capacity, stream, mr};
    cuda::mr::any_resource<cuda::mr::device_accessible> const slab_mr{slab_resource{slab.get()}};

    for (size_type level_idx = 0; level_idx < num_levels; ++level_idx) {
      auto const& level = levels[level_idx];
      if (level.delegate_to_specialized) continue;

      if (!level.is_struct && !level.is_list) {
        if (level.is_string) {
          data_buffers[level_idx] = rmm::device_buffer{level.total_child_size, stream, slab_mr};
        } else {
          std::size_t const total_bytes =
            std::accumulate(level.data_sizes.begin(), level.data_sizes.end(), std::size_t{0});
          data_buffers[level_idx] = rmm::device_buffer{total_bytes, stream, slab_mr};
        }

        char* dst_ptr = static_cast<char*>(data_buffers[level_idx].data());
        for (size_t i = 0; i < level.src_data_ptrs.size(); ++i) {
          if (level.src_data_ptrs[i] != nullptr && level.data_sizes[i] > 0) {
            src_ptrs.push_back(level.src_data_ptrs[i]);
            dst_ptrs.push_back(dst_ptr);
            sizes.push_back(level.data_sizes[i]);
          }
          dst_ptr += level.data_sizes[i];
        }
      }

      // Built directly rather than through create_offsets_child_column and create_null_mask,
      // because those take a non-owning resource reference and the buffer must own a reference to
      // the slab to keep it alive
      if (level.is_string || level.is_list) {
        auto const dtype           = output_offsets_type(level, strings_policy);
        auto const offsets_count   = level.total_rows + 1;
        offsets_columns[level_idx] = std::make_unique<column>(
          dtype,
          offsets_count,
          rmm::device_buffer{
            static_cast<std::size_t>(offsets_count) * cudf::size_of(dtype), stream, slab_mr},
          cuda::device_buffer<cudf::bitmask_type>{stream, mr, std::size_t{0}, cuda::no_init},
          0);
      }

      if (level.has_nulls) {
        null_masks[level_idx] =
          cuda::device_buffer<cudf::bitmask_type>{
            stream,
            slab_mr,
            cudf::bitmask_allocation_size_bytes(level.total_rows) / sizeof(cudf::bitmask_type),
            cuda::no_init};
      }
    }
  }

  // The copy, offsets and mask phases write disjoint buffers from disjoint sources, so the copy
  // can run on forked streams alongside the other two; every buffer was allocated above.

  if (!src_ptrs.empty()) {
    auto const scratch_mr = cudf::get_current_device_resource_ref();

    // The copy descriptors write disjoint destinations, so the parts need no coordination.
    auto const bounds =
      partition_copy_by_bytes(host_span<std::size_t const>{sizes.data(), sizes.size()}, copy_ways);

    // Every part's metadata and CUB scratch is allocated on the caller's stream, ahead of the
    // fork below. CUB's scratch has to be sized separately to get here at all, since the launch
    // would otherwise allocate it on whichever stream runs the copy. Never zero-sized: a null
    // pointer is how CUB is asked for a size, so passing one to the launch would query instead
    // of copy.
    parts.reserve(copy_ways);

    for (int w = 0; w < copy_ways; ++w) {
      auto const first = bounds[w];
      auto const count = bounds[w + 1] - first;
      if (count == 0) continue;

      auto d_src = cudf::detail::make_device_uvector_async(
        metadata_staging.stage(host_span<void const* const>{src_ptrs.data() + first, count}),
        stream,
        scratch_mr);
      auto d_dst = cudf::detail::make_device_uvector_async(
        metadata_staging.stage(host_span<void* const>{dst_ptrs.data() + first, count}),
        stream,
        scratch_mr);
      auto d_sizes = cudf::detail::make_device_uvector_async(
        metadata_staging.stage(host_span<std::size_t const>{sizes.data() + first, count}),
        stream,
        scratch_mr);

      std::size_t temp_bytes = 0;
      batched_memcpy(
        nullptr, temp_bytes, d_src.data(), d_dst.data(), d_sizes.data(), count, stream);
      temp_bytes = std::max<std::size_t>(temp_bytes, 1);
      rmm::device_buffer d_temp{temp_bytes, stream, scratch_mr};

      parts.push_back(copy_part{
        std::move(d_src), std::move(d_dst), std::move(d_sizes), std::move(d_temp), temp_bytes});
    }

    // Forking here leaves every allocation above on the caller's stream and still covers them,
    // because fork_streams records its event where it is called
    copy_streams = cudf::detail::fork_streams(stream, static_cast<std::size_t>(parts.size()));

    for (std::size_t p = 0; p < parts.size(); ++p) {
      auto& part = parts[p];
      batched_memcpy(part.d_temp.data(),
                     part.temp_bytes,
                     part.d_src.data(),
                     part.d_dst.data(),
                     part.d_sizes.data(),
                     part.d_src.size(),
                     copy_streams[p]);
    }
  }

  // Phase 3: offsets, on the caller's stream: a stream forked earlier would not be ordered after
  // this phase's own metadata uploads.
  launch_offsets_segments(levels_by_column, offsets_columns_by_column, stream, metadata_staging);

  // ========== PHASE 4: Batch mask concatenation across ALL columns ==========
  std::vector<std::vector<size_type>> null_counts_by_column(num_columns);

  std::vector<mask_concat_task> tasks;

  for (size_type col_idx = 0; col_idx < num_columns; ++col_idx) {
    auto const& levels = levels_by_column[col_idx];
    auto& null_masks   = null_masks_by_column[col_idx];
    auto& null_counts  = null_counts_by_column[col_idx];

    null_counts.resize(levels.size(), 0);

    for (size_type level_idx = 0; level_idx < static_cast<size_type>(levels.size()); ++level_idx) {
      auto const& level = levels[level_idx];
      if (level.delegate_to_specialized) continue;
      if (!level.has_nulls) continue;

      tasks.push_back({&level, reinterpret_cast<bitmask_type*>(null_masks[level_idx].data())});
      null_counts[level_idx] = level.total_null_count;
    }
  }

  launch_batched_mask_concat(tasks, stream, metadata_staging);

  // Nothing below reads the copy's output, but two things still need this join: the caller's stream
  // has to carry the copy's completion, and `parts` frees on the caller's stream. Destroying it
  // before the join would let the allocator hand that memory out while the copy is still reading
  // it.
  cudf::detail::join_streams(copy_streams, stream);

  // Build mask results
  std::vector<std::vector<std::pair<cuda::device_buffer<cudf::bitmask_type>, size_type>>>
    mask_results_by_column(num_columns);
  for (size_type col_idx = 0; col_idx < num_columns; ++col_idx) {
    auto& mask_results = mask_results_by_column[col_idx];
    auto& null_masks   = null_masks_by_column[col_idx];
    auto& null_counts  = null_counts_by_column[col_idx];

    mask_results.reserve(null_masks.size());
    for (size_t i = 0; i < null_masks.size(); ++i) {
      mask_results.emplace_back(std::move(null_masks[i]), null_counts[i]);
    }
  }

  // Nothing below reads device memory on the host, so the launches stay in flight
  CUDF_CHECK_CUDA(stream.get());

  return {std::move(data_buffers_by_column),
          std::move(mask_results_by_column),
          std::move(offsets_columns_by_column)};
}

/**
 * @brief Reconstructs the column hierarchy from the processed buffers.
 */
std::unique_ptr<column> reconstruct_column(
  std::vector<level_info> const& levels,
  std::vector<rmm::device_buffer>& data_buffers,
  std::vector<std::pair<cuda::device_buffer<cudf::bitmask_type>, size_type>>& mask_results,
  std::vector<std::unique_ptr<column>>& offsets_columns,
  host_span<std::uint8_t const> needs_superimpose,
  size_type& level_idx,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  auto const& level             = levels[level_idx];
  auto& [null_mask, null_count] = mask_results[level_idx];
  size_type const current_level = level_idx;
  ++level_idx;

  // DICTIONARY: delegate to specialized implementation
  if (level.is_dictionary) {
    return cudf::dictionary::detail::concatenate(span_of(level.original_columns), stream, mr);
  }

  if (level.is_struct) {
    std::vector<std::unique_ptr<column>> children;
    children.reserve(level.num_children);

    for (size_type i = 0; i < level.num_children; ++i) {
      children.push_back(reconstruct_column(levels,
                                            data_buffers,
                                            mask_results,
                                            offsets_columns,
                                            needs_superimpose,
                                            level_idx,
                                            stream,
                                            mr));
    }

    // Every child already agrees with this struct's nulls, so pushing them down would be a no-op
    if (needs_superimpose[current_level] == 0) {
      return create_structs_hierarchy(
        level.total_rows, std::move(children), null_count, std::move(null_mask), stream, mr);
    }

    return make_structs_column(
      level.total_rows, std::move(children), null_count, std::move(null_mask), stream, mr);
  } else if (level.is_list) {
    // LIST column: construct from offsets column and recursively reconstructed child
    std::unique_ptr<column> child_column;

    // Handle edge case: all lists were empty (total_child_size == 0).
    if (level.total_child_size == 0) {
      // The slicing-plan walk never records `empty_child_like`; if it emitted a child level,
      // reconstruct that level rather than making an empty column.
      if (level.empty_child_like.has_value()) {
        child_column = empty_like(level.empty_child_like.value());
      } else if (level.list_child_collected) {
        child_column = reconstruct_column(levels,
                                          data_buffers,
                                          mask_results,
                                          offsets_columns,
                                          needs_superimpose,
                                          level_idx,
                                          stream,
                                          mr);
      } else {
        // Fallback for simple types (or if no non-empty column was found)
        child_column = make_empty_column(level.child_dtype);
      }
    } else {
      // Recursively reconstruct child column from next level
      child_column = reconstruct_column(levels,
                                        data_buffers,
                                        mask_results,
                                        offsets_columns,
                                        needs_superimpose,
                                        level_idx,
                                        stream,
                                        mr);
    }

    return make_lists_column(level.total_rows,
                             std::move(offsets_columns[current_level]),
                             std::move(child_column),
                             null_count,
                             std::move(null_mask));
  } else if (level.is_string) {
    // String column: construct from offsets column and chars buffer
    return make_strings_column(level.total_rows,
                               std::move(offsets_columns[current_level]),
                               std::move(data_buffers[current_level]),
                               null_count,
                               std::move(null_mask));
  } else {
    return std::make_unique<column>(level.dtype,
                                    level.total_rows,
                                    std::move(data_buffers[current_level]),
                                    std::move(null_mask),
                                    null_count);
  }
}

/**
 * @brief Concatenates tables that carry a row count but no columns.
 *
 * @param tables_to_concat Tables to concatenate, all with zero columns
 * @return An empty-schema table whose row count is the sum of the inputs'
 */
std::unique_ptr<table> concatenate_zero_column_tables(host_span<table_view const> tables_to_concat)
{
  auto const total_rows = std::accumulate(
    tables_to_concat.begin(),
    tables_to_concat.end(),
    std::size_t{0},
    [](std::size_t acc, auto const& t) { return acc + static_cast<std::size_t>(t.num_rows()); });
  CUDF_EXPECTS(total_rows <= static_cast<std::size_t>(std::numeric_limits<size_type>::max()),
               "Total number of rows exceeds the column size limit",
               std::overflow_error);
  return std::make_unique<table>(std::vector<std::unique_ptr<column>>{},
                                 static_cast<size_type>(total_rows));
}

std::unique_ptr<column> batch_concatenate(host_span<column_view const> columns_to_concat,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(!columns_to_concat.empty(), "Unexpected empty list of columns to concatenate.");

  // Handle EMPTY type early (before support check)
  if (columns_to_concat.front().type().id() == cudf::type_id::EMPTY) {
    auto length = std::accumulate(
      columns_to_concat.begin(), columns_to_concat.end(), 0, [](auto a, auto const& b) {
        return a + b.size();
      });
    return make_empty_type_column(length, stream, mr);
  }

  // Too little flat data to pay for batching; the unbatched path is quicker below this size
  auto const bytes = flat_payload_bytes(columns_to_concat);
  if (bytes && *bytes < min_batch_payload_bytes) {
    return cudf::detail::concatenate(columns_to_concat, stream, mr);
  }

  // Handle all-empty case. Validation still runs: fused collect is skipped when there is nothing
  // to gather, so the standalone checker covers the empty case.
  if (std::all_of(columns_to_concat.begin(), columns_to_concat.end(), [](column_view const& c) {
        return c.is_empty();
      })) {
    batch_bounds_and_type_check(
      host_span<column_view const>{columns_to_concat.data(), columns_to_concat.size()}, stream);
    auto const& front = columns_to_concat.front();
    // Handle types that need special construction for empty columns
    auto const tid = front.type().id();

    // DICTIONARY columns: delegate to specialized implementation
    if (tid == cudf::type_id::DICTIONARY32) {
      return cudf::dictionary::detail::concatenate(columns_to_concat, stream, mr);
    }

    // LIST columns with no children: create minimal empty list structure
    if (tid == cudf::type_id::LIST && front.num_children() == 0) {
      auto offsets = make_empty_column(cudf::data_type{cudf::type_id::INT32});
      auto child   = make_empty_column(cudf::data_type{cudf::type_id::INT8});
      return make_lists_column(
        0,
        std::move(offsets),
        std::move(child),
        0,
        cudf::detail::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr));
    }

    return empty_like(front);
  }

  deferred_bounds deferred;
  deferred_null_counts nulls;

  // Step 1: Collect all level information
  std::vector<column_view const*> root_ptrs;
  root_ptrs.reserve(columns_to_concat.size());
  for (auto const& col : columns_to_concat) {
    root_ptrs.push_back(&col);
  }
  host_span<column_view const* const> const root_span{root_ptrs.data(), root_ptrs.size()};
  std::vector<std::vector<level_info>> levels_by_column(1);
  collect_levels_bfs(host_span<host_span<column_view const* const> const>{&root_span, 1},
                     host_span<std::vector<level_info>>{levels_by_column.data(), 1},
                     deferred,
                     nulls,
                     stream);
  auto& levels = levels_by_column.front();

  // Step 2: Settle the struct superimpose decision from the source masks, before anything is
  // allocated, so its readback sits next to the bounds readback rather than after every launch
  std::vector<std::uint8_t> needs_superimpose(levels.size(), 0);
  std::vector<superimpose_check> checks;
  size_type check_idx = 0;
  collect_superimpose_checks_from_inputs(levels, check_idx, needs_superimpose, checks);
  resolve_deferred_null_counts_and_superimpose(nulls, checks, stream);

  // Step 3: Batch process all data copy, mask concatenation, and string offsets. The single
  // column rides the grouped table path: one column position holding all its levels.
  auto result = batch_process_table_levels(levels_by_column, stream, mr);

  // Step 4: Reconstruct the column hierarchy
  size_type level_idx = 0;
  return reconstruct_column(levels_by_column.front(),
                            result.data_buffers_by_column.front(),
                            result.mask_results_by_column.front(),
                            result.offsets_columns_by_column.front(),
                            needs_superimpose,
                            level_idx,
                            stream,
                            mr);
}

std::unique_ptr<table> batch_concatenate(host_span<table_view const> tables_to_concat,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  if (tables_to_concat.empty()) { return std::make_unique<table>(); }

  // Validate column counts match
  table_view const first_table = tables_to_concat.front();
  validate_table_shapes(tables_to_concat);

  auto const num_columns = first_table.num_columns();
  if (num_columns == 0) { return concatenate_zero_column_tables(tables_to_concat); }

  // Too little flat data to pay for batching; the unbatched path is quicker below this size
  auto const bytes = flat_payload_bytes(tables_to_concat);
  if (bytes && *bytes < min_batch_payload_bytes) {
    return cudf::detail::concatenate(tables_to_concat, stream, mr);
  }

  // Step 1: Collect level_info for each column position
  std::vector<std::vector<level_info>> levels_by_column(num_columns);
  std::vector<std::vector<std::uint8_t>> needs_superimpose(num_columns);

  {
    // One batch of deferred child-bound reads covers every column position
    deferred_bounds deferred;
    deferred_null_counts nulls;

    // Sliced roots take the slicing plan; unsliced roots take the depth-first walk (the normal
    // whole-input path, not a rare fallback; only sliced roots pay for the plan).
    bool const use_slicing_plan =
      std::any_of(tables_to_concat.begin(), tables_to_concat.end(), [](table_view const& table) {
        return std::any_of(table.begin(), table.end(), root_needs_slicing_plan);
      });

    if (use_slicing_plan) {
      // Stable per-position source views for the global list-generation frontier: borrowed
      // pointers into live table columns, no deep nested column_view copies at gather.
      std::vector<std::vector<column_view const*>> ptr_storage(
        static_cast<std::size_t>(num_columns));
      std::vector<host_span<column_view const* const>> ptr_roots(
        static_cast<std::size_t>(num_columns));
      for (size_type col_idx = 0; col_idx < num_columns; ++col_idx) {
        collect_table_entry_roots(tables_to_concat,
                                  col_idx,
                                  static_cast<size_type>(tables_to_concat.size()),
                                  ptr_storage[col_idx],
                                  stream,
                                  [&](host_span<column_view const* const> roots_for_position) {
                                    ptr_roots[col_idx] = roots_for_position;
                                  });
      }

      collect_levels_bfs(
        host_span<host_span<column_view const* const> const>{ptr_roots.data(), ptr_roots.size()},
        host_span<std::vector<level_info>>{levels_by_column.data(), levels_by_column.size()},
        deferred,
        nulls,
        stream,
        /*resolve_nulls=*/false);
    } else {
      // Whole/unsliced DFS path.
      std::vector<std::vector<column_view const*>> ptr_storage(
        static_cast<std::size_t>(num_columns));
      for (size_type col_idx = 0; col_idx < num_columns; ++col_idx) {
        collect_table_entry_roots(
          tables_to_concat,
          col_idx,
          static_cast<size_type>(tables_to_concat.size()),
          ptr_storage[col_idx],
          stream,
          [&](host_span<column_view const* const> roots_for_position) {
            collect_level_info_recursive(
              roots_for_position, levels_by_column[col_idx], deferred, stream, &nulls);
          });
      }

      resolve_deferred_bounds(deferred, stream);
    }
    // Deferred sliced null counts + struct superimpose decision share one stream drain.
    // Null counts must land before phase-1 slab sizing (`has_nulls`); superimpose only needs the
    // filled source mask metadata already in `levels_by_column`.
    std::vector<superimpose_check> checks;
    for (size_type col_idx = 0; col_idx < num_columns; ++col_idx) {
      if (levels_by_column[col_idx].empty()) { continue; }
      needs_superimpose[col_idx].assign(levels_by_column[col_idx].size(), 0);
      size_type check_idx = 0;
      collect_superimpose_checks_from_inputs(
        levels_by_column[col_idx], check_idx, needs_superimpose[col_idx], checks);
    }
    resolve_deferred_null_counts_and_superimpose(nulls, checks, stream);
  }

  // Step 3: Batch process all levels
  auto result = batch_process_table_levels(levels_by_column, stream, mr);

  // Step 4: Reconstruct each column
  std::vector<std::unique_ptr<column>> output_columns;
  output_columns.reserve(num_columns);

  for (size_type col_idx = 0; col_idx < num_columns; ++col_idx) {
    auto const& levels = levels_by_column[col_idx];

    // Handle all-empty column case or EMPTY type columns
    if (levels.empty()) {
      auto const& first_col = first_table.column(col_idx);
      if (first_col.type().id() == type_id::EMPTY) {
        // EMPTY type: sum up row counts and create EMPTY column
        size_type total_rows = 0;
        for (auto const& table : tables_to_concat) {
          total_rows += table.column(col_idx).size();
        }
        output_columns.push_back(make_empty_type_column(total_rows, stream, mr));
      } else {
        // All columns at this position were empty (zero rows) - create empty_like
        output_columns.push_back(empty_like(first_col));
      }
      continue;
    }

    size_type level_idx = 0;
    output_columns.push_back(reconstruct_column(levels,
                                                result.data_buffers_by_column[col_idx],
                                                result.mask_results_by_column[col_idx],
                                                result.offsets_columns_by_column[col_idx],
                                                needs_superimpose[col_idx],
                                                level_idx,
                                                stream,
                                                mr));
  }

  return std::make_unique<table>(std::move(output_columns));
}
}  // namespace detail

std::unique_ptr<column> batch_concatenate(host_span<column_view const> columns_to_concat,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::batch_concatenate(columns_to_concat, stream, mr);
}

std::unique_ptr<table> batch_concatenate(host_span<table_view const> tables_to_concat,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::batch_concatenate(tables_to_concat, stream, mr);
}

}  // namespace cudf