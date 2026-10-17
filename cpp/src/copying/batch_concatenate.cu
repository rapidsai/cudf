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

// TEMPORARY SCAFFOLD: the batched-memcpy/deferred-read helpers below gain their callers in
// the following commits. Until then they are unreferenced in this TU, which trips
// -Wunused-function under -Werror. Removed at series completion.
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma nv_diag_suppress declared_but_not_referenced


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

}  // namespace (anonymous)
}  // namespace cudf::detail

// TEMPORARY SCAFFOLD (public): at namespace cudf scope so the public wrappers resolve as
// cudf::batch_concatenate. Replaced by the final wrappers at series completion.
std::unique_ptr<column> batch_concatenate(host_span<column_view const> columns_to_concat,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::concatenate(columns_to_concat, stream, mr);
}

std::unique_ptr<table> batch_concatenate(host_span<table_view const> tables_to_concat,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::concatenate(tables_to_concat, stream, mr);
}

}  // namespace cudf
