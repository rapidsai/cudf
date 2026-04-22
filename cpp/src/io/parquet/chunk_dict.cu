/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parquet_gpu.cuh"

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>

#include <rmm/exec_policy.hpp>

#include <cub/block/block_scan.cuh>
#include <cuco/static_map_ref.cuh>
#include <cuda/atomic>

namespace cudf::io::parquet::detail {

namespace {
constexpr int DEFAULT_BLOCK_SIZE = 256;

// Upper bound on the number of fragments per column chunk that the
// shared-memory histogram in `collect_map_entries_kernel` can accommodate.
// A typical workload is 1M row groups / ~5000-row fragments ≈ 200 fragments
// per chunk, so 1024 is a comfortable ceiling. If a future workload exceeds
// this, `collect_map_entries_kernel` will trip its `cudf_assert` below and
// we should add a global-memory fallback rather than silently truncating.
constexpr size_type MAX_FRAGMENTS_PER_BLOCK = 1024;
}  // namespace

template <typename T>
struct equality_functor {
  column_device_view const& col;
  __device__ bool operator()(key_type lhs_idx, key_type rhs_idx) const
  {
    // We don't call this for nulls so this is fine.
    auto constexpr equal = cudf::detail::row::equality::nan_equal_physical_equality_comparator{};
    return equal(col.element<T>(lhs_idx), col.element<T>(rhs_idx));
  }
};

template <typename T>
struct hash_functor {
  column_device_view const& col;
  uint32_t const seed = 0;
  __device__ auto operator()(key_type idx) const
  {
    return cudf::hashing::detail::MurmurHash3_x86_32<T>{seed}(col.element<T>(idx));
  }
};

template <int block_size>
struct map_insert_fn {
  storage_ref_type const& storage_ref;
  EncColumnChunk* const& chunk;

  template <typename T>
  __device__ void operator()(size_type const s_start_value_idx, size_type const end_value_idx)
  {
    if constexpr (column_device_view::has_element_accessor<T>()) {
      using block_reduce = cub::BlockReduce<size_type, block_size>;
      __shared__ typename block_reduce::TempStorage reduce_storage;

      auto const col                     = chunk->col_desc;
      column_device_view const& data_col = *col->leaf_column;
      __shared__ size_type total_num_dict_entries;

      using equality_fn_type = equality_functor<T>;
      using hash_fn_type     = hash_functor<T>;
      // Choosing `linear_probing` over `double_hashing` for slighhhtly better performance seen in
      // benchmarks.
      using probing_scheme_type = cuco::linear_probing<map_cg_size, hash_fn_type>;

      // Make a view of the hash map.
      auto hash_map_ref = cuco::static_map_ref{cuco::empty_key{KEY_SENTINEL},
                                               cuco::empty_value{VALUE_SENTINEL},
                                               equality_fn_type{data_col},
                                               probing_scheme_type{hash_fn_type{data_col}},
                                               cuco::thread_scope_block,
                                               storage_ref};

      // Create a map ref with `cuco::insert` operator
      auto map_insert_ref = hash_map_ref.rebind_operators(cuco::insert);
      auto const t        = threadIdx.x;

      // Create atomic refs to the current chunk's num_dict_entries and uniq_data_size
      cuda::atomic_ref<size_type, SCOPE> const chunk_num_dict_entries{chunk->num_dict_entries};
      cuda::atomic_ref<size_type, SCOPE> const chunk_uniq_data_size{chunk->uniq_data_size};

      // Note: Adjust the following loop to use `cg::tile<map_cg_size>` if needed in the future.
      for (thread_index_type val_idx = s_start_value_idx + t; val_idx - t < end_value_idx;
           val_idx += block_size) {
        size_type is_unique      = 0;
        size_type uniq_elem_size = 0;

        // Check if this index is valid.
        auto const is_valid =
          val_idx < end_value_idx and val_idx < data_col.size() and data_col.is_valid(val_idx);

        // Insert tile_val_idx to hash map and count successful insertions.
        if (is_valid) {
          // Insert the keys using a single thread for best performance for now.
          // Stamp `slot.second` with the (approximate) lowest column-relative fragment
          // index that inserted this value. `cuco::static_map_ref::insert` writes the
          // value exactly once per key -- on the winning insertion -- so after this
          // kernel, every non-empty slot's `second` field equals the `blockIdx.x` of
          // the block that first won that slot. Block scheduling on NVIDIA GPUs is
          // approximately monotonic in `blockIdx`, giving a strong first-appearance
          // signal that `collect_map_entries_kernel` converts into monotone dict_ids.
          //
          // NOTE: `blockIdx.x` here is the *column-relative* fragment index (the
          // populate grid is (num_fragments_per_col, num_cols)). Each `EncColumnChunk`
          // covers a contiguous range of these indices within its column, so the
          // collect kernel subtracts the chunk's first column-relative fragment to get
          // a chunk-local histogram bucket.
          //
          // The pair is constructed as `slot_type` (exactly `value_type` for the map,
          // 8 bytes total) rather than the tempting `cuco::pair{val_idx, hint}`. The
          // latter would deduce `cuco::pair<int64_t, int32_t>` because `val_idx` is
          // `thread_index_type` (int64_t), and cuco's `packed_cas` path (selected when
          // `sizeof(value_type) <= 8`) reinterprets the input as a single `uint64_t`
          // -- it would then CAS the low 8 bytes (the `int64_t` key) into the slot and
          // silently drop the payload, leaving `slot.second = 0`.
          //
          // TODO: when the cuco pin exposes `static_map::insert_or_apply` with
          // `cuco::op::min`, switch to that for exact first-fragment semantics
          // (PROBLEM.md §5.2 Option II).
          auto const fragment_hint = static_cast<mapped_type>(blockIdx.x);
          is_unique      = map_insert_ref.insert(
            slot_type{static_cast<key_type>(val_idx), fragment_hint});
          uniq_elem_size = [&]() -> size_type {
            if (not is_unique) { return 0; }
            switch (col->physical_type) {
              case Type::INT32: return 4;
              case Type::INT64: return 8;
              case Type::INT96: return 12;
              case Type::FLOAT: return 4;
              case Type::DOUBLE: return 8;
              case Type::BYTE_ARRAY: {
                auto const col_type = data_col.type().id();
                if (col_type == type_id::STRING) {
                  // Strings are stored as 4 byte length + string bytes
                  return 4 + data_col.element<string_view>(val_idx).size_bytes();
                } else if (col_type == type_id::LIST) {
                  // Binary is stored as 4 byte length + bytes
                  return 4 +
                         get_element<statistics::byte_array_view>(data_col, val_idx).size_bytes();
                }
                CUDF_UNREACHABLE(
                  "Byte array only supports string and list<byte> column types for dictionary "
                  "encoding!");
              }
              case Type::FIXED_LEN_BYTE_ARRAY:
                if (data_col.type().id() == type_id::DECIMAL128) { return sizeof(__int128_t); }
                CUDF_UNREACHABLE(
                  "Fixed length byte array only supports decimal 128 column types for dictionary "
                  "encoding!");
              default: CUDF_UNREACHABLE("Unsupported type for dictionary encoding");
            }
          }();
        }
        // Reduce num_unique and uniq_data_size from all tiles.
        auto num_unique = block_reduce(reduce_storage).Sum(is_unique);
        __syncthreads();
        auto uniq_data_size = block_reduce(reduce_storage).Sum(uniq_elem_size);
        // The first thread in the block atomically updates total num_unique and uniq_data_size
        if (t == 0) {
          total_num_dict_entries =
            chunk_num_dict_entries.fetch_add(num_unique, cuda::std::memory_order_relaxed);
          total_num_dict_entries += num_unique;
          chunk_uniq_data_size.fetch_add(uniq_data_size, cuda::std::memory_order_relaxed);
        }
        __syncthreads();

        // Check if the num unique values in chunk has already exceeded max dict size and early exit
        if (total_num_dict_entries > MAX_DICT_SIZE) { return; }
      }  // for loop
    } else {
      CUDF_UNREACHABLE("Unsupported type to insert in map");
    }
  }
};

template <int block_size>
struct map_find_fn {
  storage_ref_type const& storage_ref;
  EncColumnChunk* const& chunk;
  template <typename T>
  __device__ void operator()(size_type const s_start_value_idx,
                             size_type const end_value_idx,
                             size_type const s_ck_start_val_idx)
  {
    if constexpr (column_device_view::has_element_accessor<T>()) {
      auto const col                     = chunk->col_desc;
      column_device_view const& data_col = *col->leaf_column;

      using equality_fn_type = equality_functor<T>;
      using hash_fn_type     = hash_functor<T>;
      // Choosing `linear_probing` over `double_hashing` for slighhhtly better performance seen in
      // benchmarks.
      using probing_scheme_type = cuco::linear_probing<map_cg_size, hash_fn_type>;

      // Make a view of the hash map.
      auto hash_map_ref = cuco::static_map_ref{cuco::empty_key{KEY_SENTINEL},
                                               cuco::empty_value{VALUE_SENTINEL},
                                               equality_fn_type{data_col},
                                               probing_scheme_type{hash_fn_type{data_col}},
                                               cuco::thread_scope_block,
                                               storage_ref};

      // Create a map ref with `cuco::find` operator
      auto const map_find_ref = hash_map_ref.rebind_operators(cuco::find);
      auto const t            = threadIdx.x;

      // Note: Adjust the following loop to use `cg::tiles<map_cg_size>` if needed in the future.
      for (thread_index_type val_idx = s_start_value_idx + t; val_idx < end_value_idx;
           val_idx += block_size) {
        // Find the key using a single thread for best performance for now.
        if (data_col.is_valid(val_idx)) {
          auto const found_slot = map_find_ref.find(val_idx);
          // Fail if we didn't find the previously inserted key.
          cudf_assert(found_slot != map_find_ref.end() &&
                      "Unable to find value in map in dictionary index construction");
          // No need for atomic as this is not going to be modified by any other thread.
          chunk->dict_index[val_idx - s_ck_start_val_idx] = found_slot->second;
        }
      }
    } else {
      CUDF_UNREACHABLE("Unsupported type to find in map");
    }
  }
};

template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size)
  populate_chunk_hash_maps_kernel(device_span<slot_type> const map_storage,
                                  cudf::detail::device_2dspan<PageFragment const> frags)
{
  auto const col_idx = blockIdx.y;
  auto const block_x = blockIdx.x;
  auto const frag    = frags[col_idx][block_x];
  auto chunk         = frag.chunk;
  auto col           = chunk->col_desc;

  if (not chunk->use_dictionary) { return; }

  size_type start_row = frag.start_row;
  size_type end_row   = frag.start_row + frag.num_rows;

  // Find the bounds of values in leaf column to be inserted into the map for current chunk.
  size_type const s_start_value_idx = row_to_value_idx(start_row, *col);
  size_type const end_value_idx     = row_to_value_idx(end_row, *col);

  column_device_view const& data_col = *col->leaf_column;
  storage_ref_type const storage_ref{chunk->dict_map_size,
                                     map_storage.data() + chunk->dict_map_offset};
  type_dispatcher(data_col.type(),
                  map_insert_fn<block_size>{storage_ref, chunk},
                  s_start_value_idx,
                  end_value_idx);
}

// Assigns monotone `dict_id`s bucketed by the fragment-hint that
// `populate_chunk_hash_maps_kernel` stamped into every slot's `second` field.
// After this kernel, `slot.second` has been overwritten with the final
// `dict_id` (an index into `chunk.dict_data`), and all values first inserted
// by fragment `f_i` are assigned `dict_id`s strictly less than values first
// inserted by fragment `f_j` for `f_i < f_j`. Per-page max `dict_index` is
// therefore monotone in page order whenever the underlying value distribution
// is, which is the invariant Phase 2 exploits for per-page bit-width savings.
//
// Algorithm (one block per chunk, three passes over `dict_map_size` slots):
//   1. Histogram: count non-empty slots by (fragment_hint - f_start).
//   2. BlockScan the histogram to exclusive-prefix offsets.
//   3. Claim `dict_id = atomicAdd(&fragment_cursor[bucket], 1)` per slot,
//      overwrite slot.second, and write `dict_data[dict_id] = key`.
//
// Three passes of `dict_map_size / block_size` iterations are cheaper than
// the alternative of launching a separate `cub::DeviceHistogram` +
// `DeviceScan` per chunk (kernel-launch overhead dominates for many small
// chunks; see PROBLEM.md §5.1 and PHASE_1_ORDERING.md §1.2.2).
template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size)
  collect_map_entries_kernel(device_span<slot_type> const map_storage,
                             device_span<EncColumnChunk> chunks,
                             cudf::detail::device_2dspan<PageFragment const> frags)
{
  static_assert(block_size >= MAX_FRAGMENTS_PER_BLOCK,
                "block_size must be >= MAX_FRAGMENTS_PER_BLOCK so one BlockScan thread backs "
                "each histogram bucket.");

  auto& chunk = chunks[blockIdx.x];
  if (not chunk.use_dictionary) { return; }

  auto const t        = threadIdx.x;
  auto const col_idx  = chunk.col_desc_id;
  auto const col_frag = frags[col_idx];

  // Resolve the chunk's column-relative fragment range [f_start, f_start + n_frags).
  // `chunk.fragments` points into `col_frag` (set in writer_impl.cu during
  // row_group_fragments setup, before build_chunk_dictionaries runs), so the
  // difference is the first fragment index stamped into any of this chunk's
  // slots by populate_chunk_hash_maps_kernel.
  __shared__ size_type f_start;
  __shared__ size_type n_frags;
  if (t == 0) {
    f_start = static_cast<size_type>(chunk.fragments - col_frag.data());
    size_type n = 0;
    auto const total_col_frags = static_cast<size_type>(col_frag.size());
    while (f_start + n < total_col_frags && col_frag[f_start + n].chunk == &chunk) {
      ++n;
    }
    n_frags = n;
  }
  __syncthreads();

  cudf_assert(n_frags > 0 && n_frags <= MAX_FRAGMENTS_PER_BLOCK &&
              "Fragments-per-chunk out of range for shared-memory histogram; raise "
              "MAX_FRAGMENTS_PER_BLOCK or add a global-memory fallback.");

  // `fragment_count` starts as the per-bucket histogram, becomes per-bucket
  // exclusive offsets after the scan, then is preserved as a reference for
  // the MAX_DICT_SIZE overflow assert in Pass 3. `fragment_cursor` mirrors
  // the offsets and is what threads `atomicAdd` into to claim dict_ids.
  __shared__ size_type fragment_count[MAX_FRAGMENTS_PER_BLOCK];
  __shared__ size_type fragment_cursor[MAX_FRAGMENTS_PER_BLOCK];

  using block_scan = cub::BlockScan<size_type, block_size>;
  __shared__ typename block_scan::TempStorage scan_storage;

  // Zero the histogram (only the live range; the tail past n_frags is
  // untouched and never read).
  if (t < n_frags) { fragment_count[t] = 0; }
  __syncthreads();

  auto* const chunk_slots  = map_storage.data() + chunk.dict_map_offset;
  auto const dict_map_size = chunk.dict_map_size;

  // Pass 1: histogram slot.second values into per-fragment buckets.
  // `slot.second` is the column-relative fragment index that won the insert in
  // `populate_chunk_hash_maps_kernel`; subtracting `f_start` normalizes it into
  // a chunk-local bucket index `[0, n_frags)`.
  for (size_type slot_idx = t; slot_idx < dict_map_size; slot_idx += block_size) {
    auto const slot_key = chunk_slots[slot_idx].first;
    if (slot_key != KEY_SENTINEL) {
      auto const frag_local = chunk_slots[slot_idx].second - f_start;
      cudf_assert(frag_local >= 0 && frag_local < n_frags &&
                  "populate stamped a fragment hint outside this chunk's fragment range");
      atomicAdd(&fragment_count[frag_local], 1);
    }
  }
  __syncthreads();

  // Pass 2: in-block exclusive scan of the histogram -> offsets -> cursors.
  {
    size_type const per_thread_count = (t < n_frags) ? fragment_count[t] : 0;
    size_type per_thread_offset      = 0;
    block_scan(scan_storage).ExclusiveSum(per_thread_count, per_thread_offset);
    if (t < n_frags) {
      fragment_count[t]  = per_thread_offset;
      fragment_cursor[t] = per_thread_offset;
    }
  }
  __syncthreads();

  // Pass 3: claim a dict_id in the bucket, overwrite slot.second, materialize
  // dict_data. The atomicAdd is shared-memory only, so no global traffic.
  for (size_type slot_idx = t; slot_idx < dict_map_size; slot_idx += block_size) {
    auto const slot_key = chunk_slots[slot_idx].first;
    if (slot_key != KEY_SENTINEL) {
      auto const frag_local = chunk_slots[slot_idx].second - f_start;
      auto const loc        = atomicAdd(&fragment_cursor[frag_local], 1);
      cudf_assert(loc < MAX_DICT_SIZE && "Number of filled slots exceeds max dict size");
      chunk.dict_data[loc]          = slot_key;
      chunk_slots[slot_idx].second  = loc;
    }
  }
}

template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size)
  get_dictionary_indices_kernel(device_span<slot_type> const map_storage,
                                cudf::detail::device_2dspan<PageFragment const> frags)
{
  auto const col_idx = blockIdx.y;
  auto const block_x = blockIdx.x;
  auto const frag    = frags[col_idx][block_x];
  auto chunk         = frag.chunk;

  if (not chunk->use_dictionary) { return; }

  size_type start_row = frag.start_row;
  size_type end_row   = frag.start_row + frag.num_rows;

  auto const col = chunk->col_desc;
  // Find the bounds of values in leaf column to be searched in the map for current chunk
  auto const s_start_value_idx  = row_to_value_idx(start_row, *col);
  auto const s_ck_start_val_idx = row_to_value_idx(chunk->start_row, *col);
  auto const end_value_idx      = row_to_value_idx(end_row, *col);

  column_device_view const& data_col = *col->leaf_column;
  storage_ref_type const storage_ref{chunk->dict_map_size,
                                     map_storage.data() + chunk->dict_map_offset};

  type_dispatcher(data_col.type(),
                  map_find_fn<block_size>{storage_ref, chunk},
                  s_start_value_idx,
                  end_value_idx,
                  s_ck_start_val_idx);
}

void populate_chunk_hash_maps(device_span<slot_type> const map_storage,
                              cudf::detail::device_2dspan<PageFragment const> frags,
                              rmm::cuda_stream_view stream)
{
  dim3 const dim_grid(frags.size().second, frags.size().first);
  populate_chunk_hash_maps_kernel<DEFAULT_BLOCK_SIZE>
    <<<dim_grid, DEFAULT_BLOCK_SIZE, 0, stream.value()>>>(map_storage, frags);
}

void collect_map_entries(device_span<slot_type> const map_storage,
                         device_span<EncColumnChunk> chunks,
                         cudf::detail::device_2dspan<PageFragment const> frags,
                         rmm::cuda_stream_view stream)
{
  constexpr int block_size = 1024;
  collect_map_entries_kernel<block_size>
    <<<chunks.size(), block_size, 0, stream.value()>>>(map_storage, chunks, frags);
}

void get_dictionary_indices(device_span<slot_type> const map_storage,
                            cudf::detail::device_2dspan<PageFragment const> frags,
                            rmm::cuda_stream_view stream)
{
  dim3 const dim_grid(frags.size().second, frags.size().first);
  get_dictionary_indices_kernel<DEFAULT_BLOCK_SIZE>
    <<<dim_grid, DEFAULT_BLOCK_SIZE, 0, stream.value()>>>(map_storage, frags);
}
}  // namespace cudf::io::parquet::detail
