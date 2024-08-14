/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
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

#include "parquet_gpu.cuh"

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/table/experimental/row_operators.cuh>

#include <rmm/exec_policy.hpp>

#include <cooperative_groups.h>
#include <cuco/static_map_ref.cuh>
#include <cuda/atomic>

namespace cudf::io::parquet::detail {

namespace cg = cooperative_groups;

namespace {
constexpr int DEFAULT_BLOCK_SIZE = 256;
}

template <typename T>
struct equality_functor {
  column_device_view const& col;
  __device__ bool operator()(key_type const lhs_idx, key_type const rhs_idx) const
  {
    // We don't call this for nulls so this is fine.
    auto const equal = cudf::experimental::row::equality::nan_equal_physical_equality_comparator{};
    return equal(col.element<T>(lhs_idx), col.element<T>(rhs_idx));
  }
};

template <typename T>
struct hash_functor {
  column_device_view const& col;
  __device__ auto operator()(key_type idx) const
  {
    return cudf::hashing::detail::MurmurHash3_x86_32<T>{}(col.element<T>(idx));
  }
};

struct map_insert_fn {
  storage_ref_type const& storage_ref;

  template <typename T>
  __device__ bool operator()(column_device_view const& col,
                             cg::thread_block_tile<map_cg_size> const& tile,
                             key_type i)
  {
    if constexpr (column_device_view::has_element_accessor<T>()) {
      using equality_fn_type    = equality_functor<T>;
      using hash_fn_type        = hash_functor<T>;
      using probing_scheme_type = cuco::linear_probing<map_cg_size, hash_fn_type>;

      // Instantiate hash and equality functors.
      auto hash_fn  = hash_fn_type{col};
      auto equal_fn = equality_fn_type{col};

      // Make a view of the hash map.
      cuco::static_map_ref<key_type,
                           mapped_type,
                           SCOPE,
                           equality_fn_type,
                           probing_scheme_type,
                           storage_ref_type>
        hash_map_ref{cuco::empty_key{KEY_SENTINEL},
                     cuco::empty_value{VALUE_SENTINEL},
                     {equal_fn},
                     {hash_fn},
                     {},
                     storage_ref};

      // Create another map ref with the insert operator.
      auto map_insert_ref = hash_map_ref.with_operators(cuco::insert);
      // Insert into the hash map using the provided thread tile.
      return map_insert_ref.insert(tile, cuco::pair{i, i});
    } else {
      CUDF_UNREACHABLE("Unsupported type to insert in map");
    }
  }
};  // namespace cudf::io::parquet::detail

struct map_find_fn {
  storage_ref_type const& storage_ref;

  template <typename T>
  __device__ cuco::pair<key_type, mapped_type> operator()(
    column_device_view const& col, cg::thread_block_tile<map_cg_size> const& tile, key_type i)
  {
    if constexpr (column_device_view::has_element_accessor<T>()) {
      using equality_fn_type    = equality_functor<T>;
      using hash_fn_type        = hash_functor<T>;
      using probing_scheme_type = cuco::linear_probing<map_cg_size, hash_fn_type>;

      // Instantiate hash and equality functors.
      auto hash_fn  = hash_fn_type{col};
      auto equal_fn = equality_fn_type{col};

      // Make a view of the hash map
      cuco::static_map_ref<key_type,
                           mapped_type,
                           SCOPE,
                           equality_fn_type,
                           probing_scheme_type,
                           storage_ref_type>
        hash_map_ref{cuco::empty_key{KEY_SENTINEL},
                     cuco::empty_value{VALUE_SENTINEL},
                     {equal_fn},
                     {hash_fn},
                     {},
                     storage_ref};

      // Create another map with find operator.
      auto map_find_ref = hash_map_ref.with_operators(cuco::find);

      // Find the key = i using the provided thread tile.
      auto found_slot = map_find_ref.find(tile, i);

      // Check if didn't find the previously inserted key.
      if (tile.thread_rank() == 0) {
        cudf_assert(found_slot != map_find_ref.end() &&
                    "Unable to find value in map in dictionary index construction");
      }
      // Return a pair of the found key and value.
      return {found_slot->first, found_slot->second};
    } else {
      CUDF_UNREACHABLE("Unsupported type to find in map");
    }
  }
};

template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size)
  populate_chunk_hash_maps_kernel(window_type* const map_storage,
                                  cudf::detail::device_2dspan<PageFragment const> frags)
{
  auto const col_idx = blockIdx.y;
  auto const block_x = blockIdx.x;
  auto const frag    = frags[col_idx][block_x];
  auto chunk         = frag.chunk;
  auto col           = chunk->col_desc;

  if (not chunk->use_dictionary) { return; }

  using block_reduce = cub::BlockReduce<size_type, block_size>;
  __shared__ typename block_reduce::TempStorage reduce_storage;

  auto const block = cg::this_thread_block();
  auto const tile  = cg::tiled_partition<map_cg_size>(block);

  size_type start_row = frag.start_row;
  size_type end_row   = frag.start_row + frag.num_rows;

  // Find the bounds of values in leaf column to be inserted into the map for current chunk.
  size_type const s_start_value_idx = row_to_value_idx(start_row, *col);
  size_type const end_value_idx     = row_to_value_idx(end_row, *col);

  column_device_view const& data_col = *col->leaf_column;
  storage_ref_type const storage_ref{chunk->dict_map_size, map_storage + chunk->dict_map_offset};

  __shared__ size_type total_num_dict_entries;

  // Insert all column chunk elements to the hash map to build the dict.
  for (thread_index_type val_idx = s_start_value_idx;
       val_idx + block.thread_rank() - block_size < end_value_idx;
       val_idx += block_size) {
    // Compute the index to the start of the tile.
    size_type is_unique      = 0;
    size_type uniq_elem_size = 0;

    // Insert all elements within each tile.
    for (auto tile_offset = 0; tile_offset < tile.num_threads(); tile_offset++) {
      // Compute the index to the element being inserted within the tile.
      auto const tile_val_idx =
        val_idx + tile_offset + (tile.meta_group_rank() * tile.num_threads());

      // Check if this index is valid.
      auto const is_valid = tile_val_idx < end_value_idx and tile_val_idx < data_col.size() and
                            data_col.is_valid(tile_val_idx);

      // Insert tile_val_idx to hash map and count successful insertions.
      if (is_valid) {
        // Insert the element to the map using the entire tile.
        auto const tile_is_unique = type_dispatcher(
          data_col.type(), map_insert_fn{storage_ref}, data_col, tile, tile_val_idx);

        // tile_offset'th thread in the tile updates its number and size of unique element.
        if (tile.thread_rank() == tile_offset) {
          is_unique      = tile_is_unique;
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
                  return 4 + data_col.element<string_view>(tile_val_idx).size_bytes();
                } else if (col_type == type_id::LIST) {
                  // Binary is stored as 4 byte length + bytes
                  return 4 + get_element<statistics::byte_array_view>(data_col, tile_val_idx)
                               .size_bytes();
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
      }
    }
    // All elements in all tiles inserted, reduce num_unique and uniq_data_size across the block.
    auto num_unique = block_reduce(reduce_storage).Sum(is_unique);
    block.sync();
    auto uniq_data_size = block_reduce(reduce_storage).Sum(uniq_elem_size);
    // The first thread in the block atomically updates total num_unique and uniq_data_size
    if (block.thread_rank() == 0) {
      total_num_dict_entries = atomicAdd(&chunk->num_dict_entries, num_unique);
      total_num_dict_entries += num_unique;
      atomicAdd(&chunk->uniq_data_size, uniq_data_size);
    }
    block.sync();

    // Check if the num unique values in chunk has already exceeded max dict size and early exit
    if (total_num_dict_entries > MAX_DICT_SIZE) { return; }
  }  // for loop
}

template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size)
  collect_map_entries_kernel(window_type* const map_storage, device_span<EncColumnChunk> chunks)
{
  auto& chunk = chunks[blockIdx.x];
  if (not chunk.use_dictionary) { return; }

  auto const t = threadIdx.x;

  __shared__ cuda::atomic<size_type, SCOPE> counter;
  using cuda::std::memory_order_relaxed;
  if (t == 0) { new (&counter) cuda::atomic<size_type, SCOPE>{0}; }
  __syncthreads();

  for (size_type i = 0; i < chunk.dict_map_size; i += block_size) {
    if (t + i < chunk.dict_map_size) {
      auto* slot     = map_storage + chunk.dict_map_offset + t + i;
      auto const key = slot->data()->first;
      if (key != KEY_SENTINEL) {
        auto loc = counter.fetch_add(1, memory_order_relaxed);
        cudf_assert(loc < MAX_DICT_SIZE && "Number of filled slots exceeds max dict size");
        chunk.dict_data[loc] = key;
        // If sorting dict page ever becomes a hard requirement, enable the following statement and
        // add a dict sorting step before storing into the slot's second field.
        // chunk.dict_data_idx[loc] = t + i;
        slot->data()->second = loc;
      }
    }
  }
}

template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size)
  get_dictionary_indices_kernel(window_type* const map_storage,
                                cudf::detail::device_2dspan<PageFragment const> frags)
{
  auto const col_idx = blockIdx.y;
  auto const block_x = blockIdx.x;
  auto const frag    = frags[col_idx][block_x];
  auto chunk         = frag.chunk;
  auto const col     = chunk->col_desc;

  if (not chunk->use_dictionary) { return; }

  auto const block  = cg::this_thread_block();
  auto const tile   = cg::tiled_partition<map_cg_size>(block);
  auto const ntiles = tile.meta_group_size();

  size_type start_row = frag.start_row;
  size_type end_row   = frag.start_row + frag.num_rows;

  // Find the bounds of values in leaf column to be searched in the map for current chunk
  auto const s_start_value_idx  = row_to_value_idx(start_row, *col);
  auto const s_ck_start_val_idx = row_to_value_idx(chunk->start_row, *col);
  auto const end_value_idx      = row_to_value_idx(end_row, *col);

  column_device_view const& data_col = *col->leaf_column;
  storage_ref_type const storage_ref{chunk->dict_map_size, map_storage + chunk->dict_map_offset};

  for (thread_index_type val_idx = s_start_value_idx + tile.meta_group_rank();
       val_idx < end_value_idx;
       val_idx += ntiles) {
    if (data_col.is_valid(val_idx)) {
      auto [found_key, found_value] =
        type_dispatcher(data_col.type(), map_find_fn{storage_ref}, data_col, tile, val_idx);
      // First thread in the tile updates the dict_index
      if (tile.thread_rank() == 0) {
        // No need for atomic as this is not going to be modified by any other thread
        chunk->dict_index[val_idx - s_ck_start_val_idx] = found_value;
      }
    }
  }
}

void populate_chunk_hash_maps(window_type* const map_storage,
                              cudf::detail::device_2dspan<PageFragment const> frags,
                              rmm::cuda_stream_view stream)
{
  dim3 const dim_grid(frags.size().second, frags.size().first);
  populate_chunk_hash_maps_kernel<DEFAULT_BLOCK_SIZE>
    <<<dim_grid, DEFAULT_BLOCK_SIZE, 0, stream.value()>>>(map_storage, frags);
}

void collect_map_entries(window_type* const map_storage,
                         device_span<EncColumnChunk> chunks,
                         rmm::cuda_stream_view stream)
{
  constexpr int block_size = 1024;
  collect_map_entries_kernel<block_size>
    <<<chunks.size(), block_size, 0, stream.value()>>>(map_storage, chunks);
}

void get_dictionary_indices(window_type* const map_storage,
                            cudf::detail::device_2dspan<PageFragment const> frags,
                            rmm::cuda_stream_view stream)
{
  dim3 const dim_grid(frags.size().second, frags.size().first);
  get_dictionary_indices_kernel<DEFAULT_BLOCK_SIZE>
    <<<dim_grid, DEFAULT_BLOCK_SIZE, 0, stream.value()>>>(map_storage, frags);
}
}  // namespace cudf::io::parquet::detail
