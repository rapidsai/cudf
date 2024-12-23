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

#include <cuco/static_map_ref.cuh>
#include <cuda/atomic>

namespace cudf::io::parquet::detail {

namespace {
constexpr int DEFAULT_BLOCK_SIZE = 256;
}

template <typename T>
struct equality_functor {
  column_device_view const& col;
  __device__ bool operator()(key_type lhs_idx, key_type rhs_idx) const
  {
    // We don't call this for nulls so this is fine.
    auto constexpr equal =
      cudf::experimental::row::equality::nan_equal_physical_equality_comparator{};
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
          is_unique      = map_insert_ref.insert(cuco::pair{val_idx, val_idx});
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
  populate_chunk_hash_maps_kernel(device_span<bucket_type> const map_storage,
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

template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size)
  collect_map_entries_kernel(device_span<bucket_type> const map_storage,
                             device_span<EncColumnChunk> chunks)
{
  auto& chunk = chunks[blockIdx.x];
  if (not chunk.use_dictionary) { return; }

  auto t = threadIdx.x;
  __shared__ cuda::atomic<size_type, SCOPE> counter;
  using cuda::std::memory_order_relaxed;
  if (t == 0) { new (&counter) cuda::atomic<size_type, SCOPE>{0}; }
  __syncthreads();

  // Iterate over all buckets in the map.
  for (; t < chunk.dict_map_size; t += block_size) {
    auto bucket = map_storage.data() + chunk.dict_map_offset + t;
    // Collect all slots from each bucket.
    for (auto& slot : *bucket) {
      auto const key = slot.first;
      if (key != KEY_SENTINEL) {
        auto const loc = counter.fetch_add(1, memory_order_relaxed);
        cudf_assert(loc < MAX_DICT_SIZE && "Number of filled slots exceeds max dict size");
        chunk.dict_data[loc] = key;
        // If sorting dict page ever becomes a hard requirement, enable the following statement
        // and add a dict sorting step before storing into the slot's second field.
        // chunk.dict_data_idx[loc] = idx;
        slot.second = loc;
      }
    }
  }
}

template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size)
  get_dictionary_indices_kernel(device_span<bucket_type> const map_storage,
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

void populate_chunk_hash_maps(device_span<bucket_type> const map_storage,
                              cudf::detail::device_2dspan<PageFragment const> frags,
                              rmm::cuda_stream_view stream)
{
  dim3 const dim_grid(frags.size().second, frags.size().first);
  populate_chunk_hash_maps_kernel<DEFAULT_BLOCK_SIZE>
    <<<dim_grid, DEFAULT_BLOCK_SIZE, 0, stream.value()>>>(map_storage, frags);
}

void collect_map_entries(device_span<bucket_type> const map_storage,
                         device_span<EncColumnChunk> chunks,
                         rmm::cuda_stream_view stream)
{
  constexpr int block_size = 1024;
  collect_map_entries_kernel<block_size>
    <<<chunks.size(), block_size, 0, stream.value()>>>(map_storage, chunks);
}

void get_dictionary_indices(device_span<bucket_type> const map_storage,
                            cudf::detail::device_2dspan<PageFragment const> frags,
                            rmm::cuda_stream_view stream)
{
  dim3 const dim_grid(frags.size().second, frags.size().first);
  get_dictionary_indices_kernel<DEFAULT_BLOCK_SIZE>
    <<<dim_grid, DEFAULT_BLOCK_SIZE, 0, stream.value()>>>(map_storage, frags);
}
}  // namespace cudf::io::parquet::detail
