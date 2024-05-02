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

#include <cuda/atomic>

namespace cudf::io::parquet::detail {

namespace {
constexpr int DEFAULT_BLOCK_SIZE = 256;
}

template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size)
  initialize_chunk_hash_maps_kernel(device_span<EncColumnChunk> chunks)
{
  auto const chunk = chunks[blockIdx.x];
  auto const t     = threadIdx.x;
  // fut: Now that per-chunk dict is same size as ck.num_values, try to not use one block per chunk
  for (thread_index_type i = 0; i < chunk.dict_map_size; i += block_size) {
    if (t + i < chunk.dict_map_size) {
      new (&chunk.dict_map_slots[t + i].first) map_type::atomic_key_type{KEY_SENTINEL};
      new (&chunk.dict_map_slots[t + i].second) map_type::atomic_mapped_type{VALUE_SENTINEL};
    }
  }
}

template <typename T>
struct equality_functor {
  column_device_view const& col;
  __device__ bool operator()(size_type lhs_idx, size_type rhs_idx)
  {
    // We don't call this for nulls so this is fine
    auto const equal = cudf::experimental::row::equality::nan_equal_physical_equality_comparator{};
    return equal(col.element<T>(lhs_idx), col.element<T>(rhs_idx));
  }
};

template <typename T>
struct hash_functor {
  column_device_view const& col;
  __device__ auto operator()(size_type idx) const
  {
    return cudf::hashing::detail::MurmurHash3_x86_32<T>{}(col.element<T>(idx));
  }
};

struct map_insert_fn {
  map_type::device_mutable_view& map;

  template <typename T>
  __device__ bool operator()(column_device_view const& col, size_type i)
  {
    if constexpr (column_device_view::has_element_accessor<T>()) {
      auto hash_fn     = hash_functor<T>{col};
      auto equality_fn = equality_functor<T>{col};
      return map.insert(std::pair(i, i), hash_fn, equality_fn);
    } else {
      CUDF_UNREACHABLE("Unsupported type to insert in map");
    }
  }
};

struct map_find_fn {
  map_type::device_view& map;

  template <typename T>
  __device__ map_type::device_view::iterator operator()(column_device_view const& col, size_type i)
  {
    if constexpr (column_device_view::has_element_accessor<T>()) {
      auto hash_fn     = hash_functor<T>{col};
      auto equality_fn = equality_functor<T>{col};
      return map.find(i, hash_fn, equality_fn);
    } else {
      CUDF_UNREACHABLE("Unsupported type to find in map");
    }
  }
};

template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size)
  populate_chunk_hash_maps_kernel(cudf::detail::device_2dspan<PageFragment const> frags)
{
  auto col_idx = blockIdx.y;
  auto block_x = blockIdx.x;
  auto t       = threadIdx.x;
  auto frag    = frags[col_idx][block_x];
  auto chunk   = frag.chunk;
  auto col     = chunk->col_desc;

  if (not chunk->use_dictionary) { return; }

  using block_reduce = cub::BlockReduce<size_type, block_size>;
  __shared__ typename block_reduce::TempStorage reduce_storage;

  size_type start_row = frag.start_row;
  size_type end_row   = frag.start_row + frag.num_rows;

  // Find the bounds of values in leaf column to be inserted into the map for current chunk
  size_type const s_start_value_idx = row_to_value_idx(start_row, *col);
  size_type const end_value_idx     = row_to_value_idx(end_row, *col);

  column_device_view const& data_col = *col->leaf_column;

  // Make a view of the hash map
  auto hash_map_mutable = map_type::device_mutable_view(chunk->dict_map_slots,
                                                        chunk->dict_map_size,
                                                        cuco::empty_key{KEY_SENTINEL},
                                                        cuco::empty_value{VALUE_SENTINEL});

  __shared__ size_type total_num_dict_entries;
  thread_index_type val_idx = s_start_value_idx + t;
  while (val_idx - block_size < end_value_idx) {
    auto const is_valid =
      val_idx < end_value_idx and val_idx < data_col.size() and data_col.is_valid(val_idx);

    // insert element at val_idx to hash map and count successful insertions
    size_type is_unique      = 0;
    size_type uniq_elem_size = 0;
    if (is_valid) {
      is_unique =
        type_dispatcher(data_col.type(), map_insert_fn{hash_map_mutable}, data_col, val_idx);
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
              return 4 + get_element<statistics::byte_array_view>(data_col, val_idx).size_bytes();
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

    auto num_unique = block_reduce(reduce_storage).Sum(is_unique);
    __syncthreads();
    auto uniq_data_size = block_reduce(reduce_storage).Sum(uniq_elem_size);
    if (t == 0) {
      total_num_dict_entries = atomicAdd(&chunk->num_dict_entries, num_unique);
      total_num_dict_entries += num_unique;
      atomicAdd(&chunk->uniq_data_size, uniq_data_size);
    }
    __syncthreads();

    // Check if the num unique values in chunk has already exceeded max dict size and early exit
    if (total_num_dict_entries > MAX_DICT_SIZE) { return; }

    val_idx += block_size;
  }  // while
}

template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size)
  collect_map_entries_kernel(device_span<EncColumnChunk> chunks)
{
  auto& chunk = chunks[blockIdx.x];
  if (not chunk.use_dictionary) { return; }

  auto t   = threadIdx.x;
  auto map = map_type::device_view(chunk.dict_map_slots,
                                   chunk.dict_map_size,
                                   cuco::empty_key{KEY_SENTINEL},
                                   cuco::empty_value{VALUE_SENTINEL});

  __shared__ cuda::atomic<size_type, cuda::thread_scope_block> counter;
  using cuda::std::memory_order_relaxed;
  if (t == 0) { new (&counter) cuda::atomic<size_type, cuda::thread_scope_block>{0}; }
  __syncthreads();
  for (size_type i = 0; i < chunk.dict_map_size; i += block_size) {
    if (t + i < chunk.dict_map_size) {
      auto* slot = reinterpret_cast<map_type::value_type*>(map.begin_slot() + t + i);
      auto key   = slot->first;
      if (key != KEY_SENTINEL) {
        auto loc = counter.fetch_add(1, memory_order_relaxed);
        cudf_assert(loc < MAX_DICT_SIZE && "Number of filled slots exceeds max dict size");
        chunk.dict_data[loc] = key;
        // If sorting dict page ever becomes a hard requirement, enable the following statement and
        // add a dict sorting step before storing into the slot's second field.
        // chunk.dict_data_idx[loc] = t + i;
        slot->second = loc;
      }
    }
  }
}

template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size)
  get_dictionary_indices_kernel(cudf::detail::device_2dspan<PageFragment const> frags)
{
  auto col_idx = blockIdx.y;
  auto block_x = blockIdx.x;
  auto t       = threadIdx.x;
  auto frag    = frags[col_idx][block_x];
  auto chunk   = frag.chunk;
  auto col     = chunk->col_desc;

  if (not chunk->use_dictionary) { return; }

  size_type start_row = frag.start_row;
  size_type end_row   = frag.start_row + frag.num_rows;

  // Find the bounds of values in leaf column to be searched in the map for current chunk
  auto const s_start_value_idx  = row_to_value_idx(start_row, *col);
  auto const s_ck_start_val_idx = row_to_value_idx(chunk->start_row, *col);
  auto const end_value_idx      = row_to_value_idx(end_row, *col);

  column_device_view const& data_col = *col->leaf_column;

  auto map = map_type::device_view(chunk->dict_map_slots,
                                   chunk->dict_map_size,
                                   cuco::empty_key{KEY_SENTINEL},
                                   cuco::empty_value{VALUE_SENTINEL});

  thread_index_type val_idx = s_start_value_idx + t;
  while (val_idx < end_value_idx) {
    if (data_col.is_valid(val_idx)) {
      auto found_slot = type_dispatcher(data_col.type(), map_find_fn{map}, data_col, val_idx);
      cudf_assert(found_slot != map.end() &&
                  "Unable to find value in map in dictionary index construction");
      if (found_slot != map.end()) {
        // No need for atomic as this is not going to be modified by any other thread
        auto* val_ptr = reinterpret_cast<map_type::mapped_type*>(&found_slot->second);
        chunk->dict_index[val_idx - s_ck_start_val_idx] = *val_ptr;
      }
    }

    val_idx += block_size;
  }
}

void initialize_chunk_hash_maps(device_span<EncColumnChunk> chunks, rmm::cuda_stream_view stream)
{
  constexpr int block_size = 1024;
  initialize_chunk_hash_maps_kernel<block_size>
    <<<chunks.size(), block_size, 0, stream.value()>>>(chunks);
}

void populate_chunk_hash_maps(cudf::detail::device_2dspan<PageFragment const> frags,
                              rmm::cuda_stream_view stream)
{
  dim3 const dim_grid(frags.size().second, frags.size().first);
  populate_chunk_hash_maps_kernel<DEFAULT_BLOCK_SIZE>
    <<<dim_grid, DEFAULT_BLOCK_SIZE, 0, stream.value()>>>(frags);
}

void collect_map_entries(device_span<EncColumnChunk> chunks, rmm::cuda_stream_view stream)
{
  constexpr int block_size = 1024;
  collect_map_entries_kernel<block_size><<<chunks.size(), block_size, 0, stream.value()>>>(chunks);
}

void get_dictionary_indices(cudf::detail::device_2dspan<PageFragment const> frags,
                            rmm::cuda_stream_view stream)
{
  dim3 const dim_grid(frags.size().second, frags.size().first);
  get_dictionary_indices_kernel<DEFAULT_BLOCK_SIZE>
    <<<dim_grid, DEFAULT_BLOCK_SIZE, 0, stream.value()>>>(frags);
}
}  // namespace cudf::io::parquet::detail
