/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include "orc_gpu.hpp"

#include <cudf/detail/offsets_iterator.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/orc_types.hpp>
#include <cudf/table/experimental/row_operators.cuh>

#include <rmm/cuda_stream_view.hpp>

namespace cudf::io::orc::gpu {

/**
 * @brief Counts the number of characters in each rowgroup of each string column.
 */
CUDF_KERNEL void rowgroup_char_counts_kernel(device_2dspan<size_type> char_counts,
                                             device_span<orc_column_device_view const> orc_columns,
                                             device_2dspan<rowgroup_rows const> rowgroup_bounds,
                                             device_span<uint32_t const> str_col_indexes)
{
  // Index of the column in the `str_col_indexes` array
  auto const str_col_idx = blockIdx.y;
  // Index of the column in the `orc_columns` array
  auto const col_idx       = str_col_indexes[str_col_idx];
  auto const row_group_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (row_group_idx >= rowgroup_bounds.size().first) { return; }

  auto const& str_col  = orc_columns[col_idx];
  auto const start_row = rowgroup_bounds[row_group_idx][col_idx].begin + str_col.offset();
  auto const num_rows  = rowgroup_bounds[row_group_idx][col_idx].size();

  size_type char_count = 0;
  if (num_rows > 0) {
    auto const& offsets    = str_col.child(strings_column_view::offsets_column_index);
    auto const offsets_itr = cudf::detail::input_offsetalator(offsets.head(), offsets.type());
    char_count = static_cast<size_type>(offsets_itr[start_row + num_rows] - offsets_itr[start_row]);
  }
  char_counts[str_col_idx][row_group_idx] = char_count;
}

void rowgroup_char_counts(device_2dspan<size_type> counts,
                          device_span<orc_column_device_view const> orc_columns,
                          device_2dspan<rowgroup_rows const> rowgroup_bounds,
                          device_span<uint32_t const> str_col_indexes,
                          rmm::cuda_stream_view stream)
{
  if (rowgroup_bounds.count() == 0) { return; }

  auto const num_rowgroups = rowgroup_bounds.size().first;
  auto const num_str_cols  = str_col_indexes.size();
  if (num_str_cols == 0) { return; }

  int block_size    = 0;  // suggested thread count to use
  int min_grid_size = 0;  // minimum block count required
  CUDF_CUDA_TRY(
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, rowgroup_char_counts_kernel));
  auto const grid_size =
    dim3(cudf::util::div_rounding_up_unsafe<unsigned int>(num_rowgroups, block_size),
         static_cast<unsigned int>(num_str_cols));

  rowgroup_char_counts_kernel<<<grid_size, block_size, 0, stream.value()>>>(
    counts, orc_columns, rowgroup_bounds, str_col_indexes);
}

template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size)
  initialize_dictionary_hash_maps_kernel(device_span<stripe_dictionary> dictionaries)
{
  auto const dict_map = dictionaries[blockIdx.x].map_slots;
  auto const t        = threadIdx.x;
  for (size_type i = 0; i < dict_map.size(); i += block_size) {
    if (t + i < dict_map.size()) {
      new (&dict_map[t + i].first) map_type::atomic_key_type{KEY_SENTINEL};
      new (&dict_map[t + i].second) map_type::atomic_mapped_type{VALUE_SENTINEL};
    }
  }
}

struct equality_functor {
  column_device_view const& col;
  __device__ bool operator()(size_type lhs_idx, size_type rhs_idx) const
  {
    // We don't call this for nulls so this is fine
    auto const equal = cudf::experimental::row::equality::nan_equal_physical_equality_comparator{};
    return equal(col.element<string_view>(lhs_idx), col.element<string_view>(rhs_idx));
  }
};

struct hash_functor {
  column_device_view const& col;
  __device__ auto operator()(size_type idx) const
  {
    return cudf::hashing::detail::MurmurHash3_x86_32<string_view>{}(col.element<string_view>(idx));
  }
};

template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size)
  populate_dictionary_hash_maps_kernel(device_2dspan<stripe_dictionary> dictionaries,
                                       device_span<orc_column_device_view const> columns)
{
  auto const col_idx    = blockIdx.x;
  auto const stripe_idx = blockIdx.y;
  auto const t          = threadIdx.x;
  auto& dict            = dictionaries[col_idx][stripe_idx];
  auto const& col       = columns[dict.column_idx];

  // Make a view of the hash map
  auto hash_map_mutable  = map_type::device_mutable_view(dict.map_slots.data(),
                                                        dict.map_slots.size(),
                                                        cuco::empty_key{KEY_SENTINEL},
                                                        cuco::empty_value{VALUE_SENTINEL});
  auto const hash_fn     = hash_functor{col};
  auto const equality_fn = equality_functor{col};

  auto const start_row = dict.start_row;
  auto const end_row   = dict.start_row + dict.num_rows;

  size_type entry_count{0};
  size_type char_count{0};
  // all threads should loop the same number of times
  for (thread_index_type cur_row = start_row + t; cur_row - t < end_row; cur_row += block_size) {
    auto const is_valid = cur_row < end_row and col.is_valid(cur_row);

    if (is_valid) {
      // insert element at cur_row to hash map and count successful insertions
      auto const is_unique =
        hash_map_mutable.insert(std::pair(cur_row, cur_row), hash_fn, equality_fn);

      if (is_unique) {
        ++entry_count;
        char_count += col.element<string_view>(cur_row).size_bytes();
      }
    }
    // ensure that threads access adjacent rows in each iteration
    __syncthreads();
  }

  using block_reduce = cub::BlockReduce<size_type, block_size>;
  __shared__ typename block_reduce::TempStorage reduce_storage;

  auto const block_entry_count = block_reduce(reduce_storage).Sum(entry_count);
  __syncthreads();
  auto const block_char_count = block_reduce(reduce_storage).Sum(char_count);

  if (t == 0) {
    dict.entry_count = block_entry_count;
    dict.char_count  = block_char_count;
  }
}

template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size)
  collect_map_entries_kernel(device_2dspan<stripe_dictionary> dictionaries)
{
  auto const col_idx    = blockIdx.x;
  auto const stripe_idx = blockIdx.y;
  auto const& dict      = dictionaries[col_idx][stripe_idx];

  if (not dict.is_enabled) { return; }

  auto const t = threadIdx.x;
  auto map     = map_type::device_view(dict.map_slots.data(),
                                   dict.map_slots.size(),
                                   cuco::empty_key{KEY_SENTINEL},
                                   cuco::empty_value{VALUE_SENTINEL});

  __shared__ cuda::atomic<size_type, cuda::thread_scope_block> counter;

  using cuda::std::memory_order_relaxed;
  if (t == 0) { new (&counter) cuda::atomic<size_type, cuda::thread_scope_block>{0}; }
  __syncthreads();
  for (size_type i = 0; i < dict.map_slots.size(); i += block_size) {
    if (t + i < dict.map_slots.size()) {
      auto* slot = reinterpret_cast<map_type::value_type*>(map.begin_slot() + t + i);
      auto key   = slot->first;
      if (key != KEY_SENTINEL) {
        auto loc       = counter.fetch_add(1, memory_order_relaxed);
        dict.data[loc] = key;
        slot->second   = loc;
      }
    }
  }
}

template <int block_size>
CUDF_KERNEL void __launch_bounds__(block_size)
  get_dictionary_indices_kernel(device_2dspan<stripe_dictionary> dictionaries,
                                device_span<orc_column_device_view const> columns)
{
  auto const col_idx    = blockIdx.x;
  auto const stripe_idx = blockIdx.y;
  auto const& dict      = dictionaries[col_idx][stripe_idx];
  auto const& col       = columns[dict.column_idx];

  if (not dict.is_enabled) { return; }

  auto const t         = threadIdx.x;
  auto const start_row = dict.start_row;
  auto const end_row   = dict.start_row + dict.num_rows;

  auto const map = map_type::device_view(dict.map_slots.data(),
                                         dict.map_slots.size(),
                                         cuco::empty_key{KEY_SENTINEL},
                                         cuco::empty_value{VALUE_SENTINEL});

  thread_index_type cur_row = start_row + t;
  while (cur_row < end_row) {
    if (col.is_valid(cur_row)) {
      auto const hash_fn     = hash_functor{col};
      auto const equality_fn = equality_functor{col};
      auto const found_slot  = map.find(cur_row, hash_fn, equality_fn);
      cudf_assert(found_slot != map.end() &&
                  "Unable to find value in map in dictionary index construction");
      if (found_slot != map.end()) {
        // No need for atomic as this is not going to be modified by any other thread
        auto const val_ptr  = reinterpret_cast<map_type::mapped_type const*>(&found_slot->second);
        dict.index[cur_row] = *val_ptr;
      }
    }
    cur_row += block_size;
  }
}

void initialize_dictionary_hash_maps(device_2dspan<stripe_dictionary> dictionaries,
                                     rmm::cuda_stream_view stream)
{
  if (dictionaries.count() == 0) { return; }
  constexpr int block_size = 1024;
  initialize_dictionary_hash_maps_kernel<block_size>
    <<<dictionaries.count(), block_size, 0, stream.value()>>>(dictionaries.flat_view());
}

void populate_dictionary_hash_maps(device_2dspan<stripe_dictionary> dictionaries,
                                   device_span<orc_column_device_view const> columns,
                                   rmm::cuda_stream_view stream)
{
  if (dictionaries.count() == 0) { return; }
  constexpr int block_size = 256;
  dim3 const dim_grid(dictionaries.size().first, dictionaries.size().second);
  populate_dictionary_hash_maps_kernel<block_size>
    <<<dim_grid, block_size, 0, stream.value()>>>(dictionaries, columns);
}

void collect_map_entries(device_2dspan<stripe_dictionary> dictionaries,
                         rmm::cuda_stream_view stream)
{
  if (dictionaries.count() == 0) { return; }
  constexpr int block_size = 1024;
  dim3 const dim_grid(dictionaries.size().first, dictionaries.size().second);
  collect_map_entries_kernel<block_size><<<dim_grid, block_size, 0, stream.value()>>>(dictionaries);
}

void get_dictionary_indices(device_2dspan<stripe_dictionary> dictionaries,
                            device_span<orc_column_device_view const> columns,
                            rmm::cuda_stream_view stream)
{
  if (dictionaries.count() == 0) { return; }
  constexpr int block_size = 1024;
  dim3 const dim_grid(dictionaries.size().first, dictionaries.size().second);
  get_dictionary_indices_kernel<block_size>
    <<<dim_grid, block_size, 0, stream.value()>>>(dictionaries, columns);
}

}  // namespace cudf::io::orc::gpu
