/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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

#pragma once

#include "parquet_gpu.hpp"

#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/types.hpp>

#include <cooperative_groups.h>
#include <cuco/static_map.cuh>
#include <cuco/static_map_ref.cuh>
#include <cuco/storage.cuh>
#include <thrust/device_vector.h>

#include <limits>

namespace cudf::io::parquet::detail {

using key_type    = uint32_t;
using mapped_type = uint32_t;

auto constexpr cg_size     = 1;  ///< A CUDA Cooperative Group of 8 threads to handle each subset
auto constexpr window_size = 1;  ///< Number of concurrent slots handled by each thread

auto constexpr KEY_SENTINEL   = key_type{std::numeric_limits<uint32_t>::max()};
auto constexpr VALUE_SENTINEL = mapped_type{std::numeric_limits<uint32_t>::max()};
auto constexpr SCOPE          = cuda::thread_scope_device;

using slot_type = cuco::pair<key_type, mapped_type>;

using storage_type     = cuco::aow_storage<slot_type, window_size>;
using storage_ref_type = typename storage_type::ref_type;

template <typename T>
struct my_hasher {
  __device__ auto operator()(T index) const { return index; }
};

using probing_scheme_type = cuco::linear_probing<cg_size, my_hasher<key_type>>;

using map_ref_type = cuco::static_map_ref<key_type,
                                          mapped_type,
                                          SCOPE,
                                          thrust::equal_to<key_type>,
                                          probing_scheme_type,
                                          storage_ref_type>;  ///< Map ref type

/**
 * @brief Insert chunk values into their respective hash maps
 *
 * @param frags Column fragments
 * @param stream CUDA stream to use
 */
void populate_chunk_hash_maps(map_ref_type* map_refs,
                              cudf::detail::device_2dspan<PageFragment const> frags,
                              rmm::cuda_stream_view stream);

/**
 * @brief Compact dictionary hash map entries into chunk.dict_data
 *
 * @param chunks Flat span of chunks to compact hash maps for
 * @param stream CUDA stream to use
 */
void collect_map_entries(map_ref_type* map_refs,
                         device_span<EncColumnChunk> chunks,
                         rmm::cuda_stream_view stream);

/**
 * @brief Get the Dictionary Indices for each row
 *
 * For each row of a chunk, gets the indices into chunk.dict_data which contains the value otherwise
 * stored in input column [row]. Stores these indices into chunk.dict_index.
 *
 * Since dict_data itself contains indices into the original cudf column, this means that
 * col[row] == col[dict_data[dict_index[row - chunk.start_row]]]
 *
 * @param frags Column fragments
 * @param stream CUDA stream to use
 */
void get_dictionary_indices(map_ref_type* map_refs,
                            cudf::detail::device_2dspan<PageFragment const> frags,
                            rmm::cuda_stream_view stream);

/**
 * @brief Return the byte length of parquet dtypes that are physically represented by INT32
 */
inline uint32_t __device__ int32_logical_len(type_id id)
{
  switch (id) {
    case cudf::type_id::INT8: [[fallthrough]];
    case cudf::type_id::UINT8: return 1;
    case cudf::type_id::INT16: [[fallthrough]];
    case cudf::type_id::UINT16: return 2;
    case cudf::type_id::DURATION_SECONDS: [[fallthrough]];
    case cudf::type_id::DURATION_MILLISECONDS: return 8;
    default: return 4;
  }
}

/**
 * @brief Translate the row index of a parent column_device_view into the index of the first value
 * in the leaf child.
 * Only works in the context of parquet writer where struct columns are previously modified s.t.
 * they only have one immediate child.
 */
inline size_type __device__ row_to_value_idx(size_type idx,
                                             parquet_column_device_view const& parquet_col)
{
  // with a byte array, we can't go all the way down to the leaf node, but instead we want to leave
  // the size at the parent level because we are writing out parent row byte arrays.
  auto col = *parquet_col.parent_column;
  while (col.type().id() == type_id::LIST or col.type().id() == type_id::STRUCT) {
    if (col.type().id() == type_id::STRUCT) {
      idx += col.offset();
      col = col.child(0);
    } else {
      auto list_col = cudf::detail::lists_column_device_view(col);
      auto child    = list_col.child();
      if (parquet_col.output_as_byte_array && child.type().id() == type_id::UINT8) { break; }
      idx = list_col.offset_at(idx);
      col = child;
    }
  }
  return idx;
}

}  // namespace cudf::io::parquet::detail
