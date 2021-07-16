/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

/**
 * @file column_statistics.cuh
 * @brief Functors for statistics calculation to be used in ORC and PARQUET
 */

#pragma once

#include "temp_storage_wrapper.cuh"

#include "typed_statistics_chunk.cuh"

#include "statistics.cuh"

namespace cudf {
namespace io {

/**
 * @brief shared state for statistics calculation kernel
 */
struct stats_state_s {
  stats_column_desc col;   ///< Column information
  statistics_group group;  ///< Group description
  statistics_chunk ck;     ///< Output statistics chunk
};

/**
 * @brief shared state for statistics merge kernel
 */
struct merge_state_s {
  stats_column_desc col;         ///< Column information
  statistics_merge_group group;  ///< Group description
  statistics_chunk ck;           ///< Resulting statistics chunk
};

template <int dimension>
using block_reduce_storage = detail::block_reduce_storage<dimension>;

/**
 * @brief Functor to calculate the statistics of rows in a column belonging to a
 * statistics group
 *
 * @tparam block_size Dimension of the block
 * @tparam IO File format for which statistics calculation is being done
 */
template <int block_size, detail::io_file_format IO>
struct calculate_group_statistics_functor {
  block_reduce_storage<block_size>& temp_storage;

  /**
   * @brief Construct a statistics calculator
   *
   * @param d_temp_storage Temporary storage to be used by cub calls
   */
  __device__ calculate_group_statistics_functor(block_reduce_storage<block_size>& d_temp_storage)
    : temp_storage(d_temp_storage)
  {
  }

  template <typename T,
            std::enable_if_t<detail::statistics_type_category<T, IO>::is_ignored>* = nullptr>
  __device__ void operator()(stats_state_s& s, uint32_t t)
  {
    // No-op for unsupported aggregation types
  }

  /**
   * @brief Iterates through the rows specified by statistics group and stores the combined
   * statistics into the statistics chunk.
   *
   * @param s Statistics state which specifies the column, the group being worked and the chunk
   * the results will be stored into
   * @param t thread id
   */
  template <typename T,
            std::enable_if_t<not detail::statistics_type_category<T, IO>::is_ignored>* = nullptr>
  __device__ void operator()(stats_state_s& s, uint32_t t)
  {
    detail::storage_wrapper<block_size> storage(temp_storage);

    using type_convert = detail::type_conversion<detail::conversion_map<IO>>;
    using CT           = typename type_convert::template type<T>;
    typed_statistics_chunk<CT, detail::statistics_type_category<T, IO>::is_aggregated> chunk(
      s.group.num_rows);

    for (uint32_t i = 0; i < s.group.num_rows; i += block_size) {
      uint32_t r          = i + t;
      uint32_t row        = r + s.group.start_row;
      auto const is_valid = (r < s.group.num_rows) ? s.col.leaf_column->is_valid(row) : 0;
      if (is_valid) {
        auto converted_value = type_convert::convert(s.col.leaf_column->element<T>(row));
        chunk.reduce(converted_value);
      }
    }

    chunk = block_reduce(chunk, storage);

    if (t == 0) { s.ck = get_untyped_chunk(chunk); }
  }
};

/**
 * @brief Functor to merge the statistics chunks of a column belonging to a
 * merge group
 *
 * @tparam block_size Dimension of the block
 * @tparam IO File format for which statistics calculation is being done
 */
template <int block_size, detail::io_file_format IO>
struct merge_group_statistics_functor {
  block_reduce_storage<block_size>& temp_storage;

  __device__ merge_group_statistics_functor(block_reduce_storage<block_size>& d_temp_storage)
    : temp_storage(d_temp_storage)
  {
  }

  template <typename T,
            std::enable_if_t<detail::statistics_type_category<T, IO>::is_ignored>* = nullptr>
  __device__ void operator()(merge_state_s& s,
                             const statistics_chunk* chunks,
                             const uint32_t num_chunks,
                             uint32_t t)
  {
    // No-op for unsupported aggregation types
  }

  template <typename T,
            std::enable_if_t<not detail::statistics_type_category<T, IO>::is_ignored>* = nullptr>
  __device__ void operator()(merge_state_s& s,
                             const statistics_chunk* chunks,
                             const uint32_t num_chunks,
                             uint32_t t)
  {
    detail::storage_wrapper<block_size> storage(temp_storage);

    typed_statistics_chunk<T, detail::statistics_type_category<T, IO>::is_aggregated> chunk;

    for (uint32_t i = t; i < num_chunks; i += block_size) {
      chunk.reduce(chunks[i]);
    }
    chunk.has_minmax = (chunk.minimum_value <= chunk.maximum_value);

    chunk = block_reduce(chunk, storage);

    if (t == 0) { s.ck = get_untyped_chunk(chunk); }
  }
};

/**
 * @brief Function to cooperatively load an object from a pointer
 *
 * If the pointer is nullptr then the members of the object are set to 0
 *
 * @param[out] destination Object being loaded
 * @param[in] source Source object
 * @tparam T Type of object
 */
template <typename T>
__device__ void cooperative_load(T& destination, const T* source = nullptr)
{
  using load_type = std::conditional_t<((sizeof(T) % sizeof(uint32_t)) == 0), uint32_t, uint8_t>;
  if (source == nullptr) {
    for (auto i = threadIdx.x; i < (sizeof(T) / sizeof(load_type)); i += blockDim.x) {
      reinterpret_cast<load_type*>(&destination)[i] = load_type{0};
    }
  } else {
    for (auto i = threadIdx.x; i < sizeof(T) / sizeof(load_type); i += blockDim.x) {
      reinterpret_cast<load_type*>(&destination)[i] = reinterpret_cast<const load_type*>(source)[i];
    }
  }
}

/**
 * @brief Kernel to calculate group statistics
 *
 * @param[out] chunks Statistics results [num_chunks]
 * @param[in] groups Statistics row groups [num_chunks]
 * @tparam block_size Dimension of the block
 * @tparam IO File format for which statistics calculation is being done
 */
template <int block_size, detail::io_file_format IO>
__global__ void __launch_bounds__(block_size, 1)
  gpu_calculate_group_statistics(statistics_chunk* chunks, const statistics_group* groups)
{
  __shared__ __align__(8) stats_state_s state;
  __shared__ block_reduce_storage<block_size> storage;

  // Load state members
  cooperative_load(state.group, &groups[blockIdx.x]);
  cooperative_load(state.ck);
  __syncthreads();
  cooperative_load(state.col, state.group.col);
  __syncthreads();

  // Calculate statistics
  type_dispatcher(state.col.leaf_column->type(),
                  calculate_group_statistics_functor<block_size, IO>(storage),
                  state,
                  threadIdx.x);
  __syncthreads();

  cooperative_load(chunks[blockIdx.x], &state.ck);
}

namespace detail {

/**
 * @brief Launches kernel to calculate group statistics
 *
 * @param[out] chunks Statistics results [num_chunks]
 * @param[in] groups Statistics row groups [num_chunks]
 * @param[in] num_chunks Number of chunks & rowgroups
 * @param[in] stream CUDA stream to use
 * @tparam IO File format for which statistics calculation is being done
 */
template <detail::io_file_format IO>
void calculate_group_statistics(statistics_chunk* chunks,
                                const statistics_group* groups,
                                uint32_t num_chunks,
                                rmm::cuda_stream_view stream)
{
  constexpr int block_size = 256;
  gpu_calculate_group_statistics<block_size, IO>
    <<<num_chunks, block_size, 0, stream.value()>>>(chunks, groups);
}

/**
 * @brief Kernel to merge column statistics
 *
 * @param[out] chunks_out Statistics results [num_chunks]
 * @param[in] chunks_in Input statistics
 * @param[in] groups Statistics groups [num_chunks]
 * @tparam block_size Dimension of the block
 * @tparam IO File format for which statistics calculation is being done
 */
template <int block_size, detail::io_file_format IO>
__global__ void __launch_bounds__(block_size, 1)
  gpu_merge_group_statistics(statistics_chunk* chunks_out,
                             const statistics_chunk* chunks_in,
                             const statistics_merge_group* groups)
{
  __shared__ __align__(8) merge_state_s state;
  __shared__ block_reduce_storage<block_size> storage;

  cooperative_load(state.group, &groups[blockIdx.x]);
  __syncthreads();
  cooperative_load(state.col, state.group.col);
  __syncthreads();

  type_dispatcher(state.col.leaf_column->type(),
                  merge_group_statistics_functor<block_size, IO>(storage),
                  state,
                  chunks_in + state.group.start_chunk,
                  state.group.num_chunks,
                  threadIdx.x);
  __syncthreads();

  cooperative_load(chunks_out[blockIdx.x], &state.ck);
}

/**
 * @brief Launches kernel to merge column statistics
 *
 * @param[out] chunks_out Statistics results [num_chunks]
 * @param[in] chunks_in Input statistics
 * @param[in] groups Statistics groups [num_chunks]
 * @param[in] num_chunks Number of chunks & groups
 * @param[in] stream CUDA stream to use
 * @tparam IO File format for which statistics calculation is being done
 */
template <detail::io_file_format IO>
void merge_group_statistics(statistics_chunk* chunks_out,
                            const statistics_chunk* chunks_in,
                            const statistics_merge_group* groups,
                            uint32_t num_chunks,
                            rmm::cuda_stream_view stream)
{
  constexpr int block_size = 256;
  gpu_merge_group_statistics<block_size, IO>
    <<<num_chunks, block_size, 0, stream.value()>>>(chunks_out, chunks_in, groups);
}

}  // namespace detail
}  // namespace io
}  // namespace cudf
