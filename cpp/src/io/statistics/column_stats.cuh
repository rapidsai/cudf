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

#pragma once

#include "temp_storage_wrapper.cuh"

#include "typed_statistics_chunk.cuh"

#include "statistics.cuh"

namespace cudf {
namespace io {

/**
 * @brief shared state for statistics gather kernel
 */
struct stats_state_s {
  stats_column_desc col;                 ///< Column information
  statistics_group group;                ///< Group description
  statistics_chunk ck;                   ///< Output statistics chunk
};

/**
 * @brief shared state for statistics merge kernel
 */
struct merge_state_s {
  stats_column_desc col;                 ///< Column information
  statistics_merge_group group;          ///< Group description
  statistics_chunk ck;                   ///< Resulting statistics chunk
};

template <int dimension>
using block_reduce_storage = detail::block_reduce_storage<dimension>;

template <int block_size, detail::io_type IO>
struct gather_statistics {

  block_reduce_storage<block_size> &temp_storage;

  __device__
  gather_statistics(block_reduce_storage<block_size> &d_temp_storage) : temp_storage(d_temp_storage) {}

  template <typename T, std::enable_if_t<detail::statistics_type_category<T, IO>::ignored_statistics>* = nullptr>
  __device__ void operator()(stats_state_s& s, uint32_t t){
    //No-op for unsupported aggregation types
  }

  template <typename T, std::enable_if_t<not detail::statistics_type_category<T, IO>::ignored_statistics>* = nullptr>
  __device__ void operator()(stats_state_s& s, uint32_t t){
    using E = typename detail::extrema_type<T>::type;
    using extrema_reduce = cub::BlockReduce<E, block_size>;
    detail::temp_storage_wrapper<E, block_size> extrema_storage(temp_storage);

    typed_statistics_chunk<T> chunk(s.group.num_rows);

    for (uint32_t i = 0; i < s.group.num_rows; i += block_size) {
      uint32_t r        = i + t;
      uint32_t row      = r + s.group.start_row;
      uint32_t is_valid = (r < s.group.num_rows) ? s.col.leaf_column->is_valid(row) : 0;
      if (is_valid) {
        chunk.reduce(s.col.leaf_column->element<T>(row));
      }
    }
    chunk.minimum_value =
      extrema_reduce(extrema_storage.get()).Reduce(chunk.minimum_value, cub::Min());
    __syncthreads();
    chunk.maximum_value =
      extrema_reduce(extrema_storage.get()).Reduce(chunk.maximum_value, cub::Max());
    __syncthreads();
    chunk.has_minmax = __syncthreads_or(chunk.has_minmax);

    if constexpr (detail::aggregation_type<T>::is_supported) {
      if (chunk.has_minmax) {
        using A = typename detail::aggregation_type<T>::type;
        using aggregate_reduce = cub::BlockReduce<A, block_size>;
        detail::temp_storage_wrapper<A, block_size> aggregate_storage(temp_storage);
        chunk.aggregate = aggregate_reduce(aggregate_storage.get()).Sum(chunk.aggregate);
      }
    }

    if (threadIdx.x == 0) {
      s.ck = get_untyped_chunk(chunk);
      if (s.col.stats_dtype == dtype_timestamp64) {
        if (s.col.ts_scale < -1) {
          s.ck.min_value.i_val /= -s.col.ts_scale;
          s.ck.max_value.i_val /= -s.col.ts_scale;
        } else if (s.col.ts_scale > 1) {
          s.ck.min_value.i_val *= s.col.ts_scale;
          s.ck.max_value.i_val *= s.col.ts_scale;
        }
      }
    }
  }
};

template <int block_size, detail::io_type IO>
struct merge_statistics {

  block_reduce_storage<block_size> &temp_storage;

  __device__
  merge_statistics(block_reduce_storage<block_size> &d_temp_storage) : temp_storage(d_temp_storage) {}

  template <typename T, std::enable_if_t<detail::statistics_type_category<T, IO>::ignored_statistics>* = nullptr>
  __device__ void operator()(merge_state_s& s, const statistics_chunk *chunks, const uint32_t num_chunks, uint32_t t){
    //No-op for unsupported aggregation types
  }

  template <typename T, std::enable_if_t<not detail::statistics_type_category<T, IO>::ignored_statistics>* = nullptr>
  __device__ void operator()(merge_state_s& s, const statistics_chunk *chunks, const uint32_t num_chunks, uint32_t t){
    using E = typename detail::extrema_type<T>::type;
    using extrema_reduce = cub::BlockReduce<E, block_size>;
    using count_reduce = cub::BlockReduce<uint32_t, block_size>;
    //detail::temp_storage_wrapper<E, block_size> extrema_storage(temp_storage);
    //detail::temp_storage_wrapper<uint32_t, block_size> count_storage(temp_storage);

    detail::storage_wrapper<block_size> storage(temp_storage);

    typed_statistics_chunk<T> chunk;

    for (uint32_t i = t; i < num_chunks; i += block_size) {
      chunk.reduce(chunks[i]);
    }
    chunk.has_minmax = (chunk.minimum_value <= chunk.maximum_value);

    chunk.minimum_value =
      extrema_reduce(storage.template get<E>()).Reduce(chunk.minimum_value, cub::Min());
    __syncthreads();
    chunk.maximum_value =
      extrema_reduce(storage.template get<E>()).Reduce(chunk.maximum_value, cub::Max());
    __syncthreads();
    chunk.non_nulls =
      count_reduce(storage.template get<uint32_t>()).Sum(chunk.non_nulls);
    __syncthreads();
    chunk.null_count =
      count_reduce(storage.template get<uint32_t>()).Sum(chunk.null_count);
    __syncthreads();
    chunk.has_minmax = __syncthreads_or(chunk.has_minmax);

    if constexpr (detail::aggregation_type<T>::is_supported) {
      if (chunk.has_minmax) {
        using A = typename detail::aggregation_type<T>::type;
        using aggregate_reduce = cub::BlockReduce<A, block_size>;
        //detail::temp_storage_wrapper<A, block_size> aggregate_storage(temp_storage);
        chunk.aggregate = aggregate_reduce(storage.template get<A>()).Sum(chunk.aggregate);
      }
    }

    if (threadIdx.x == 0) {
      s.ck = get_untyped_chunk(chunk);
    }
  }

};

template <typename T>
__device__
void cooperative_load(T& destination) {
  using load_type = std::conditional_t<((sizeof(T) % sizeof(uint32_t)) == 0),
        uint32_t, uint8_t>;
  for (auto i = threadIdx.x; i < sizeof(T) / sizeof(load_type); i += blockDim.x) {
    reinterpret_cast<load_type *>(&destination)[i] = load_type{0};
  }
}

template <typename T>
__device__
void cooperative_load(T& destination, const T& source) {
  using load_type = std::conditional_t<((sizeof(T) % sizeof(uint32_t)) == 0),
        uint32_t, uint8_t>;
  for (auto i = threadIdx.x; i < (sizeof(T) / sizeof(load_type)); i += blockDim.x) {
    reinterpret_cast<load_type *>(&destination)[i] =
      reinterpret_cast<const load_type *>(&source)[i];
  }
}

template <typename T>
__device__
void cooperative_load(T& destination, const T* source) {
  using load_type = std::conditional_t<((sizeof(T) % sizeof(uint32_t)) == 0),
        uint32_t, uint8_t>;
  for (auto i = threadIdx.x; i < (sizeof(T) / sizeof(load_type)); i += blockDim.x) {
    reinterpret_cast<load_type *>(&destination)[i] =
      reinterpret_cast<const load_type *>(source)[i];
  }
}

/**
 * @brief Launches kernel to gather column statistics
 *
 * @param[out] chunks Statistics results [num_chunks]
 * @param[in] groups Statistics row groups [num_chunks]
 * @param[in] num_chunks Number of chunks & rowgroups
 * @param[in] stream CUDA stream to use, default 0
 */
template <int block_size, detail::io_type IO>
__global__ void __launch_bounds__(block_size, 1)
  gpuGatherColumnStatistics(statistics_chunk *chunks, const statistics_group *groups)
{
  __shared__ __align__(8) stats_state_s state;
  __shared__ block_reduce_storage<block_size> storage;

  //Load state members
  cooperative_load(state.group, groups[blockIdx.x]);
  cooperative_load(state.ck);
  __syncthreads();
  cooperative_load(state.col, state.group.col);
  __syncthreads();

  //Calculate statistics
  type_dispatcher(state.col.leaf_column->type(), gather_statistics<block_size, IO>(storage), state, threadIdx.x);
  __syncthreads();

  cooperative_load(chunks[blockIdx.x], state.ck);
}
void GatherColumnStatistics(statistics_chunk *chunks,
                            const statistics_group *groups,
                            uint32_t num_chunks,
                            rmm::cuda_stream_view stream);
void MergeColumnStatistics(statistics_chunk *chunks_out,
                           const statistics_chunk *chunks_in,
                           const statistics_merge_group *groups,
                           uint32_t num_chunks,
                           rmm::cuda_stream_view stream);

namespace detail {

template <detail::io_type IO>
void GatherColumnStatistics(statistics_chunk *chunks,
                            const statistics_group *groups,
                            uint32_t num_chunks,
                            rmm::cuda_stream_view stream)
{
  constexpr int block_size = 256;
  gpuGatherColumnStatistics<block_size, IO>
    <<<num_chunks, block_size, 0, stream.value()>>>(chunks, groups);
}

template <int block_size, detail::io_type IO>
__global__ void __launch_bounds__(block_size, 1)
  gpuMergeColumnStatistics(statistics_chunk *chunks_out,
                           const statistics_chunk *chunks_in,
                           const statistics_merge_group *groups)
{
  __shared__ __align__(8) merge_state_s state;
  __shared__ block_reduce_storage<block_size> storage;

  cooperative_load(state.group, groups[blockIdx.x]);
  __syncthreads();
  cooperative_load(state.col, state.group.col);
  __syncthreads();

  type_dispatcher(
      state.col.leaf_column->type(),
      merge_statistics<block_size, IO>(storage),
      state, chunks_in + state.group.start_chunk,
      state.group.num_chunks,
      threadIdx.x);
  __syncthreads();

  cooperative_load(chunks_out[blockIdx.x], state.ck);
}

/**
 * @brief Launches kernel to merge column statistics
 *
 * @param[out] chunks_out Statistics results [num_chunks]
 * @param[out] chunks_in Input statistics
 * @param[in] groups Statistics groups [num_chunks]
 * @param[in] num_chunks Number of chunks & groups
 * @param[in] stream CUDA stream to use, default 0
 */
template <detail::io_type IO>
void MergeColumnStatistics(statistics_chunk *chunks_out,
                           const statistics_chunk *chunks_in,
                           const statistics_merge_group *groups,
                           uint32_t num_chunks,
                           rmm::cuda_stream_view stream)
{
  constexpr int block_size = 256;
  gpuMergeColumnStatistics<block_size, IO>
    <<<num_chunks, block_size, 0, stream.value()>>>(chunks_out, chunks_in, groups);
}

}  // namespace detail
}  // namespace io
}  // namespace cudf
