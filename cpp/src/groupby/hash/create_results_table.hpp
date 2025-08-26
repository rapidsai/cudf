/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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

#include "helpers.cuh"

#include <cudf/aggregation.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cuco/static_map.cuh>
#include <cuco/static_set.cuh>

namespace cudf::groupby::detail::hash {

// struct key_indices_hasher_t {
//   size_type const* key_indices{nullptr};
//   using hasher = cuco::default_hash_function<size_type>;

//   __device__ bool operator()(size_type idx) const { return hasher{}(key_indices[idx]); }
// };
// struct simplified_probing_scheme_t : cuco::linear_probing<GROUPBY_CG_SIZE, key_indices_hasher_t>
// {
//   __device__ simplified_probing_scheme_t(key_indices_hasher_t const& hasher)
//     : cuco::linear_probing<GROUPBY_CG_SIZE, key_indices_hasher_t>{hasher}
//   {
//   }

//   simplified_probing_scheme_t(size_type const* key_indices)
//     : cuco::linear_probing<GROUPBY_CG_SIZE,
//     key_indices_hasher_t>{key_indices_hasher_t{key_indices}}
//   {
//   }
// };

// struct key_indices_comparator_t {
//   size_type const* key_indices{nullptr};

//   __device__ bool operator()(size_type const lhs, size_type const rhs) const
//   {
//     return key_indices[lhs] == key_indices[rhs];
//   }
// };

// using simplified_row_comparator_t = key_indices_comparator_t;

// using simplified_global_set_t = cuco::static_set<cudf::size_type,
//                                                  cuco::extent<int64_t>,
//                                                  cuda::thread_scope_device,
//                                                  simplified_row_comparator_t,
//                                                  simplified_probing_scheme_t,
//                                                  cudf::detail::cuco_allocator<char>,
//                                                  cuco::storage<GROUPBY_BUCKET_SIZE>>;

#if 0
struct eq_t {
  int* count;
  eq_t(int* count) : count(count) {}

  __device__ bool operator()(int x, int y) const
  {
    auto c = atomicAdd(count, 1);
    printf("compare %d, %d, count = %d\n", x, y, c + 1);

    return x == y;
  }
};

struct hash_t {
  int* count;
  hash_t(int* count) : count(count) {}
  __device__ uint32_t operator()(int x) const
  {
    auto c  = atomicAdd(count, 1);
    using h = cuco::default_hash_function<cudf::size_type>;
    auto t  = h{}(x);
    printf("hash %d = %d, count = %d\n", x, (int)t, c + 1);
    return t;
  }
};
#endif

using key_map_t = cuco::static_map<
  cudf::size_type,
  cudf::size_type,
  cuco::extent<std::size_t>,
  cuda::thread_scope_device,
  cuda::std::equal_to<cudf::size_type>,
  cuco::linear_probing<GROUPBY_CG_SIZE, cuco::default_hash_function<cudf::size_type>>,
  // eq_t,
  // cuco::linear_probing<GROUPBY_CG_SIZE, hash_t>,
  cudf::detail::cuco_allocator<char>,
  cuco::storage<GROUPBY_BUCKET_SIZE>>;

// TODO
std::pair<key_map_t, rmm::device_uvector<size_type>> find_output_indices(
  device_span<size_type> key_indices,
  device_span<size_type const> unique_indices,
  rmm::cuda_stream_view stream);

/**
 * @brief Computes and returns a device vector containing all populated keys in
 * `key_set`.
 *
 * @tparam SetType Type of the key hash set
 *
 * @param key_set Key hash set
 * @param populated_keys Array of unique keys
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return An array of unique keys contained in `key_set`
 */
template <typename SetType>
void extract_populated_keys(SetType const& key_set,
                            rmm::device_uvector<size_type>& populated_keys,
                            rmm::cuda_stream_view stream);

table create_results_table(cudf::size_type output_size,
                           table_view const& flattened_values,
                           host_span<aggregation::Kind const> agg_kinds,
                           rmm::cuda_stream_view stream);
}  // namespace cudf::groupby::detail::hash
