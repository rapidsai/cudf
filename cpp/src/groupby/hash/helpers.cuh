/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/types.hpp>

#include <cuco/static_set.cuh>

namespace cudf::groupby::detail::hash {
// TODO: similar to `contains_table`, using larger CG size like 2 or 4 for nested
// types and `cg_size = 1`for flat data to improve performance
/// Number of threads to handle each input element
CUDF_HOST_DEVICE auto constexpr GROUPBY_CG_SIZE = 1;

/// Number of slots per thread
CUDF_HOST_DEVICE auto constexpr GROUPBY_WINDOW_SIZE = 1;

/// Thread block size
CUDF_HOST_DEVICE auto constexpr GROUPBY_BLOCK_SIZE = 128;

/// Threshold cardinality to switch between shared memory aggregations and global memory
/// aggregations
CUDF_HOST_DEVICE auto constexpr GROUPBY_CARDINALITY_THRESHOLD = 128;

// We add additional `block_size`, because after the number of elements in the local hash set
// exceeds the threshold, all threads in the thread block can still insert one more element.
/// The maximum number of elements handled per block
CUDF_HOST_DEVICE auto constexpr GROUPBY_SHM_MAX_ELEMENTS =
  GROUPBY_CARDINALITY_THRESHOLD + GROUPBY_BLOCK_SIZE;

// GROUPBY_SHM_MAX_ELEMENTS with 0.7 occupancy
/// Shared memory hash set extent type
using shmem_extent_t =
  cuco::extent<cudf::size_type,
               static_cast<cudf::size_type>(static_cast<double>(GROUPBY_SHM_MAX_ELEMENTS) * 1.43)>;

/// Number of windows needed by each shared memory hash set
CUDF_HOST_DEVICE auto constexpr window_extent =
  cuco::make_window_extent<GROUPBY_CG_SIZE, GROUPBY_WINDOW_SIZE>(shmem_extent_t{});

/**
 * @brief Returns the smallest multiple of 8 that is greater than or equal to the given integer.
 */
CUDF_HOST_DEVICE constexpr std::size_t round_to_multiple_of_8(std::size_t num)
{
  std::size_t constexpr base = 8;
  return cudf::util::div_rounding_up_safe(num, base) * base;
}

using row_hash_t =
  cudf::experimental::row::hash::device_row_hasher<cudf::hashing::detail::default_hash,
                                                   cudf::nullate::DYNAMIC>;

/// Probing scheme type used by groupby hash table
using probing_scheme_t = cuco::linear_probing<GROUPBY_CG_SIZE, row_hash_t>;

using row_comparator_t = cudf::experimental::row::equality::device_row_comparator<
  false,
  cudf::nullate::DYNAMIC,
  cudf::experimental::row::equality::nan_equal_physical_equality_comparator>;

using nullable_row_comparator_t = cudf::experimental::row::equality::device_row_comparator<
  true,
  cudf::nullate::DYNAMIC,
  cudf::experimental::row::equality::nan_equal_physical_equality_comparator>;

using global_set_t = cuco::static_set<cudf::size_type,
                                      cuco::extent<int64_t>,
                                      cuda::thread_scope_device,
                                      row_comparator_t,
                                      probing_scheme_t,
                                      cudf::detail::cuco_allocator<char>,
                                      cuco::storage<GROUPBY_WINDOW_SIZE>>;

using nullable_global_set_t = cuco::static_set<cudf::size_type,
                                               cuco::extent<int64_t>,
                                               cuda::thread_scope_device,
                                               nullable_row_comparator_t,
                                               probing_scheme_t,
                                               cudf::detail::cuco_allocator<char>,
                                               cuco::storage<GROUPBY_WINDOW_SIZE>>;

template <typename Op>
using hash_set_ref_t = cuco::static_set_ref<
  cudf::size_type,
  cuda::thread_scope_device,
  row_comparator_t,
  probing_scheme_t,
  cuco::aow_storage_ref<cudf::size_type, GROUPBY_WINDOW_SIZE, cuco::window_extent<int64_t>>,
  Op>;

template <typename Op>
using nullable_hash_set_ref_t = cuco::static_set_ref<
  cudf::size_type,
  cuda::thread_scope_device,
  nullable_row_comparator_t,
  probing_scheme_t,
  cuco::aow_storage_ref<cudf::size_type, GROUPBY_WINDOW_SIZE, cuco::window_extent<int64_t>>,
  Op>;
}  // namespace cudf::groupby::detail::hash
