/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/primitive_row_operators.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuco/bucket_storage.cuh>
#include <cuco/extent.cuh>

// Forward declaration
namespace cudf::experimental::row::equality {
class preprocessed_table;
}

namespace CUDF_EXPORT cudf {
namespace detail {

struct filtered_join {
 public:
  using key = size_type;

  using storage_type =
    cuco::bucket_storage<key,
                         1,  /// fixing bucket size to be 1 i.e each thread handles one slot
                         cuco::extent<cudf::size_type>,
                         cudf::detail::cuco_allocator<char>>;
  using storage_type_ref = typename storage_type::ref_type;

  using primitive_row_hasher =
    cudf::row::primitive::row_hasher<cudf::hashing::detail::default_hash>;
  using primitive_probing_scheme = cuco::linear_probing<1, primitive_row_hasher>;
  using primitive_row_comparator = cudf::row::primitive::row_equality_comparator;

  using row_hasher            = cudf::experimental::row::hash::device_row_hasher<cudf::hashing::detail::default_hash, nullate::DYNAMIC>;
  using nested_probing_scheme = cuco::linear_probing<4, row_hasher>;
  using simple_probing_scheme = cuco::linear_probing<1, row_hasher>;
  using row_comparator = cudf::experimental::row::equality::device_row_comparator<
    true,
    cudf::nullate::DYNAMIC,
    cudf::experimental::row::equality::nan_equal_physical_equality_comparator>;

  filtered_join()                            = delete;
  ~filtered_join()                           = default;
  filtered_join(filtered_join const&)            = delete;
  filtered_join(filtered_join&&)                 = delete;
  filtered_join& operator=(filtered_join const&) = delete;
  filtered_join& operator=(filtered_join&&)      = delete;

 private:
  bool _has_nested_columns;  ///< True if nested columns are present in build and probe tables
  cudf::null_equality const _nulls_equal;  ///< whether to consider nulls as equal
  cudf::table_view _build;                 ///< input table to build the hash map
  std::shared_ptr<cudf::experimental::row::equality::preprocessed_table>
    _preprocessed_build;  ///< input table preprocssed for row operators
  storage_type _bucket_storage;

 public:
  filtered_join(cudf::table_view const& build,
            cudf::null_equality compare_nulls,
            rmm::cuda_stream_view stream);

  /**
   * @copydoc filtered_join(cudf::table_view const&, bool, null_equality, rmm::cuda_stream_view)
   *
   * @param load_factor The hash table occupancy ratio in (0,1]. A value of 0.5 means 50% occupancy.
   */
  filtered_join(cudf::table_view const& build,
            cudf::null_equality compare_nulls,
            double load_factor,
            rmm::cuda_stream_view stream);

  std::unique_ptr<rmm::device_uvector<cudf::size_type>> semi_join(cudf::table_view const& probe, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> anti_join(cudf::table_view const& probe, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);
};
}  // namespace detail
}  // namespace CUDF_EXPORT cudf
