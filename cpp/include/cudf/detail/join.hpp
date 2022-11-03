/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/detail/utilities/hash_functions.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/device/polymorphic_allocator.hpp>

#include <cuco/static_multimap.cuh>

#include <cstddef>
#include <memory>
#include <optional>

// Forward declaration
template <typename T>
class default_allocator;

namespace cudf {
namespace detail {

constexpr int DEFAULT_JOIN_CG_SIZE = 2;

enum class join_kind { INNER_JOIN, LEFT_JOIN, FULL_JOIN, LEFT_SEMI_JOIN, LEFT_ANTI_JOIN };

/**
 * @brief Hash join that builds hash table in creation and probes results in subsequent `*_join`
 * member functions.
 *
 * User-defined hash function can be passed via the template parameter `Hasher`
 *
 * @tparam Hasher Unary callable type
 */
template <typename Hasher>
struct hash_join {
 public:
  using map_type =
    cuco::static_multimap<hash_value_type,
                          cudf::size_type,
                          cuda::thread_scope_device,
                          rmm::mr::stream_allocator_adaptor<default_allocator<char>>,
                          cuco::double_hashing<DEFAULT_JOIN_CG_SIZE, Hasher, Hasher>>;

  hash_join()                 = delete;
  ~hash_join()                = default;
  hash_join(hash_join const&) = delete;
  hash_join(hash_join&&)      = delete;
  hash_join& operator=(hash_join const&) = delete;
  hash_join& operator=(hash_join&&) = delete;

 private:
  bool const _is_empty;                         ///< true if `_hash_table` is empty
  rmm::device_buffer const _composite_bitmask;  ///< Bitmask to denote whether a row is valid
  cudf::null_equality const _nulls_equal;       ///< whether to consider nulls as equal
  cudf::table_view _build;                      ///< input table to build the hash map
  cudf::structs::detail::flattened_table
    _flattened_build_table;  ///< flattened data structures for `_build`
  map_type _hash_table;      ///< hash table built on `_build`

 public:
  /**
   * @brief Constructor that internally builds the hash table based on the given `build` table.
   *
   * @throw cudf::logic_error if the number of columns in `build` table is 0.
   * @throw cudf::logic_error if the number of rows in `build` table exceeds MAX_JOIN_SIZE.
   *
   * @param build The build table, from which the hash table is built.
   * @param compare_nulls Controls whether null join-key values should match or not.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   */
  hash_join(cudf::table_view const& build,
            cudf::null_equality compare_nulls,
            rmm::cuda_stream_view stream);

  /**
   * @copydoc cudf::hash_join::inner_join
   */
  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  inner_join(cudf::table_view const& probe,
             std::optional<std::size_t> output_size,
             rmm::cuda_stream_view stream,
             rmm::mr::device_memory_resource* mr) const;

  /**
   * @copydoc cudf::hash_join::left_join
   */
  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  left_join(cudf::table_view const& probe,
            std::optional<std::size_t> output_size,
            rmm::cuda_stream_view stream,
            rmm::mr::device_memory_resource* mr) const;

  /**
   * @copydoc cudf::hash_join::full_join
   */
  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  full_join(cudf::table_view const& probe,
            std::optional<std::size_t> output_size,
            rmm::cuda_stream_view stream,
            rmm::mr::device_memory_resource* mr) const;

  /**
   * @copydoc cudf::hash_join::inner_join_size
   */
  [[nodiscard]] std::size_t inner_join_size(cudf::table_view const& probe,
                                            rmm::cuda_stream_view stream) const;

  /**
   * @copydoc cudf::hash_join::left_join_size
   */
  [[nodiscard]] std::size_t left_join_size(cudf::table_view const& probe,
                                           rmm::cuda_stream_view stream) const;

  /**
   * @copydoc cudf::hash_join::full_join_size
   */
  std::size_t full_join_size(cudf::table_view const& probe,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr) const;

 private:
  /**
   * @brief Probes the `_hash_table` built from `_build` for tuples in `probe_table`,
   * and returns the output indices of `build_table` and `probe_table` as a combined table,
   * i.e. if full join is specified as the join type then left join is called. Behavior
   * is undefined if the provided `output_size` is smaller than the actual output size.
   *
   * @throw cudf::logic_error if build table is empty and `JoinKind == INNER_JOIN`.
   *
   * @tparam JoinKind The type of join to be performed.
   *
   * @param probe_table Table of probe side columns to join.
   * @param output_size Optional value which allows users to specify the exact output size.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the returned vectors.
   *
   * @return Join output indices vector pair.
   */
  template <cudf::detail::join_kind JoinKind>
  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  probe_join_indices(cudf::table_view const& probe_table,
                     std::optional<std::size_t> output_size,
                     rmm::cuda_stream_view stream,
                     rmm::mr::device_memory_resource* mr) const;

  /**
   * @copydoc cudf::detail::hash_join::probe_join_indices
   *
   * @throw cudf::logic_error if probe table is empty.
   * @throw cudf::logic_error if the size of probe table exceeds `MAX_JOIN_SIZE`.
   * @throw cudf::logic_error if the number of columns in build table and probe table do not match.
   * @throw cudf::logic_error if the column data types in build table and probe table do not match.
   */
  template <cudf::detail::join_kind JoinKind>
  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  compute_hash_join(cudf::table_view const& probe,
                    std::optional<std::size_t> output_size,
                    rmm::cuda_stream_view stream,
                    rmm::mr::device_memory_resource* mr) const;
};
}  // namespace detail
}  // namespace cudf
