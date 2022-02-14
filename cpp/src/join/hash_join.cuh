/*
 * Copyright (c) 2019-2022, NVIDIA CORPORATION.
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

#include <join/join_common_utils.cuh>
#include <join/join_common_utils.hpp>

#include <cudf/detail/concatenate.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/sequence.h>

#include <cstddef>
#include <limits>

namespace cudf {
namespace detail {

/**
 * @brief Remaps a hash value to a new value if it is equal to the specified sentinel value.
 *
 * @param hash The hash value to potentially remap
 * @param sentinel The reserved value
 */
template <typename H, typename S>
constexpr auto remap_sentinel_hash(H hash, S sentinel)
{
  // Arbitrarily choose hash - 1
  return (hash == sentinel) ? (hash - 1) : hash;
}

/**
 * @brief Device functor to create a pair of hash value and index for a given row.
 */
class make_pair_function {
 public:
  CUDF_HOST_DEVICE make_pair_function(row_hash const& hash,
                                      hash_value_type const empty_key_sentinel)
    : _hash{hash}, _empty_key_sentinel{empty_key_sentinel}
  {
  }

  __device__ __forceinline__ cudf::detail::pair_type operator()(size_type i) const noexcept
  {
    // Compute the hash value of row `i`
    auto row_hash_value = remap_sentinel_hash(_hash(i), _empty_key_sentinel);
    return cuco::make_pair(row_hash_value, i);
  }

 private:
  row_hash _hash;
  hash_value_type const _empty_key_sentinel;
};

/**
 * @brief Calculates the exact size of the join output produced when
 * joining two tables together.
 *
 * @throw cudf::logic_error if JoinKind is not INNER_JOIN or LEFT_JOIN
 *
 * @tparam JoinKind The type of join to be performed
 * @tparam multimap_type The type of the hash table
 *
 * @param build_table The right hand table
 * @param probe_table The left hand table
 * @param hash_table A hash table built on the build table that maps the index
 * of every row to the hash value of that row.
 * @param nulls_equal Flag to denote nulls are equal or not.
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return The exact size of the output of the join operation
 */
template <join_kind JoinKind, typename multimap_type>
std::size_t compute_join_output_size(table_device_view build_table,
                                     table_device_view probe_table,
                                     multimap_type const& hash_table,
                                     bool const has_nulls,
                                     cudf::null_equality const nulls_equal,
                                     rmm::cuda_stream_view stream)
{
  const size_type build_table_num_rows{build_table.num_rows()};
  const size_type probe_table_num_rows{probe_table.num_rows()};

  // If the build table is empty, we know exactly how large the output
  // will be for the different types of joins and can return immediately
  if (0 == build_table_num_rows) {
    switch (JoinKind) {
      // Inner join with an empty table will have no output
      case join_kind::INNER_JOIN: return 0;

      // Left join with an empty table will have an output of NULL rows
      // equal to the number of rows in the probe table
      case join_kind::LEFT_JOIN: return probe_table_num_rows;

      default: CUDF_FAIL("Unsupported join type");
    }
  }

  auto const probe_nulls = cudf::nullate::DYNAMIC{has_nulls};
  pair_equality equality{probe_table, build_table, probe_nulls, nulls_equal};

  row_hash hash_probe{probe_nulls, probe_table};
  auto const empty_key_sentinel = hash_table.get_empty_key_sentinel();
  make_pair_function pair_func{hash_probe, empty_key_sentinel};

  auto iter = cudf::detail::make_counting_transform_iterator(0, pair_func);

  std::size_t size;
  if constexpr (JoinKind == join_kind::LEFT_JOIN) {
    size = hash_table.pair_count_outer(iter, iter + probe_table_num_rows, equality, stream.value());
  } else {
    size = hash_table.pair_count(iter, iter + probe_table_num_rows, equality, stream.value());
  }

  return size;
}

std::pair<std::unique_ptr<table>, std::unique_ptr<table>> get_empty_joined_table(
  table_view const& probe, table_view const& build);

std::unique_ptr<cudf::table> combine_table_pair(std::unique_ptr<cudf::table>&& left,
                                                std::unique_ptr<cudf::table>&& right);

/**
 * @brief Builds the hash table based on the given `build_table`.
 *
 * @tparam MultimapType The type of the hash table
 *
 * @param build Table of columns used to build join hash.
 * @param hash_table Build hash table.
 * @param nulls_equal Flag to denote nulls are equal or not.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 *
 */
template <typename MultimapType>
void build_join_hash_table(cudf::table_view const& build,
                           MultimapType& hash_table,
                           null_equality const nulls_equal,
                           rmm::cuda_stream_view stream)
{
  auto build_table_ptr = cudf::table_device_view::create(build, stream);

  CUDF_EXPECTS(0 != build_table_ptr->num_columns(), "Selected build dataset is empty");
  CUDF_EXPECTS(0 != build_table_ptr->num_rows(), "Build side table has no rows");

  row_hash hash_build{nullate::DYNAMIC{cudf::has_nulls(build)}, *build_table_ptr};
  auto const empty_key_sentinel = hash_table.get_empty_key_sentinel();
  make_pair_function pair_func{hash_build, empty_key_sentinel};

  auto iter = cudf::detail::make_counting_transform_iterator(0, pair_func);

  size_type const build_table_num_rows{build_table_ptr->num_rows()};
  if (nulls_equal == cudf::null_equality::EQUAL or (not nullable(build))) {
    hash_table.insert(iter, iter + build_table_num_rows, stream.value());
  } else {
    thrust::counting_iterator<size_type> stencil(0);
    auto const row_bitmask = cudf::detail::bitmask_and(build, stream).first;
    row_is_valid pred{static_cast<bitmask_type const*>(row_bitmask.data())};

    // insert valid rows
    hash_table.insert_if(iter, iter + build_table_num_rows, stencil, pred, stream.value());
  }
}
}  // namespace detail

struct hash_join::hash_join_impl {
 public:
  hash_join_impl() = delete;
  ~hash_join_impl();
  hash_join_impl(hash_join_impl const&) = delete;
  hash_join_impl(hash_join_impl&&)      = delete;
  hash_join_impl& operator=(hash_join_impl const&) = delete;
  hash_join_impl& operator=(hash_join_impl&&) = delete;

 private:
  bool const _is_empty;
  cudf::null_equality const _nulls_equal;
  cudf::table_view _build;
  std::vector<std::unique_ptr<cudf::column>> _created_null_columns;
  cudf::structs::detail::flattened_table _flattened_build_table;
  cudf::detail::multimap_type _hash_table;

 public:
  /**
   * @brief Constructor that internally builds the hash table based on the given `build` table
   *
   * @throw cudf::logic_error if the number of columns in `build` table is 0.
   * @throw cudf::logic_error if the number of rows in `build` table exceeds MAX_JOIN_SIZE.
   *
   * @param build The build table, from which the hash table is built.
   * @param compare_nulls Controls whether null join-key values should match or not.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   */
  hash_join_impl(cudf::table_view const& build,
                 null_equality compare_nulls,
                 rmm::cuda_stream_view stream = rmm::cuda_stream_default);

  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  inner_join(cudf::table_view const& probe,
             std::optional<std::size_t> output_size,
             rmm::cuda_stream_view stream,
             rmm::mr::device_memory_resource* mr) const;

  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  left_join(cudf::table_view const& probe,
            std::optional<std::size_t> output_size,
            rmm::cuda_stream_view stream,
            rmm::mr::device_memory_resource* mr) const;

  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  full_join(cudf::table_view const& probe,
            std::optional<std::size_t> output_size,
            rmm::cuda_stream_view stream,
            rmm::mr::device_memory_resource* mr) const;

  [[nodiscard]] std::size_t inner_join_size(cudf::table_view const& probe,
                                            rmm::cuda_stream_view stream) const;

  [[nodiscard]] std::size_t left_join_size(cudf::table_view const& probe,
                                           rmm::cuda_stream_view stream) const;

  std::size_t full_join_size(cudf::table_view const& probe,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr) const;

 private:
  template <cudf::detail::join_kind JoinKind>
  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  compute_hash_join(cudf::table_view const& probe,
                    std::optional<std::size_t> output_size,
                    rmm::cuda_stream_view stream,
                    rmm::mr::device_memory_resource* mr) const;

  /**
   * @brief Probes the `_hash_table` built from `_build` for tuples in `probe_table`,
   * and returns the output indices of `build_table` and `probe_table` as a combined table,
   * i.e. if full join is specified as the join type then left join is called. Behavior
   * is undefined if the provided `output_size` is smaller than the actual output size.
   *
   * @throw cudf::logic_error if hash table is null.
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
};

}  // namespace cudf
