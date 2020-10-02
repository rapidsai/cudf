/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/join.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>

#include "cudf/types.hpp"
#include "join_common_utils.hpp"
#include "join_kernels.cuh"

#include <limits>

namespace cudf {
namespace detail {
/**
 * @brief Gives an estimate of the size of the join output produced when
 * joining two tables together.
 *
 * If the two tables are of relatively equal size, then the returned output
 * size will be the exact output size. However, if the probe table is
 * significantly larger than the build table, then we attempt to estimate the
 * output size by using only a subset of the rows in the probe table.
 *
 * @throw cudf::logic_error if JoinKind is not INNER_JOIN or LEFT_JOIN
 * @throw cudf::logic_error if the estimated size overflows cudf::size_type
 *
 * @tparam JoinKind The type of join to be performed
 * @tparam multimap_type The type of the hash table
 *
 * @param build_table The right hand table
 * @param probe_table The left hand table
 * @param hash_table A hash table built on the build table that maps the index
 * of every row to the hash value of that row.
 * @param compare_nulls Controls whether null join-key values should match or not.
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return An estimate of the size of the output of the join operation
 */
template <join_kind JoinKind, typename multimap_type>
size_type estimate_join_output_size(table_device_view build_table,
                                    table_device_view probe_table,
                                    multimap_type const& hash_table,
                                    null_equality compare_nulls,
                                    cudaStream_t stream)
{
  using estimate_size_type = int64_t;  // use 64-bit size so we can detect overflow

  const size_type build_table_num_rows{build_table.num_rows()};
  const size_type probe_table_num_rows{probe_table.num_rows()};

  // If the probe table is significantly larger (5x) than the build table,
  // then we attempt to only use a subset of the probe table rows to compute an
  // estimate of the join output size.
  size_type probe_to_build_ratio{0};
  if (build_table_num_rows > 0) {
    probe_to_build_ratio = static_cast<size_type>(
      std::ceil(static_cast<float>(probe_table_num_rows) / build_table_num_rows));
  } else {
    // If the build table is empty, we know exactly how large the output
    // will be for the different types of joins and can return immediately
    switch (JoinKind) {
      // Inner join with an empty table will have no output
      case join_kind::INNER_JOIN: return 0;

      // Left join with an empty table will have an output of NULL rows
      // equal to the number of rows in the probe table
      case join_kind::LEFT_JOIN: return probe_table_num_rows;

      default: CUDF_FAIL("Unsupported join type");
    }
  }

  size_type sample_probe_num_rows{probe_table_num_rows};
  constexpr size_type MAX_RATIO{5};
  if (probe_to_build_ratio > MAX_RATIO) { sample_probe_num_rows = build_table_num_rows; }

  // Allocate storage for the counter used to get the size of the join output
  estimate_size_type h_size_estimate{0};
  rmm::device_scalar<estimate_size_type> size_estimate(0, stream);

  CHECK_CUDA(stream);

  constexpr int block_size{DEFAULT_JOIN_BLOCK_SIZE};
  int numBlocks{-1};

  CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &numBlocks, compute_join_output_size<JoinKind, multimap_type, block_size>, block_size, 0));

  int dev_id{-1};
  CUDA_TRY(cudaGetDevice(&dev_id));

  int num_sms{-1};
  CUDA_TRY(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

  // Continue probing with a subset of the probe table until either:
  // a non-zero output size estimate is found OR
  // all of the rows in the probe table have been sampled
  do {
    sample_probe_num_rows = std::min(sample_probe_num_rows, probe_table_num_rows);

    size_estimate.set_value(0);

    row_hash hash_probe{probe_table};
    row_equality equality{probe_table, build_table, compare_nulls == null_equality::EQUAL};
    // Probe the hash table without actually building the output to simply
    // find what the size of the output will be.
    compute_join_output_size<JoinKind, multimap_type, block_size>
      <<<numBlocks * num_sms, block_size, 0, stream>>>(hash_table,
                                                       build_table,
                                                       probe_table,
                                                       hash_probe,
                                                       equality,
                                                       sample_probe_num_rows,
                                                       size_estimate.data());
    CHECK_CUDA(stream);

    // Only in case subset of probe table is chosen,
    // increase the estimated output size by a factor of the ratio between the
    // probe and build tables
    if (sample_probe_num_rows < probe_table_num_rows) {
      h_size_estimate = size_estimate.value() * probe_to_build_ratio;
    } else {
      h_size_estimate = size_estimate.value();
    }

    // Detect overflow
    CUDF_EXPECTS(h_size_estimate <
                   static_cast<estimate_size_type>(std::numeric_limits<cudf::size_type>::max()),
                 "Maximum join output size exceeded");

    // If the size estimate is non-zero, then we have a valid estimate and can break
    // If sample_probe_num_rows >= probe_table_num_rows, then we've sampled the entire
    // probe table, in which case the estimate is exact and we can break
    if ((h_size_estimate > 0) || (sample_probe_num_rows >= probe_table_num_rows)) { break; }

    // If the size estimate is zero, then double the number of sampled rows in the probe
    // table. Reduce the ratio of the number of probe rows sampled to the number of rows
    // in the build table by the same factor
    if (0 == h_size_estimate) {
      constexpr size_type GROW_RATIO{2};
      sample_probe_num_rows *= GROW_RATIO;
      probe_to_build_ratio =
        static_cast<size_type>(std::ceil(static_cast<float>(probe_to_build_ratio) / GROW_RATIO));
    }

  } while (true);

  return static_cast<cudf::size_type>(h_size_estimate);
}

/**
 * @brief Computes the trivial left join operation for the case when the
 * right table is empty. In this case all the valid indices of the left table
 * are returned with their corresponding right indices being set to
 * JoinNoneValue, i.e. -1.
 *
 * @param left Table of left columns to join
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return Join output indices vector pair
 */
inline std::pair<rmm::device_vector<size_type>, rmm::device_vector<size_type>>
get_trivial_left_join_indices(table_view const& left, cudaStream_t stream)
{
  rmm::device_vector<size_type> left_indices(left.num_rows());
  thrust::sequence(
    rmm::exec_policy(stream)->on(stream), left_indices.begin(), left_indices.end(), 0);
  rmm::device_vector<size_type> right_indices(left.num_rows());
  thrust::fill(rmm::exec_policy(stream)->on(stream),
               right_indices.begin(),
               right_indices.end(),
               JoinNoneValue);
  return std::make_pair(std::move(left_indices), std::move(right_indices));
}

std::unique_ptr<cudf::table> combine_table_pair(std::unique_ptr<cudf::table>&& left,
                                                std::unique_ptr<cudf::table>&& right);

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
  cudf::table_view _build;
  cudf::table_view _build_selected;
  std::vector<size_type> _build_on;
  std::unique_ptr<cudf::detail::multimap_type, std::function<void(cudf::detail::multimap_type*)>>
    _hash_table;

 public:
  /**
   * @brief Constructor that internally builds the hash table based on the given `build` table and
   * column indices specified by `build_on` for subsequent probe calls.
   *
   * @throw cudf::logic_error if the number of columns in `build` table is 0.
   * @throw cudf::logic_error if the number of rows in `build` table exceeds MAX_JOIN_SIZE.
   * @throw std::out_of_range if elements of `build_on` exceed the number of columns in the `build`
   * table.
   *
   * @param build The build table, from which the hash table is built.
   * @param build_on The column indices from `build` to join on.
   */
  hash_join_impl(cudf::table_view const& build, std::vector<size_type> const& build_on);

  std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::table>> inner_join(
    cudf::table_view const& probe,
    std::vector<size_type> const& probe_on,
    std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
    common_columns_output_side common_columns_output_side,
    null_equality compare_nulls,
    rmm::mr::device_memory_resource* mr) const;

  std::unique_ptr<cudf::table> left_join(
    cudf::table_view const& probe,
    std::vector<size_type> const& probe_on,
    std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
    null_equality compare_nulls,
    rmm::mr::device_memory_resource* mr) const;

  std::unique_ptr<cudf::table> full_join(
    cudf::table_view const& probe,
    std::vector<size_type> const& probe_on,
    std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
    null_equality compare_nulls,
    rmm::mr::device_memory_resource* mr) const;

 private:
  /**
   * @brief Performs hash join by probing the columns provided in `probe` as per
   * the joining indices given in `probe_on` and returns a (`probe`, `_build`) table pair, which
   * contains the probe and build portions of the logical joined table respectively.
   *
   * @throw cudf::logic_error if `columns_in_common` contains a pair of indices
   * (`P`, `B`) where `P` does not exist in `probe_on` or `B` does not exist in
   * `_build_on`.
   * @throw cudf::logic_error if `columns_in_common` contains a pair of indices
   * (`P`, `B`) such that the location of `P` within `probe_on` is not equal to
   * the location of `B` within `_build_on`.
   * @throw cudf::logic_error if the number of elements in `probe_on` and
   * `_build_on` are not equal.
   * @throw cudf::logic_error if the number of columns in `probe` is 0.
   * @throw cudf::logic_error if the number of rows in `probe` table exceeds MAX_JOIN_SIZE.
   * @throw std::out_of_range if elements of `probe_on` exceed the number of columns in the `probe`
   * table.
   * @throw cudf::logic_error if types do not match between joining columns.
   *
   * @tparam JoinKind The type of join to be performed.
   *
   * @param probe The probe table.
   * @param probe_on The column's indices from `probe` to join on.
   * Column `i` from `probe_on` will be compared against column `i` of `_build_on`.
   * @param columns_in_common is a vector of pairs of column indices into
   * `probe` and `_build`, respectively, that are "in common". For "common"
   * columns, only a single output column will be produced, which is gathered
   * from `probe_on` columns. Else, for every column in `probe_on` and `_build_on`,
   * an output column will be produced. For each of these pairs (P, B), P
   * should exist in `probe_on` and B should exist in `_build_on`.
   * @param common_columns_output_side @see cudf::hash_join::common_columns_output_side.
   * @param compare_nulls Controls whether null join-key values should match or not.
   * @param mr Device memory resource used to allocate the returned table's device memory.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   *
   * @return Table pair of (`probe`, `_build`) of joining both tables on the columns
   * specified by `probe_on` and `_build_on`. The resulting table pair will be joined columns of
   * (`probe(including common columns)`, `_build(excluding common columns)`) if
   * `common_columns_output_side` is `PROBE`, or (`probe(excluding common columns)`,
   * `_build(including common columns)`) if `common_columns_output_side` is `BUILD`.
   */
  template <cudf::detail::join_kind JoinKind>
  std::pair<std::unique_ptr<cudf::table>, std::unique_ptr<cudf::table>> compute_hash_join(
    cudf::table_view const& probe,
    std::vector<size_type> const& probe_on,
    std::vector<std::pair<cudf::size_type, cudf::size_type>> const& columns_in_common,
    common_columns_output_side common_columns_output_side,
    null_equality compare_nulls,
    rmm::mr::device_memory_resource* mr,
    cudaStream_t stream = 0) const;

  /**
   * @brief Probes the `_hash_table` built from `_build` for tuples in `probe_table`,
   * and returns the output indices of `build_table` and `probe_table` as a combined table,
   * i.e. if full join is specified as the join type then left join is called.
   *
   * @throw cudf::logic_error if hash table is null.
   *
   * @tparam JoinKind The type of join to be performed.
   *
   * @param probe_table Table of probe side columns to join.
   * @param compare_nulls Controls whether null join-key values should match or not.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   *
   * @return Join output indices vector pair.
   */
  template <cudf::detail::join_kind JoinKind>
  std::enable_if_t<JoinKind != cudf::detail::join_kind::FULL_JOIN,
                   std::pair<rmm::device_vector<size_type>, rmm::device_vector<size_type>>>
  probe_join_indices(cudf::table_view const& probe,
                     null_equality compare_nulls,
                     cudaStream_t stream) const;
};

}  // namespace cudf
