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

#include <cudf/detail/concatenate.cuh>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/gather.hpp>
#include <join/join_common_utils.hpp>
#include <join/join_kernels.cuh>

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
 * @brief Calculates the exact size of the join output produced when
 * joining two tables together.
 *
 * @throw cudf::logic_error if JoinKind is not INNER_JOIN or LEFT_JOIN
 * @throw cudf::logic_error if the exact size overflows cudf::size_type
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
 * @return The exact size of the output of the join operation
 */
template <join_kind JoinKind, typename multimap_type>
std::size_t compute_join_output_size(table_device_view build_table,
                                     table_device_view probe_table,
                                     multimap_type const& hash_table,
                                     null_equality compare_nulls,
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

  // Allocate storage for the counter used to get the size of the join output
  std::size_t h_size{0};
  rmm::device_scalar<std::size_t> d_size(0, stream);

  CHECK_CUDA(stream.value());

  constexpr int block_size{DEFAULT_JOIN_BLOCK_SIZE};
  int numBlocks{-1};

  CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &numBlocks, compute_join_output_size<JoinKind, multimap_type, block_size>, block_size, 0));

  int dev_id{-1};
  CUDA_TRY(cudaGetDevice(&dev_id));

  int num_sms{-1};
  CUDA_TRY(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

  row_hash hash_probe{probe_table};
  row_equality equality{probe_table, build_table, compare_nulls == null_equality::EQUAL};
  // Probe the hash table without actually building the output to simply
  // find what the size of the output will be.
  compute_join_output_size<JoinKind, multimap_type, block_size>
    <<<numBlocks * num_sms, block_size, 0, stream.value()>>>(hash_table,
                                                             build_table,
                                                             probe_table,
                                                             hash_probe,
                                                             equality,
                                                             probe_table_num_rows,
                                                             d_size.data());

  CHECK_CUDA(stream.value());
  h_size = d_size.value(stream);

  return h_size;
}

/**
 * @brief Computes the trivial left join operation for the case when the
 * right table is empty. In this case all the valid indices of the left table
 * are returned with their corresponding right indices being set to
 * JoinNoneValue, i.e. -1.
 *
 * @param left Table of left columns to join
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the result
 *
 * @return Join output indices vector pair
 */
inline std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                 std::unique_ptr<rmm::device_uvector<size_type>>>
get_trivial_left_join_indices(
  table_view const& left,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource())
{
  auto left_indices = std::make_unique<rmm::device_uvector<size_type>>(left.num_rows(), stream, mr);
  thrust::sequence(rmm::exec_policy(stream), left_indices->begin(), left_indices->end(), 0);
  auto right_indices =
    std::make_unique<rmm::device_uvector<size_type>>(left.num_rows(), stream, mr);
  thrust::fill(
    rmm::exec_policy(stream), right_indices->begin(), right_indices->end(), JoinNoneValue);
  return std::make_pair(std::move(left_indices), std::move(right_indices));
}

std::pair<std::unique_ptr<table>, std::unique_ptr<table>> get_empty_joined_table(
  table_view const& probe, table_view const& build);

std::unique_ptr<cudf::table> combine_table_pair(std::unique_ptr<cudf::table>&& left,
                                                std::unique_ptr<cudf::table>&& right);

VectorPair concatenate_vector_pairs(VectorPair& a, VectorPair& b, rmm::cuda_stream_view stream);

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
get_left_join_indices_complement(
  std::unique_ptr<rmm::device_uvector<size_type>>& right_indices,
  size_type left_table_row_count,
  size_type right_table_row_count,
  rmm::cuda_stream_view stream,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());

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
  std::vector<std::unique_ptr<cudf::column>> _created_null_columns;
  std::unique_ptr<cudf::detail::multimap_type, std::function<void(cudf::detail::multimap_type*)>>
    _hash_table;

 public:
  /**
   * @brief Constructor that internally builds the hash table based on the given `build` table
   *
   * @throw cudf::logic_error if the number of columns in `build` table is 0.
   * @throw cudf::logic_error if the number of rows in `build` table exceeds MAX_JOIN_SIZE.
   *
   * @param build The build table, from which the hash table is built.
   * @param compare_nulls Controls whether null join-key values should match or not.
   */
  hash_join_impl(cudf::table_view const& build,
                 null_equality compare_nulls,
                 rmm::cuda_stream_view stream = rmm::cuda_stream_default);

  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  inner_join(cudf::table_view const& probe,
             null_equality compare_nulls,
             std::optional<std::size_t> output_size,
             rmm::cuda_stream_view stream,
             rmm::mr::device_memory_resource* mr) const;

  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  left_join(cudf::table_view const& probe,
            null_equality compare_nulls,
            std::optional<std::size_t> output_size,
            rmm::cuda_stream_view stream,
            rmm::mr::device_memory_resource* mr) const;

  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  full_join(cudf::table_view const& probe,
            null_equality compare_nulls,
            std::optional<std::size_t> output_size,
            rmm::cuda_stream_view stream,
            rmm::mr::device_memory_resource* mr) const;

  std::size_t inner_join_size(cudf::table_view const& probe,
                              null_equality compare_nulls,
                              rmm::cuda_stream_view stream) const;

  std::size_t left_join_size(cudf::table_view const& probe,
                             null_equality compare_nulls,
                             rmm::cuda_stream_view stream) const;

  std::size_t full_join_size(cudf::table_view const& probe,
                             null_equality compare_nulls,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr) const;

 private:
  template <cudf::detail::join_kind JoinKind>
  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  compute_hash_join(cudf::table_view const& probe,
                    null_equality compare_nulls,
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
   * @param compare_nulls Controls whether null join-key values should match or not.
   * @param output_size Optional value which allows users to specify the exact output size.
   * @param stream CUDA stream used for device memory operations and kernel launches.
   * @param mr Device memory resource used to allocate the returned vectors.
   *
   * @return Join output indices vector pair.
   */
  template <cudf::detail::join_kind JoinKind>
  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  probe_join_indices(cudf::table_view const& probe,
                     null_equality compare_nulls,
                     std::optional<std::size_t> output_size,
                     rmm::cuda_stream_view stream,
                     rmm::mr::device_memory_resource* mr) const;
};

}  // namespace cudf
