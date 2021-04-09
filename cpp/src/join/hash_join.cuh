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
                                    rmm::cuda_stream_view stream)
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

  CHECK_CUDA(stream.value());

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

    size_estimate.set_value_zero(stream);

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
                                                               sample_probe_num_rows,
                                                               size_estimate.data());
    CHECK_CUDA(stream.value());

    // Only in case subset of probe table is chosen,
    // increase the estimated output size by a factor of the ratio between the
    // probe and build tables
    if (sample_probe_num_rows < probe_table_num_rows) {
      h_size_estimate = size_estimate.value(stream) * probe_to_build_ratio;
    } else {
      h_size_estimate = size_estimate.value(stream);
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
             rmm::cuda_stream_view stream,
             rmm::mr::device_memory_resource* mr) const;

  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  left_join(cudf::table_view const& probe,
            null_equality compare_nulls,
            rmm::cuda_stream_view stream,
            rmm::mr::device_memory_resource* mr) const;

  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  full_join(cudf::table_view const& probe,
            null_equality compare_nulls,
            rmm::cuda_stream_view stream,
            rmm::mr::device_memory_resource* mr) const;

 private:
  template <cudf::detail::join_kind JoinKind>
  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  compute_hash_join(cudf::table_view const& probe,
                    null_equality compare_nulls,
                    rmm::cuda_stream_view stream,
                    rmm::mr::device_memory_resource* mr) const;

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
   * @param mr Device memory resource used to allocate the returned vectors.
   *
   * @return Join output indices vector pair.
   */
  template <cudf::detail::join_kind JoinKind>
  std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
            std::unique_ptr<rmm::device_uvector<size_type>>>
  probe_join_indices(cudf::table_view const& probe,
                     null_equality compare_nulls,
                     rmm::cuda_stream_view stream,
                     rmm::mr::device_memory_resource* mr) const;
};

}  // namespace cudf
