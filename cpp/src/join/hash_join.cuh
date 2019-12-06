/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf/table/table_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <hash/concurrent_unordered_multimap.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/detail/utilities/cuda.cuh>

#include <join/join_common_utils.hpp>
#include <join/join_kernels.cuh>
#include <join/full_join.cuh>

namespace cudf {

namespace experimental {

namespace detail {

/* --------------------------------------------------------------------------*/
/**
 * @brief  Gives an estimate of the size of the join output produced when
 * joining two tables together.
 *
 * If the two tables are of relatively equal size, then the returned output
 * size will be the exact output size. However, if the probe table is
 * significantly larger than the build table, then we attempt to estimate the
 * output size by using only a subset of the rows in the probe table.
 *
 * @throws cudf::logic_error if JoinKind is not INNER_JOIN or LEFT_JOIN
 *
 * @param build_table The right hand table
 * @param probe_table The left hand table
 * @param hash_table A hash table built on the build table that maps the index
 * of every row to the hash value of that row.
 *
 * @returns An estimate of the size of the output of the join operation
 */
/* ----------------------------------------------------------------------------*/
template <join_kind JoinKind,
          typename multimap_type>
std::unique_ptr<rmm::device_scalar<size_type>>
estimate_join_output_size(
    table_device_view build_table,
    table_device_view probe_table,
    multimap_type hash_table,
    cudaStream_t stream) {

  const size_type build_table_num_rows{build_table.num_rows()};
  const size_type probe_table_num_rows{probe_table.num_rows()};

  // If the probe table is significantly larger (5x) than the build table,
  // then we attempt to only use a subset of the probe table rows to compute an
  // estimate of the join output size.
  size_type probe_to_build_ratio{0};
  if(build_table_num_rows > 0) {
    probe_to_build_ratio =
      static_cast<size_type>(std::ceil(
            static_cast<float>(probe_table_num_rows)/build_table_num_rows));
  } else {
    // If the build table is empty, we know exactly how large the output
    // will be for the different types of joins and can return immediately
    switch(JoinKind) {

      // Inner join with an empty table will have no output
      case join_kind::INNER_JOIN:
        { return std::make_unique<rmm::device_scalar<size_type>>(
            0, stream);
        }

      // Left join with an empty table will have an output of NULL rows
      // equal to the number of rows in the probe table
      case join_kind::LEFT_JOIN:
        { return std::make_unique<rmm::device_scalar<size_type>>(
            probe_table_num_rows, stream);
        }

      default:
        CUDF_FAIL("Unsupported join type");
    }
  }

  size_type sample_probe_num_rows{probe_table_num_rows};
  constexpr size_type MAX_RATIO{5};
  if(probe_to_build_ratio > MAX_RATIO) {
    sample_probe_num_rows = build_table_num_rows;
  }

  // Allocate storage for the counter used to get the size of the join output
  size_type h_size_estimate{0};
  rmm::device_scalar<size_type> size_estimate(0, stream);

  CUDA_CHECK_LAST();

  constexpr int block_size {DEFAULT_JOIN_BLOCK_SIZE};
  int numBlocks {-1};

  CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &numBlocks, compute_join_output_size<JoinKind, multimap_type, block_size>,
    block_size, 0
  ));

  int dev_id {-1};
  CUDA_TRY(cudaGetDevice(&dev_id));

  int num_sms {-1};
  CUDA_TRY(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev_id));

  // Continue probing with a subset of the probe table until either:
  // a non-zero output size estimate is found OR
  // all of the rows in the probe table have been sampled
  do{

    sample_probe_num_rows = std::min(sample_probe_num_rows, probe_table_num_rows);

    size_estimate.set_value(0);

    row_hash hash_probe{probe_table};
    row_equality equality{probe_table, build_table};
    // Probe the hash table without actually building the output to simply
    // find what the size of the output will be.
    compute_join_output_size<JoinKind,
                             multimap_type,
                             block_size>
    <<<numBlocks * num_sms, block_size, 0, stream>>>(
        hash_table,
        build_table,
        probe_table,
        hash_probe,
        equality,
        sample_probe_num_rows,
        size_estimate.data());
    CUDA_CHECK_LAST();

    // Only in case subset of probe table is chosen,
    // increase the estimated output size by a factor of the ratio between the
    // probe and build tables
    if(sample_probe_num_rows < probe_table_num_rows) {
      h_size_estimate = size_estimate.value() * probe_to_build_ratio;
    } else {
      h_size_estimate = size_estimate.value();
    }

    // If the size estimate is non-zero, then we have a valid estimate and can break
    // If sample_probe_num_rows >= probe_table_num_rows, then we've sampled the entire
    // probe table, in which case the estimate is exact and we can break
    if((h_size_estimate > 0)
       || (sample_probe_num_rows >= probe_table_num_rows))
    {
      break;
    }

    // If the size estimate is zero, then double the number of sampled rows in the probe
    // table. Reduce the ratio of the number of probe rows sampled to the number of rows
    // in the build table by the same factor
    if(0 == h_size_estimate) {
      constexpr size_type GROW_RATIO{2};
      sample_probe_num_rows *= GROW_RATIO;
      probe_to_build_ratio = static_cast<size_type>(std::ceil(static_cast<float>(probe_to_build_ratio)/GROW_RATIO));
    }

  } while(true);

  size_estimate.set_value(h_size_estimate);
  return std::make_unique<rmm::device_scalar<size_type>>(
        std::move(size_estimate));
}

std::pair<rmm::device_vector<output_index_type>,
rmm::device_vector<output_index_type>>
get_trivial_left_join_indices(table_view const& left, cudaStream_t stream) {
  rmm::device_vector<output_index_type> left_indices(left.num_rows());
  thrust::sequence(
      rmm::exec_policy(stream)->on(stream),
      left_indices.begin(),
      left_indices.end(),
      0);
  rmm::device_vector<output_index_type> right_indices(left.num_rows());
  thrust::fill(
      rmm::exec_policy(stream)->on(stream),
      right_indices.begin(),
      right_indices.end(),
      JoinNoneValue);
  return std::make_pair(std::move(left_indices), std::move(right_indices));
}

/* --------------------------------------------------------------------------*/
/**
 * @brief  Computes the join operation between two tables and returns the
 * output indices of left and right table as a combined table
 *
 * @param left  Table of left columns to join
 * @param right Table of right  columns to join
 * @param flip_join_indices Flag that indicates whether the left and right
 * tables have been flipped, meaning the output indices should also be flipped.
 * @param stream stream on which all memory allocations and copies
 * will be performed
 * @tparam join_kind The type of join to be performed
 * @tparam index_type The datatype used for the output indices
 *
 * @returns Join output indices vector pair
 */
/* ----------------------------------------------------------------------------*/
template <join_kind JoinKind, typename index_type>
std::enable_if_t<(JoinKind != join_kind::FULL_JOIN),
std::pair<rmm::device_vector<size_type>,
rmm::device_vector<size_type>>>
get_base_hash_join_indices(
    table_view const& left,
    table_view const& right,
    bool flip_join_indices,
    cudaStream_t stream) {

  if ((JoinKind == join_kind::INNER_JOIN) && (right.num_rows() > left.num_rows())) {
    return get_base_hash_join_indices<JoinKind, index_type>(right, left, true, stream);
  }
  //Trivial left join case - exit early
  if ((JoinKind == join_kind::LEFT_JOIN) && (right.num_rows() == 0)) {
    return get_trivial_left_join_indices(left, stream);
  }

  using multimap_type = multimap_t<index_type>;

  //TODO : attach stream to hash map class according to PR discussion #3272

  auto build_table = table_device_view::create(right, stream);
  const size_type build_table_num_rows{build_table->num_rows()};

  // Probe with the left table
  auto probe_table = table_device_view::create(left, stream);

  // Hash table size must be at least 1 in order to have a valid allocation.
  // Even if the hash table will be empty, it still must be allocated for the
  // probing phase in the event of an outer join
  size_t const hash_table_size = compute_hash_table_size(build_table_num_rows);

  auto hash_table = multimap_type::create(hash_table_size);

  // build the hash table
  if (build_table_num_rows > 0) {
    row_hash hash_build{*build_table};
    rmm::device_scalar<int> failure(0, stream);
    constexpr int block_size{DEFAULT_JOIN_BLOCK_SIZE};
    experimental::detail::grid_1d config(build_table_num_rows, block_size);
    build_hash_table<<<config.num_blocks, config.num_threads_per_block, 0, stream>>>(
        *hash_table,
        *build_table,
        hash_build,
        build_table_num_rows,
        failure.data());
    // Check error code from the kernel
    if (failure.value() == 1) { CUDF_FAIL("Hash Table insert failure."); }
  }

  auto estimated_join_output_size =
    estimate_join_output_size<JoinKind, multimap_type>(
        *build_table, *probe_table, *hash_table, stream);

  size_type estimated_size = estimated_join_output_size->value(stream);
  // If the estimated output size is zero, return immediately
  if (estimated_size == 0) {
    rmm::device_vector<size_type> left_empty, right_empty;
    return std::make_pair(left_empty, right_empty);
  }

  // Because we are approximating the number of joined elements, our approximation
  // might be incorrect and we might have underestimated the number of joined elements.
  // As such we will need to de-allocate memory and re-allocate memory to ensure
  // that the final output is correct.
  rmm::device_scalar<size_type> write_index(0, stream);
  size_type join_size{0};

  rmm::device_vector<size_type> left_indices;
  rmm::device_vector<size_type> right_indices;
  while (true) {
    left_indices.resize(estimated_size);
    right_indices.resize(estimated_size);

    constexpr int block_size{DEFAULT_JOIN_BLOCK_SIZE};
    experimental::detail::grid_1d config(probe_table->num_rows(), block_size);
    write_index.set_value(0);

    row_hash hash_probe{*probe_table};
    row_equality equality{*probe_table, *build_table};
    probe_hash_table<JoinKind,
                     multimap_type,
                     hash_value_type,
                     output_index_type,
                     block_size,
                     DEFAULT_JOIN_CACHE_SIZE>
    <<<config.num_blocks, config.num_threads_per_block, 0, stream>>>(
        *hash_table,
        *build_table,
        *probe_table,
        hash_probe,
        equality,
        probe_table->num_rows(),
        static_cast<index_type*>(left_indices.data().get()),
        static_cast<index_type*>(right_indices.data().get()),
        write_index.data(),
        estimated_size,
        flip_join_indices);

    CUDA_CHECK_LAST();

    join_size = write_index.value();
    if (estimated_size < join_size) {
      estimated_size *= 2;
    } else {
      break;
    }
  }

  left_indices.resize(join_size);
  right_indices.resize(join_size);
  return std::make_pair(std::move(left_indices), std::move(right_indices));

}

}//namespace detail

} //namespace experimental

}//namespace cudf
