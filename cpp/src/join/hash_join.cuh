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

#include "join_common_utils.hpp"
#include "join_kernels.cuh"
#include "full_join.cuh"

namespace cudf {

namespace detail {

using JoinType = ::cudf::detail::JoinType;
using output_index_type = ::cudf::detail::output_index_type;

/* --------------------------------------------------------------------------*/
/**
 * @brief  Gives an estimate of the size of the join output produced when
 * joining two tables together. If the two tables are of relatively equal size,
 * then the returned output size will be the exact output size. However, if the
 * probe table is significantly larger than the build table, then we attempt
 * to estimate the output size by using only a subset of the rows in the probe table.
 *
 * @throws cudf::logic_error if join_type is not INNER_JOIN or LEFT_JOIN
 *
 * @param build_table The right hand table
 * @param probe_table The left hand table
 * @param hash_table A hash table built on the build table that maps the index
 * of every row to the hash value of that row.
 *
 * @returns An estimate of the size of the output of the join operation
 */
/* ----------------------------------------------------------------------------*/
template <JoinType join_type,
          typename multimap_type>
std::unique_ptr<experimental::scalar_type_t<size_type>>
estimate_join_output_size(
    table_device_view build_table,
    table_device_view probe_table,
    multimap_type hash_table,
    cudaStream_t stream,
    rmm::mr::device_memory_resource* mr) {

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
    switch(join_type) {

      // Inner join with an empty table will have no output
      case JoinType::INNER_JOIN:
        { return std::make_unique<experimental::scalar_type_t<size_type>>(
            0, true, stream, mr);
        }

      // Left join with an empty table will have an output of NULL rows
      // equal to the number of rows in the probe table
      case JoinType::LEFT_JOIN:
        { return std::make_unique<experimental::scalar_type_t<size_type>>(
            probe_table_num_rows, true, stream, mr);
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
  experimental::scalar_type_t<size_type> e(0, true, stream, mr);

  CUDA_CHECK_LAST();

  constexpr int block_size {DEFAULT_CUDA_BLOCK_SIZE};
  int numBlocks {-1};

  CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &numBlocks, compute_join_output_size<join_type, multimap_type, block_size, DEFAULT_CUDA_CACHE_SIZE>,
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

    e.set_value(0);
    auto size_estimate = get_scalar_device_view(e);

    RowHash hash_probe{probe_table};
    row_equality equality{probe_table, build_table};
    // Probe the hash table without actually building the output to simply
    // find what the size of the output will be.
    compute_join_output_size<join_type,
                             multimap_type,
                             block_size,
                             DEFAULT_CUDA_CACHE_SIZE>
    <<<numBlocks * num_sms, block_size, 0, stream>>>(
        hash_table,
        build_table,
        probe_table,
        hash_probe,
        equality,
        sample_probe_num_rows,
        size_estimate);
    CUDA_CHECK_LAST();

    // Only in case subset of probe table is chosen,
    // increase the estimated output size by a factor of the ratio between the
    // probe and build tables
    if(sample_probe_num_rows < probe_table_num_rows) {
      h_size_estimate = e.value() * probe_to_build_ratio;
    } else {
      h_size_estimate = e.value();
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

  e.set_value(h_size_estimate);
  return std::make_unique<
    experimental::scalar_type_t<size_type>>(
        std::move(e));
}

template <JoinType join_type, typename index_type>
std::enable_if_t<(join_type != JoinType::FULL_JOIN),
std::unique_ptr<experimental::table>>
get_base_hash_join_indices(
    table_view const& left,
    table_view const& right,
    bool flip_join_indices,
    cudaStream_t stream,
    rmm::mr::device_memory_resource* mr) {

  if ((join_type == JoinType::INNER_JOIN) && (right.num_rows() > left.num_rows())) {
    return get_base_hash_join_indices<join_type, index_type>(right, left,  true, stream, mr);
  }
  if ((join_type == JoinType::LEFT_JOIN) && (right.num_rows() == 0)) {
    rmm::device_buffer sequence_indices{
      sizeof(index_type)*left.num_rows(), stream, mr};
    thrust::device_ptr<index_type> sequence_indices_ptr(
        static_cast<index_type*>(sequence_indices.data()));
    thrust::sequence(
        rmm::exec_policy(stream)->on(stream),
        sequence_indices_ptr,
        sequence_indices_ptr + left.num_rows(),
        0);
    rmm::device_buffer invalid_indices{
      sizeof(index_type)*left.num_rows(), stream, mr};
    thrust::device_ptr<index_type> inv_index_ptr(
        static_cast<index_type*>(invalid_indices.data()));
    thrust::fill(
        rmm::exec_policy(stream)->on(stream),
        inv_index_ptr,
        inv_index_ptr + left.num_rows(),
        JoinNoneValue);
    return get_indices_table<index_type>(
        std::move(sequence_indices), std::move(invalid_indices),
        left.num_rows(), stream, mr);
  }

  //TODO : Add trivial left join case and exit early

  using multimap_type = multimap_T<index_type>;

  //TODO : attach stream to hash map class according to PR discussion #3272

  auto build_table = table_device_view::create(right, stream);
  const size_type build_table_num_rows{build_table->num_rows()};

  // Probe with the left table
  auto probe_table = table_device_view::create(left, stream);

  // Hash table size must be at least 1 in order to have a valid allocation.
  // Even if the hash table will be empty, it still must be allocated for the
  // probing phase in the event of an outer join
  size_t const hash_table_size =
      std::max(compute_hash_table_size(build_table_num_rows), size_t{1});

  auto hash_table = multimap_type::create(hash_table_size);

  // build the hash table
  if (build_table_num_rows > 0) {
    RowHash hash_build{*build_table};
    experimental::scalar_type_t<int> f(0, true, stream, mr);
    auto failure = get_scalar_device_view(f);
    constexpr int block_size{DEFAULT_CUDA_BLOCK_SIZE};
    experimental::detail::grid_1d config(build_table_num_rows, block_size);
    build_hash_table<<<config.num_blocks, config.num_threads_per_block, 0, stream>>>(
        *hash_table,
        *build_table,
        hash_build,
        build_table_num_rows,
        failure);
    // Check error code from the kernel
    if (f.value() == 1) { CUDF_FAIL("Hash Table insert failure."); }
  }

  auto estimated_join_output_size =
    estimate_join_output_size<join_type, multimap_type>(
        *build_table, *probe_table, *hash_table, stream, mr);

  size_type estimated_size = estimated_join_output_size->value();
  // If the estimated output size is zero, return immediately
  if (estimated_size == 0) {
    return get_empty_index_table<index_type>(stream, mr);
  }

  // Because we are approximating the number of joined elements, our approximation
  // might be incorrect and we might have underestimated the number of joined elements.
  // As such we will need to de-allocate memory and re-allocate memory to ensure
  // that the final output is correct.
  experimental::scalar_type_t<size_type> global_write_index(0, true, stream, mr);
  size_type join_size{0};

  rmm::device_buffer left_indices{0, stream, mr};
  rmm::device_buffer right_indices{0, stream, mr};
  while (true) {
    left_indices = rmm::device_buffer{
      sizeof(index_type)*estimated_size,
      stream, mr};
    right_indices = rmm::device_buffer{
      sizeof(index_type)*estimated_size,
      stream, mr};

    constexpr int block_size{DEFAULT_CUDA_BLOCK_SIZE};
    experimental::detail::grid_1d config(probe_table->num_rows(), block_size);
    global_write_index.set_value(0);
    auto write_index = get_scalar_device_view(global_write_index);

    RowHash hash_probe{*probe_table};
    row_equality equality{*probe_table, *build_table};
    probe_hash_table<join_type,
                     multimap_type,
                     hash_value_type,
                     output_index_type,
                     block_size,
                     DEFAULT_CUDA_CACHE_SIZE>
    <<<config.num_blocks, config.num_threads_per_block, 0, stream>>>(
        *hash_table,
        *build_table,
        *probe_table,
        hash_probe,
        equality,
        probe_table->num_rows(),
        static_cast<index_type*>(left_indices.data()),
        static_cast<index_type*>(right_indices.data()),
        write_index,
        estimated_size,
        flip_join_indices);

    CUDA_CHECK_LAST();

    join_size = global_write_index.value();
    if (estimated_size < join_size) {
      estimated_size *= 2;
    } else {
      break;
    }
  }

  left_indices.resize(sizeof(index_type)*join_size, stream);
  right_indices.resize(sizeof(index_type)*join_size, stream);

  return get_indices_table<index_type>(
      std::move(left_indices), std::move(right_indices),
      join_size, stream, mr);

}

}//namespace detail

}//namespace cudf
