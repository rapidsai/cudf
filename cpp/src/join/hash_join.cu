/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <thrust/uninitialized_fill.h>
#include <join/hash_join.cuh>
#include <structs/utilities.hpp>

#include <cudf/detail/concatenate.cuh>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/gather.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cstddef>
#include <iostream>
#include <numeric>

namespace cudf {
namespace detail {

std::pair<std::unique_ptr<table>, std::unique_ptr<table>> get_empty_joined_table(
  table_view const &probe, table_view const &build)
{
  std::unique_ptr<table> empty_probe = empty_like(probe);
  std::unique_ptr<table> empty_build = empty_like(build);
  return std::make_pair(std::move(empty_probe), std::move(empty_build));
}

VectorPair concatenate_vector_pairs(VectorPair &a, VectorPair &b, rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS((a.first->size() == a.second->size()),
               "Mismatch between sizes of vectors in vector pair");
  CUDF_EXPECTS((b.first->size() == b.second->size()),
               "Mismatch between sizes of vectors in vector pair");
  if (a.first->is_empty()) {
    return std::move(b);
  } else if (b.first->is_empty()) {
    return std::move(a);
  }
  auto original_size = a.first->size();
  a.first->resize(a.first->size() + b.first->size(), stream);
  a.second->resize(a.second->size() + b.second->size(), stream);
  thrust::copy(
    rmm::exec_policy(stream), b.first->begin(), b.first->end(), a.first->begin() + original_size);
  thrust::copy(rmm::exec_policy(stream),
               b.second->begin(),
               b.second->end(),
               a.second->begin() + original_size);
  return std::move(a);
}

template <typename T>
struct valid_range {
  T start, stop;
  __host__ __device__ valid_range(const T begin, const T end) : start(begin), stop(end) {}

  __host__ __device__ __forceinline__ bool operator()(const T index)
  {
    return ((index >= start) && (index < stop));
  }
};

/**
 * @brief  Creates a table containing the complement of left join indices.
 * This table has two columns. The first one is filled with JoinNoneValue(-1)
 * and the second one contains values from 0 to right_table_row_count - 1
 * excluding those found in the right_indices column.
 *
 * @param right_indices Vector of indices
 * @param left_table_row_count Number of rows of left table
 * @param right_table_row_count Number of rows of right table
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned vectors.
 *
 * @return Pair of vectors containing the left join indices complement
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
get_left_join_indices_complement(std::unique_ptr<rmm::device_uvector<size_type>> &right_indices,
                                 size_type left_table_row_count,
                                 size_type right_table_row_count,
                                 rmm::cuda_stream_view stream,
                                 rmm::mr::device_memory_resource *mr)
{
  // Get array of indices that do not appear in right_indices

  // Vector allocated for unmatched result
  auto right_indices_complement =
    std::make_unique<rmm::device_uvector<size_type>>(right_table_row_count, stream);

  // If left table is empty in a full join call then all rows of the right table
  // should be represented in the joined indices. This is an optimization since
  // if left table is empty and full join is called all the elements in
  // right_indices will be JoinNoneValue, i.e. -1. This if path should
  // produce exactly the same result as the else path but will be faster.
  if (left_table_row_count == 0) {
    thrust::sequence(rmm::exec_policy(stream),
                     right_indices_complement->begin(),
                     right_indices_complement->end(),
                     0);
  } else {
    // Assume all the indices in invalid_index_map are invalid
    auto invalid_index_map =
      std::make_unique<rmm::device_uvector<size_type>>(right_table_row_count, stream);
    thrust::uninitialized_fill(
      rmm::exec_policy(stream), invalid_index_map->begin(), invalid_index_map->end(), int32_t{1});

    // Functor to check for index validity since left joins can create invalid indices
    valid_range<size_type> valid(0, right_table_row_count);

    // invalid_index_map[index_ptr[i]] = 0 for i = 0 to right_table_row_count
    // Thus specifying that those locations are valid
    thrust::scatter_if(rmm::exec_policy(stream),
                       thrust::make_constant_iterator(0),
                       thrust::make_constant_iterator(0) + right_indices->size(),
                       right_indices->begin(),      // Index locations
                       right_indices->begin(),      // Stencil - Check if index location is valid
                       invalid_index_map->begin(),  // Output indices
                       valid);                      // Stencil Predicate
    size_type begin_counter = static_cast<size_type>(0);
    size_type end_counter   = static_cast<size_type>(right_table_row_count);

    // Create list of indices that have been marked as invalid
    size_type indices_count = thrust::copy_if(rmm::exec_policy(stream),
                                              thrust::make_counting_iterator(begin_counter),
                                              thrust::make_counting_iterator(end_counter),
                                              invalid_index_map->begin(),
                                              right_indices_complement->begin(),
                                              thrust::identity<size_type>()) -
                              right_indices_complement->begin();
    right_indices_complement->resize(indices_count, stream);
  }

  auto left_invalid_indices =
    std::make_unique<rmm::device_uvector<size_type>>(right_indices_complement->size(), stream);
  thrust::fill(rmm::exec_policy(stream),
               left_invalid_indices->begin(),
               left_invalid_indices->end(),
               JoinNoneValue);

  return std::make_pair(std::move(left_invalid_indices), std::move(right_indices_complement));
}

/**
 * @brief Builds the hash table based on the given `build_table`.
 *
 * @throw cudf::logic_error if the number of columns in `build` table is 0.
 * @throw cudf::logic_error if the number of rows in `build` table is 0.
 * @throw cudf::logic_error if insertion to the hash table fails.
 *
 * @param build Table of columns used to build join hash.
 * @param compare_nulls Controls whether null join-key values should match or not.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 *
 * @return Built hash table.
 */
std::unique_ptr<multimap_type, std::function<void(multimap_type *)>> build_join_hash_table(
  cudf::table_view const &build, null_equality compare_nulls, rmm::cuda_stream_view stream)
{
  auto build_device_table = cudf::table_device_view::create(build, stream);

  CUDF_EXPECTS(0 != build_device_table->num_columns(), "Selected build dataset is empty");
  CUDF_EXPECTS(0 != build_device_table->num_rows(), "Build side table has no rows");

  size_type const build_table_num_rows{build_device_table->num_rows()};
  std::size_t const hash_table_size = compute_hash_table_size(build_table_num_rows);

  auto hash_table = multimap_type::create(hash_table_size,
                                          stream,
                                          true,
                                          multimap_type::hasher(),
                                          multimap_type::key_equal(),
                                          multimap_type::allocator_type());

  row_hash hash_build{*build_device_table};
  rmm::device_scalar<int> failure(0, stream);
  constexpr int block_size{DEFAULT_JOIN_BLOCK_SIZE};
  detail::grid_1d config(build_table_num_rows, block_size);
  auto const row_bitmask = (compare_nulls == null_equality::EQUAL)
                             ? rmm::device_buffer{0, stream}
                             : cudf::detail::bitmask_and(build, stream);
  build_hash_table<<<config.num_blocks, config.num_threads_per_block, 0, stream.value()>>>(
    *hash_table,
    hash_build,
    build_table_num_rows,
    static_cast<bitmask_type const *>(row_bitmask.data()),
    failure.data());
  // Check error code from the kernel
  if (failure.value(stream) == 1) { CUDF_FAIL("Hash Table insert failure."); }

  return hash_table;
}

/**
 * @brief Probes the `hash_table` built from `build_table` for tuples in `probe_table`,
 * and returns the output indices of `build_table` and `probe_table` as a combined table.
 * Behavior is undefined if the provided `output_size` is smaller than the actual output size.
 *
 * @tparam JoinKind The type of join to be performed.
 *
 * @param build_table Table of build side columns to join.
 * @param probe_table Table of probe side columns to join.
 * @param hash_table Hash table built from `build_table`.
 * @param compare_nulls Controls whether null join-key values should match or not.
 * @param output_size Optional value which allows users to specify the exact output size.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned vectors.
 *
 * @return Join output indices vector pair.
 */
template <join_kind JoinKind>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
probe_join_hash_table(cudf::table_device_view build_table,
                      cudf::table_device_view probe_table,
                      multimap_type const &hash_table,
                      null_equality compare_nulls,
                      std::optional<std::size_t> output_size,
                      rmm::cuda_stream_view stream,
                      rmm::mr::device_memory_resource *mr)
{
  // Use the output size directly if provided. Otherwise, compute the exact output size
  constexpr cudf::detail::join_kind ProbeJoinKind = (JoinKind == cudf::detail::join_kind::FULL_JOIN)
                                                      ? cudf::detail::join_kind::LEFT_JOIN
                                                      : JoinKind;
  std::size_t const join_size = output_size.value_or(compute_join_output_size<ProbeJoinKind>(
    build_table, probe_table, hash_table, compare_nulls, stream));

  // If output size is zero, return immediately
  if (join_size == 0) {
    return std::make_pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                          std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
  }

  rmm::device_scalar<size_type> write_index(0, stream);

  auto left_indices  = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);
  auto right_indices = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);

  constexpr int block_size{DEFAULT_JOIN_BLOCK_SIZE};
  detail::grid_1d config(probe_table.num_rows(), block_size);

  row_hash hash_probe{probe_table};
  row_equality equality{probe_table, build_table, compare_nulls == null_equality::EQUAL};
  if constexpr (JoinKind == cudf::detail::join_kind::FULL_JOIN) {
    probe_hash_table<cudf::detail::join_kind::LEFT_JOIN,
                     multimap_type,
                     block_size,
                     DEFAULT_JOIN_CACHE_SIZE>
      <<<config.num_blocks, config.num_threads_per_block, 0, stream.value()>>>(
        hash_table,
        build_table,
        probe_table,
        hash_probe,
        equality,
        left_indices->data(),
        right_indices->data(),
        write_index.data(),
        join_size);
    auto const actual_size = write_index.value(stream);
    left_indices->resize(actual_size, stream);
    right_indices->resize(actual_size, stream);
  } else {
    probe_hash_table<JoinKind, multimap_type, block_size, DEFAULT_JOIN_CACHE_SIZE>
      <<<config.num_blocks, config.num_threads_per_block, 0, stream.value()>>>(
        hash_table,
        build_table,
        probe_table,
        hash_probe,
        equality,
        left_indices->data(),
        right_indices->data(),
        write_index.data(),
        join_size);
  }
  return std::make_pair(std::move(left_indices), std::move(right_indices));
}

/**
 * @brief Probes the `hash_table` built from `build_table` for tuples in `probe_table` twice,
 * and returns the output size of a full join operation between `build_table` and `probe_table`.
 * TODO: this is a temporary solution as part of `full_join_size`. To be refactored during
 * cuco integration.
 *
 * @param build_table Table of build side columns to join.
 * @param probe_table Table of probe side columns to join.
 * @param hash_table Hash table built from `build_table`.
 * @param compare_nulls Controls whether null join-key values should match or not.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the intermediate vectors.
 *
 * @return Output size of full join.
 */
std::size_t get_full_join_size(cudf::table_device_view build_table,
                               cudf::table_device_view probe_table,
                               multimap_type const &hash_table,
                               null_equality compare_nulls,
                               rmm::cuda_stream_view stream,
                               rmm::mr::device_memory_resource *mr)
{
  std::size_t join_size = compute_join_output_size<cudf::detail::join_kind::LEFT_JOIN>(
    build_table, probe_table, hash_table, compare_nulls, stream);

  // If output size is zero, return immediately
  if (join_size == 0) { return join_size; }

  rmm::device_scalar<size_type> write_index(0, stream);

  auto left_indices  = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);
  auto right_indices = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);

  constexpr int block_size{DEFAULT_JOIN_BLOCK_SIZE};
  detail::grid_1d config(probe_table.num_rows(), block_size);

  row_hash hash_probe{probe_table};
  row_equality equality{probe_table, build_table, compare_nulls == null_equality::EQUAL};
  probe_hash_table<cudf::detail::join_kind::LEFT_JOIN,
                   multimap_type,
                   block_size,
                   DEFAULT_JOIN_CACHE_SIZE>
    <<<config.num_blocks, config.num_threads_per_block, 0, stream.value()>>>(hash_table,
                                                                             build_table,
                                                                             probe_table,
                                                                             hash_probe,
                                                                             equality,
                                                                             left_indices->data(),
                                                                             right_indices->data(),
                                                                             write_index.data(),
                                                                             join_size);
  // Rlease intermediate memory alloation
  left_indices->resize(0, stream);

  auto const left_table_row_count  = probe_table.num_rows();
  auto const right_table_row_count = build_table.num_rows();

  std::size_t left_join_complement_size;

  // If left table is empty then all rows of the right table should be represented in the joined
  // indices.
  if (left_table_row_count == 0) {
    left_join_complement_size = right_table_row_count;
  } else {
    // Assume all the indices in invalid_index_map are invalid
    auto invalid_index_map =
      std::make_unique<rmm::device_uvector<size_type>>(right_table_row_count, stream);
    thrust::uninitialized_fill(
      rmm::exec_policy(stream), invalid_index_map->begin(), invalid_index_map->end(), int32_t{1});

    // Functor to check for index validity since left joins can create invalid indices
    valid_range<size_type> valid(0, right_table_row_count);

    // invalid_index_map[index_ptr[i]] = 0 for i = 0 to right_table_row_count
    // Thus specifying that those locations are valid
    thrust::scatter_if(rmm::exec_policy(stream),
                       thrust::make_constant_iterator(0),
                       thrust::make_constant_iterator(0) + right_indices->size(),
                       right_indices->begin(),      // Index locations
                       right_indices->begin(),      // Stencil - Check if index location is valid
                       invalid_index_map->begin(),  // Output indices
                       valid);                      // Stencil Predicate

    // Create list of indices that have been marked as invalid
    left_join_complement_size = thrust::count_if(rmm::exec_policy(stream),
                                                 invalid_index_map->begin(),
                                                 invalid_index_map->end(),
                                                 thrust::identity<size_type>());
  }
  return join_size + left_join_complement_size;
}

std::unique_ptr<cudf::table> combine_table_pair(std::unique_ptr<cudf::table> &&left,
                                                std::unique_ptr<cudf::table> &&right)
{
  auto joined_cols = left->release();
  auto right_cols  = right->release();
  joined_cols.insert(joined_cols.end(),
                     std::make_move_iterator(right_cols.begin()),
                     std::make_move_iterator(right_cols.end()));
  return std::make_unique<cudf::table>(std::move(joined_cols));
}

}  // namespace detail

hash_join::hash_join_impl::~hash_join_impl() = default;

hash_join::hash_join_impl::hash_join_impl(cudf::table_view const &build,
                                          null_equality compare_nulls,
                                          rmm::cuda_stream_view stream)
  : _hash_table(nullptr)
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(0 != build.num_columns(), "Hash join build table is empty");
  CUDF_EXPECTS(build.num_rows() < cudf::detail::MAX_JOIN_SIZE,
               "Build column size is too big for hash join");

  auto flattened_build = structs::detail::flatten_nested_columns(
    build, {}, {}, structs::detail::column_nullability::FORCE);
  _build = std::get<0>(flattened_build);
  // need to store off the owning structures for some of the views in _build
  _created_null_columns = std::move(std::get<3>(flattened_build));

  if (0 == build.num_rows()) { return; }

  _hash_table = build_join_hash_table(_build, compare_nulls, stream);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::hash_join_impl::inner_join(cudf::table_view const &probe,
                                      null_equality compare_nulls,
                                      std::optional<std::size_t> output_size,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource *mr) const
{
  CUDF_FUNC_RANGE();
  return compute_hash_join<cudf::detail::join_kind::INNER_JOIN>(
    probe, compare_nulls, output_size, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::hash_join_impl::left_join(cudf::table_view const &probe,
                                     null_equality compare_nulls,
                                     std::optional<std::size_t> output_size,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource *mr) const
{
  CUDF_FUNC_RANGE();
  return compute_hash_join<cudf::detail::join_kind::LEFT_JOIN>(
    probe, compare_nulls, output_size, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::hash_join_impl::full_join(cudf::table_view const &probe,
                                     null_equality compare_nulls,
                                     std::optional<std::size_t> output_size,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource *mr) const
{
  CUDF_FUNC_RANGE();
  return compute_hash_join<cudf::detail::join_kind::FULL_JOIN>(
    probe, compare_nulls, output_size, stream, mr);
}

std::size_t hash_join::hash_join_impl::inner_join_size(cudf::table_view const &probe,
                                                       null_equality compare_nulls,
                                                       rmm::cuda_stream_view stream) const
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(_hash_table, "Hash table of hash join is null.");

  auto build_table = cudf::table_device_view::create(_build, stream);
  auto probe_table = cudf::table_device_view::create(probe, stream);

  return cudf::detail::compute_join_output_size<cudf::detail::join_kind::INNER_JOIN>(
    *build_table, *probe_table, *_hash_table, compare_nulls, stream);
}

std::size_t hash_join::hash_join_impl::left_join_size(cudf::table_view const &probe,
                                                      null_equality compare_nulls,
                                                      rmm::cuda_stream_view stream) const
{
  CUDF_FUNC_RANGE();

  // Trivial left join case - exit early
  if (!_hash_table) { return probe.num_rows(); }

  auto build_table = cudf::table_device_view::create(_build, stream);
  auto probe_table = cudf::table_device_view::create(probe, stream);

  return cudf::detail::compute_join_output_size<cudf::detail::join_kind::LEFT_JOIN>(
    *build_table, *probe_table, *_hash_table, compare_nulls, stream);
}

std::size_t hash_join::hash_join_impl::full_join_size(cudf::table_view const &probe,
                                                      null_equality compare_nulls,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::mr::device_memory_resource *mr) const
{
  CUDF_FUNC_RANGE();

  // Trivial left join case - exit early
  if (!_hash_table) { return probe.num_rows(); }

  auto build_table = cudf::table_device_view::create(_build, stream);
  auto probe_table = cudf::table_device_view::create(probe, stream);

  return get_full_join_size(*build_table, *probe_table, *_hash_table, compare_nulls, stream, mr);
}

template <cudf::detail::join_kind JoinKind>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::hash_join_impl::compute_hash_join(cudf::table_view const &probe,
                                             null_equality compare_nulls,
                                             std::optional<std::size_t> output_size,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource *mr) const
{
  CUDF_EXPECTS(0 != probe.num_columns(), "Hash join probe table is empty");
  CUDF_EXPECTS(probe.num_rows() < cudf::detail::MAX_JOIN_SIZE,
               "Probe column size is too big for hash join");

  auto flattened_probe = structs::detail::flatten_nested_columns(
    probe, {}, {}, structs::detail::column_nullability::FORCE);
  auto const flattened_probe_table = std::get<0>(flattened_probe);

  CUDF_EXPECTS(_build.num_columns() == flattened_probe_table.num_columns(),
               "Mismatch in number of columns to be joined on");

  if (is_trivial_join(flattened_probe_table, _build, JoinKind)) {
    return std::make_pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                          std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
  }

  CUDF_EXPECTS(std::equal(std::cbegin(_build),
                          std::cend(_build),
                          std::cbegin(flattened_probe_table),
                          std::cend(flattened_probe_table),
                          [](const auto &b, const auto &p) { return b.type() == p.type(); }),
               "Mismatch in joining column data types");

  return probe_join_indices<JoinKind>(
    flattened_probe_table, compare_nulls, output_size, stream, mr);
}

template <cudf::detail::join_kind JoinKind>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::hash_join_impl::probe_join_indices(cudf::table_view const &probe,
                                              null_equality compare_nulls,
                                              std::optional<std::size_t> output_size,
                                              rmm::cuda_stream_view stream,
                                              rmm::mr::device_memory_resource *mr) const
{
  // Trivial left join case - exit early
  if (!_hash_table && JoinKind != cudf::detail::join_kind::INNER_JOIN) {
    return get_trivial_left_join_indices(probe, stream, mr);
  }

  CUDF_EXPECTS(_hash_table, "Hash table of hash join is null.");

  auto build_table = cudf::table_device_view::create(_build, stream);
  auto probe_table = cudf::table_device_view::create(probe, stream);

  auto join_indices = cudf::detail::probe_join_hash_table<JoinKind>(
    *build_table, *probe_table, *_hash_table, compare_nulls, output_size, stream, mr);

  if (JoinKind == cudf::detail::join_kind::FULL_JOIN) {
    auto complement_indices = detail::get_left_join_indices_complement(
      join_indices.second, probe.num_rows(), _build.num_rows(), stream, mr);
    join_indices = detail::concatenate_vector_pairs(join_indices, complement_indices, stream);
  }
  return join_indices;
}

}  // namespace cudf
