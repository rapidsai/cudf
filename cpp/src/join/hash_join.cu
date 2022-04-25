/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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
#include <join/hash_join.cuh>

#include <cudf/copying.hpp>
#include <cudf/detail/concatenate.cuh>
#include <cudf/detail/structs/utilities.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/count.h>
#include <thrust/functional.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/scatter.h>
#include <thrust/uninitialized_fill.h>

#include <cstddef>
#include <iostream>
#include <numeric>

namespace cudf {
namespace detail {

std::pair<std::unique_ptr<table>, std::unique_ptr<table>> get_empty_joined_table(
  table_view const& probe, table_view const& build)
{
  std::unique_ptr<table> empty_probe = empty_like(probe);
  std::unique_ptr<table> empty_build = empty_like(build);
  return std::pair(std::move(empty_probe), std::move(empty_build));
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
                      multimap_type const& hash_table,
                      bool has_nulls,
                      null_equality compare_nulls,
                      std::optional<std::size_t> output_size,
                      rmm::cuda_stream_view stream,
                      rmm::mr::device_memory_resource* mr)
{
  // Use the output size directly if provided. Otherwise, compute the exact output size
  constexpr cudf::detail::join_kind ProbeJoinKind = (JoinKind == cudf::detail::join_kind::FULL_JOIN)
                                                      ? cudf::detail::join_kind::LEFT_JOIN
                                                      : JoinKind;

  std::size_t const join_size =
    output_size ? *output_size
                : compute_join_output_size<ProbeJoinKind>(
                    build_table, probe_table, hash_table, has_nulls, compare_nulls, stream);

  // If output size is zero, return immediately
  if (join_size == 0) {
    return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                     std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
  }

  auto left_indices  = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);
  auto right_indices = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);

  auto const probe_nulls = cudf::nullate::DYNAMIC{has_nulls};
  pair_equality equality{probe_table, build_table, probe_nulls, compare_nulls};

  row_hash hash_probe{probe_nulls, probe_table};
  auto const empty_key_sentinel = hash_table.get_empty_key_sentinel();
  make_pair_function pair_func{hash_probe, empty_key_sentinel};

  auto iter = cudf::detail::make_counting_transform_iterator(0, pair_func);

  const cudf::size_type probe_table_num_rows = probe_table.num_rows();

  auto out1_zip_begin = thrust::make_zip_iterator(
    thrust::make_tuple(thrust::make_discard_iterator(), left_indices->begin()));
  auto out2_zip_begin = thrust::make_zip_iterator(
    thrust::make_tuple(thrust::make_discard_iterator(), right_indices->begin()));

  if constexpr (JoinKind == cudf::detail::join_kind::FULL_JOIN or
                JoinKind == cudf::detail::join_kind::LEFT_JOIN) {
    [[maybe_unused]] auto [out1_zip_end, out2_zip_end] = hash_table.pair_retrieve_outer(
      iter, iter + probe_table_num_rows, out1_zip_begin, out2_zip_begin, equality, stream.value());

    if constexpr (JoinKind == cudf::detail::join_kind::FULL_JOIN) {
      auto const actual_size = out1_zip_end - out1_zip_begin;
      left_indices->resize(actual_size, stream);
      right_indices->resize(actual_size, stream);
    }
  } else {
    hash_table.pair_retrieve(
      iter, iter + probe_table_num_rows, out1_zip_begin, out2_zip_begin, equality, stream.value());
  }
  return std::pair(std::move(left_indices), std::move(right_indices));
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
                               multimap_type const& hash_table,
                               bool const has_nulls,
                               null_equality const compare_nulls,
                               rmm::cuda_stream_view stream,
                               rmm::mr::device_memory_resource* mr)
{
  std::size_t join_size = compute_join_output_size<cudf::detail::join_kind::LEFT_JOIN>(
    build_table, probe_table, hash_table, has_nulls, compare_nulls, stream);

  // If output size is zero, return immediately
  if (join_size == 0) { return join_size; }

  rmm::device_scalar<size_type> write_index(0, stream);

  auto left_indices  = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);
  auto right_indices = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);

  auto const probe_nulls = cudf::nullate::DYNAMIC{has_nulls};
  pair_equality equality{probe_table, build_table, probe_nulls, compare_nulls};

  row_hash hash_probe{probe_nulls, probe_table};
  auto const empty_key_sentinel = hash_table.get_empty_key_sentinel();
  make_pair_function pair_func{hash_probe, empty_key_sentinel};

  auto iter = cudf::detail::make_counting_transform_iterator(0, pair_func);

  const cudf::size_type probe_table_num_rows = probe_table.num_rows();

  auto out1_zip_begin = thrust::make_zip_iterator(
    thrust::make_tuple(thrust::make_discard_iterator(), left_indices->begin()));
  auto out2_zip_begin = thrust::make_zip_iterator(
    thrust::make_tuple(thrust::make_discard_iterator(), right_indices->begin()));

  hash_table.pair_retrieve_outer(
    iter, iter + probe_table_num_rows, out1_zip_begin, out2_zip_begin, equality, stream.value());

  // Release intermediate memory allocation
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
                                                 thrust::identity());
  }
  return join_size + left_join_complement_size;
}

std::unique_ptr<cudf::table> combine_table_pair(std::unique_ptr<cudf::table>&& left,
                                                std::unique_ptr<cudf::table>&& right)
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

hash_join::hash_join_impl::hash_join_impl(cudf::table_view const& build,
                                          null_equality compare_nulls,
                                          rmm::cuda_stream_view stream)
  : _is_empty{build.num_rows() == 0},
    _nulls_equal{compare_nulls},
    _hash_table{compute_hash_table_size(build.num_rows()),
                std::numeric_limits<hash_value_type>::max(),
                cudf::detail::JoinNoneValue,
                stream.value(),
                detail::hash_table_allocator_type{default_allocator<char>{}, stream}}
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(0 != build.num_columns(), "Hash join build table is empty");
  CUDF_EXPECTS(build.num_rows() < cudf::detail::MAX_JOIN_SIZE,
               "Build column size is too big for hash join");

  // need to store off the owning structures for some of the views in _build
  _flattened_build_table = structs::detail::flatten_nested_columns(
    build, {}, {}, structs::detail::column_nullability::FORCE);
  _build = _flattened_build_table;

  if (_is_empty) { return; }

  cudf::detail::build_join_hash_table(_build, _hash_table, _nulls_equal, stream);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::hash_join_impl::inner_join(cudf::table_view const& probe,
                                      std::optional<std::size_t> output_size,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr) const
{
  CUDF_FUNC_RANGE();
  return compute_hash_join<cudf::detail::join_kind::INNER_JOIN>(probe, output_size, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::hash_join_impl::left_join(cudf::table_view const& probe,
                                     std::optional<std::size_t> output_size,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
{
  CUDF_FUNC_RANGE();
  return compute_hash_join<cudf::detail::join_kind::LEFT_JOIN>(probe, output_size, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::hash_join_impl::full_join(cudf::table_view const& probe,
                                     std::optional<std::size_t> output_size,
                                     rmm::cuda_stream_view stream,
                                     rmm::mr::device_memory_resource* mr) const
{
  CUDF_FUNC_RANGE();
  return compute_hash_join<cudf::detail::join_kind::FULL_JOIN>(probe, output_size, stream, mr);
}

std::size_t hash_join::hash_join_impl::inner_join_size(cudf::table_view const& probe,
                                                       rmm::cuda_stream_view stream) const
{
  CUDF_FUNC_RANGE();

  // Return directly if build table is empty
  if (_is_empty) { return 0; }

  auto flattened_probe = structs::detail::flatten_nested_columns(
    probe, {}, {}, structs::detail::column_nullability::FORCE);
  auto const flattened_probe_table = flattened_probe.flattened_columns();

  auto build_table_ptr           = cudf::table_device_view::create(_build, stream);
  auto flattened_probe_table_ptr = cudf::table_device_view::create(flattened_probe_table, stream);

  return cudf::detail::compute_join_output_size<cudf::detail::join_kind::INNER_JOIN>(
    *build_table_ptr,
    *flattened_probe_table_ptr,
    _hash_table,
    cudf::has_nulls(flattened_probe_table) | cudf::has_nulls(_build),
    _nulls_equal,
    stream);
}

std::size_t hash_join::hash_join_impl::left_join_size(cudf::table_view const& probe,
                                                      rmm::cuda_stream_view stream) const
{
  CUDF_FUNC_RANGE();

  // Trivial left join case - exit early
  if (_is_empty) { return probe.num_rows(); }

  auto flattened_probe = structs::detail::flatten_nested_columns(
    probe, {}, {}, structs::detail::column_nullability::FORCE);
  auto const flattened_probe_table = flattened_probe.flattened_columns();

  auto build_table_ptr           = cudf::table_device_view::create(_build, stream);
  auto flattened_probe_table_ptr = cudf::table_device_view::create(flattened_probe_table, stream);

  return cudf::detail::compute_join_output_size<cudf::detail::join_kind::LEFT_JOIN>(
    *build_table_ptr,
    *flattened_probe_table_ptr,
    _hash_table,
    cudf::has_nulls(flattened_probe_table) | cudf::has_nulls(_build),
    _nulls_equal,
    stream);
}

std::size_t hash_join::hash_join_impl::full_join_size(cudf::table_view const& probe,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::mr::device_memory_resource* mr) const
{
  CUDF_FUNC_RANGE();

  // Trivial left join case - exit early
  if (_is_empty) { return probe.num_rows(); }

  auto flattened_probe = structs::detail::flatten_nested_columns(
    probe, {}, {}, structs::detail::column_nullability::FORCE);
  auto const flattened_probe_table = flattened_probe.flattened_columns();

  auto build_table_ptr           = cudf::table_device_view::create(_build, stream);
  auto flattened_probe_table_ptr = cudf::table_device_view::create(flattened_probe_table, stream);

  return cudf::detail::get_full_join_size(
    *build_table_ptr,
    *flattened_probe_table_ptr,
    _hash_table,
    cudf::has_nulls(flattened_probe_table) | cudf::has_nulls(_build),
    _nulls_equal,
    stream,
    mr);
}

template <cudf::detail::join_kind JoinKind>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::hash_join_impl::compute_hash_join(cudf::table_view const& probe,
                                             std::optional<std::size_t> output_size,
                                             rmm::cuda_stream_view stream,
                                             rmm::mr::device_memory_resource* mr) const
{
  CUDF_EXPECTS(0 != probe.num_columns(), "Hash join probe table is empty");
  CUDF_EXPECTS(probe.num_rows() < cudf::detail::MAX_JOIN_SIZE,
               "Probe column size is too big for hash join");

  auto flattened_probe = structs::detail::flatten_nested_columns(
    probe, {}, {}, structs::detail::column_nullability::FORCE);
  auto const flattened_probe_table = flattened_probe.flattened_columns();

  CUDF_EXPECTS(_build.num_columns() == flattened_probe_table.num_columns(),
               "Mismatch in number of columns to be joined on");

  if (is_trivial_join(flattened_probe_table, _build, JoinKind)) {
    return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                     std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
  }

  CUDF_EXPECTS(std::equal(std::cbegin(_build),
                          std::cend(_build),
                          std::cbegin(flattened_probe_table),
                          std::cend(flattened_probe_table),
                          [](const auto& b, const auto& p) { return b.type() == p.type(); }),
               "Mismatch in joining column data types");

  return probe_join_indices<JoinKind>(flattened_probe_table, output_size, stream, mr);
}

template <cudf::detail::join_kind JoinKind>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::hash_join_impl::probe_join_indices(cudf::table_view const& probe_table,
                                              std::optional<std::size_t> output_size,
                                              rmm::cuda_stream_view stream,
                                              rmm::mr::device_memory_resource* mr) const
{
  // Trivial left join case - exit early
  if (_is_empty and JoinKind != cudf::detail::join_kind::INNER_JOIN) {
    return get_trivial_left_join_indices(probe_table, stream, mr);
  }

  CUDF_EXPECTS(!_is_empty, "Hash table of hash join is null.");

  auto build_table_ptr = cudf::table_device_view::create(_build, stream);
  auto probe_table_ptr = cudf::table_device_view::create(probe_table, stream);

  auto join_indices = cudf::detail::probe_join_hash_table<JoinKind>(
    *build_table_ptr,
    *probe_table_ptr,
    _hash_table,
    cudf::has_nulls(probe_table) | cudf::has_nulls(_build),
    _nulls_equal,
    output_size,
    stream,
    mr);

  if constexpr (JoinKind == cudf::detail::join_kind::FULL_JOIN) {
    auto complement_indices = detail::get_left_join_indices_complement(
      join_indices.second, probe_table.num_rows(), _build.num_rows(), stream, mr);
    join_indices = detail::concatenate_vector_pairs(join_indices, complement_indices, stream);
  }
  return join_indices;
}

}  // namespace cudf
