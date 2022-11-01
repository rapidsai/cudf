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
#include "join_common_utils.cuh"

#include <cudf/copying.hpp>
#include <cudf/detail/concatenate.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/join.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/join.hpp>

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
#include <thrust/tuple.h>
#include <thrust/uninitialized_fill.h>

#include <cstddef>
#include <iostream>
#include <numeric>

namespace cudf {
namespace detail {
namespace {
/**
 * @brief Calculates the exact size of the join output produced when
 * joining two tables together.
 *
 * @throw cudf::logic_error if JoinKind is not INNER_JOIN or LEFT_JOIN
 *
 * @tparam JoinKind The type of join to be performed
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
template <join_kind JoinKind>
std::size_t compute_join_output_size(table_device_view build_table,
                                     table_device_view probe_table,
                                     cudf::detail::multimap_type const& hash_table,
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
                      cudf::detail::multimap_type const& hash_table,
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
                               cudf::detail::multimap_type const& hash_table,
                               bool const has_nulls,
                               null_equality const compare_nulls,
                               rmm::cuda_stream_view stream,
                               rmm::mr::device_memory_resource* mr)
{
  std::size_t join_size = compute_join_output_size<cudf::detail::join_kind::LEFT_JOIN>(
    build_table, probe_table, hash_table, has_nulls, compare_nulls, stream);

  // If output size is zero, return immediately
  if (join_size == 0) { return join_size; }

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
}  // namespace

template <typename Hasher>
hash_join<Hasher>::hash_join(cudf::table_view const& build,
                             cudf::null_equality compare_nulls,
                             rmm::cuda_stream_view stream)
  : _is_empty{build.num_rows() == 0},
    _composite_bitmask{cudf::detail::bitmask_and(build, stream).first},
    _nulls_equal{compare_nulls},
    _hash_table{compute_hash_table_size(build.num_rows()),
                cuco::sentinel::empty_key{std::numeric_limits<hash_value_type>::max()},
                cuco::sentinel::empty_value{cudf::detail::JoinNoneValue},
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

  cudf::detail::build_join_hash_table(_build,
                                      _hash_table,
                                      _nulls_equal,
                                      static_cast<bitmask_type const*>(_composite_bitmask.data()),
                                      stream);
}

template <typename Hasher>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<Hasher>::inner_join(cudf::table_view const& probe,
                              std::optional<std::size_t> output_size,
                              rmm::cuda_stream_view stream,
                              rmm::mr::device_memory_resource* mr) const
{
  CUDF_FUNC_RANGE();
  return compute_hash_join<cudf::detail::join_kind::INNER_JOIN>(probe, output_size, stream, mr);
}

template <typename Hasher>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<Hasher>::left_join(cudf::table_view const& probe,
                             std::optional<std::size_t> output_size,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr) const
{
  CUDF_FUNC_RANGE();
  return compute_hash_join<cudf::detail::join_kind::LEFT_JOIN>(probe, output_size, stream, mr);
}

template <typename Hasher>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<Hasher>::full_join(cudf::table_view const& probe,
                             std::optional<std::size_t> output_size,
                             rmm::cuda_stream_view stream,
                             rmm::mr::device_memory_resource* mr) const
{
  CUDF_FUNC_RANGE();
  return compute_hash_join<cudf::detail::join_kind::FULL_JOIN>(probe, output_size, stream, mr);
}

template <typename Hasher>
std::size_t hash_join<Hasher>::inner_join_size(cudf::table_view const& probe,
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

template <typename Hasher>
std::size_t hash_join<Hasher>::left_join_size(cudf::table_view const& probe,
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

template <typename Hasher>
std::size_t hash_join<Hasher>::full_join_size(cudf::table_view const& probe,
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

template <typename Hasher>
template <cudf::detail::join_kind JoinKind>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<Hasher>::probe_join_indices(cudf::table_view const& probe_table,
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

template <typename Hasher>
template <cudf::detail::join_kind JoinKind>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<Hasher>::compute_hash_join(cudf::table_view const& probe,
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
}  // namespace detail

hash_join::~hash_join() = default;

hash_join::hash_join(cudf::table_view const& build,
                     null_equality compare_nulls,
                     rmm::cuda_stream_view stream)
  : _impl{std::make_unique<const impl_type>(build, compare_nulls, stream)}
{
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::inner_join(cudf::table_view const& probe,
                      std::optional<std::size_t> output_size,
                      rmm::cuda_stream_view stream,
                      rmm::mr::device_memory_resource* mr) const
{
  return _impl->inner_join(probe, output_size, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::left_join(cudf::table_view const& probe,
                     std::optional<std::size_t> output_size,
                     rmm::cuda_stream_view stream,
                     rmm::mr::device_memory_resource* mr) const
{
  return _impl->left_join(probe, output_size, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::full_join(cudf::table_view const& probe,
                     std::optional<std::size_t> output_size,
                     rmm::cuda_stream_view stream,
                     rmm::mr::device_memory_resource* mr) const
{
  return _impl->full_join(probe, output_size, stream, mr);
}

std::size_t hash_join::inner_join_size(cudf::table_view const& probe,
                                       rmm::cuda_stream_view stream) const
{
  return _impl->inner_join_size(probe, stream);
}

std::size_t hash_join::left_join_size(cudf::table_view const& probe,
                                      rmm::cuda_stream_view stream) const
{
  return _impl->left_join_size(probe, stream);
}

std::size_t hash_join::full_join_size(cudf::table_view const& probe,
                                      rmm::cuda_stream_view stream,
                                      rmm::mr::device_memory_resource* mr) const
{
  return _impl->full_join_size(probe, stream, mr);
}

}  // namespace cudf
