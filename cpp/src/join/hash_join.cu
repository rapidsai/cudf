/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.
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
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/join/hash_join.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/hashing/detail/helper_functions.cuh>
#include <cudf/join/hash_join.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <thrust/count.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/scatter.h>
#include <thrust/tuple.h>
#include <thrust/uninitialized_fill.h>

#include <cstddef>

namespace cudf {
namespace detail {
namespace {
// TODO: `pair_equal` and `pair_fn` to be moved to common utils during
// mixed-join migration
struct pair_fn {
  using hasher =
    cudf::experimental::row::hash::device_row_hasher<cudf::hashing::detail::default_hash,
                                                     cudf::nullate::DYNAMIC>;
  pair_fn(hasher hash) : _hash{hash} {}

  __device__ cuco::pair<hash_value_type, size_type> operator()(size_type i) const noexcept
  {
    return cuco::pair{_hash(i), i};
  }

 private:
  hasher _hash;
};

template <typename Equal>
class pair_equal {
 public:
  pair_equal(Equal check_row_equality) : _check_row_equality{std::move(check_row_equality)} {}

  __device__ __forceinline__ bool operator()(
    cuco::pair<hash_value_type, size_type> const& lhs,
    cuco::pair<hash_value_type, size_type> const& rhs) const noexcept
  {
    using experimental::row::lhs_index_type;
    using experimental::row::rhs_index_type;

    return lhs.first == rhs.first and
           _check_row_equality(lhs_index_type{lhs.second}, rhs_index_type{rhs.second});
  }

 private:
  Equal _check_row_equality;
};

struct output_fn {
  __device__ constexpr cudf::size_type operator()(
    cuco::pair<hash_value_type, cudf::size_type> const& slot) const
  {
    return slot.second;
  }
};

/**
 * @brief Calculates the exact size of the join output produced when
 * joining two tables together.
 *
 * @throw cudf::logic_error if join is not INNER_JOIN or LEFT_JOIN
 *
 * @param build_table The right hand table
 * @param probe_table The left hand table
 * @param preprocessed_build shared_ptr to cudf::experimental::row::equality::preprocessed_table
 * for build_table
 * @param preprocessed_probe shared_ptr to cudf::experimental::row::equality::preprocessed_table
 * for probe_table
 * @param hash_table A hash table built on the build table that maps the index
 *                   of every row to the hash value of that row
 * @param join The type of join to be performed
 * @param has_nulls Flag to denote if build or probe tables have nested nulls
 * @param nulls_equal Flag to denote nulls are equal or not
 * @param stream CUDA stream used for device memory operations and kernel launches
 *
 * @return The exact size of the output of the join operation
 */
std::size_t compute_join_output_size(
  table_view const& build_table,
  table_view const& probe_table,
  std::shared_ptr<cudf::experimental::row::equality::preprocessed_table> const& preprocessed_build,
  std::shared_ptr<cudf::experimental::row::equality::preprocessed_table> const& preprocessed_probe,
  cudf::detail::multimap_type const& hash_table,
  join_kind join,
  bool has_nulls,
  cudf::null_equality nulls_equal,
  rmm::cuda_stream_view stream)
{
  size_type const build_table_num_rows{build_table.num_rows()};
  size_type const probe_table_num_rows{probe_table.num_rows()};

  // If the build table is empty, we know exactly how large the output
  // will be for the different types of joins and can return immediately
  if (0 == build_table_num_rows) {
    switch (join) {
      // Inner join with an empty table will have no output
      case join_kind::INNER_JOIN: return 0;

      // Left join with an empty table will have an output of NULL rows
      // equal to the number of rows in the probe table
      case join_kind::LEFT_JOIN: return probe_table_num_rows;

      default: CUDF_FAIL("Unsupported join type");
    }
  }

  auto const row_hash = cudf::experimental::row::hash::row_hasher{preprocessed_probe};
  auto const iter     = cudf::detail::make_counting_transform_iterator(
    0, pair_fn{row_hash.device_hasher(cudf::nullate::DYNAMIC{has_nulls})});

  auto const row_comparator =
    cudf::experimental::row::equality::two_table_comparator{preprocessed_probe, preprocessed_build};
  auto const comparator_helper = [&](auto device_comparator) {
    auto equality = pair_equal{device_comparator};

    if (join == join_kind::LEFT_JOIN) {
      return hash_table.count_outer(
        iter, iter + probe_table_num_rows, equality, hash_table.hash_function(), stream.value());
    } else {
      return hash_table.count(
        iter, iter + probe_table_num_rows, equality, hash_table.hash_function(), stream.value());
    }
  };

  if (cudf::detail::has_nested_columns(probe_table)) {
    auto const device_comparator = row_comparator.equal_to<true>(has_nulls, nulls_equal);
    return comparator_helper(device_comparator);
  } else {
    auto const device_comparator = row_comparator.equal_to<false>(has_nulls, nulls_equal);
    return comparator_helper(device_comparator);
  }
}

/**
 * @brief Probes the `hash_table` built from `build_table` for tuples in `probe_table`,
 * and returns the output indices of `build_table` and `probe_table` as a combined table.
 * Behavior is undefined if the provided `output_size` is smaller than the actual output size.
 *
 * @param build_table Table of build side columns to join
 * @param probe_table Table of probe side columns to join
 * @param preprocessed_build shared_ptr to cudf::experimental::row::equality::preprocessed_table
 * for build_table
 * @param preprocessed_probe shared_ptr to cudf::experimental::row::equality::preprocessed_table
 * for probe_table
 * @param hash_table Hash table built from `build_table`
 * @param join The type of join to be performed
 * @param has_nulls Flag to denote if build or probe tables have nested nulls
 * @param compare_nulls Controls whether null join-key values should match or not
 * @param output_size Optional value which allows users to specify the exact output size
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned vectors
 *
 * @return Join output indices vector pair.
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
probe_join_hash_table(
  cudf::table_view const& build_table,
  cudf::table_view const& probe_table,
  std::shared_ptr<cudf::experimental::row::equality::preprocessed_table> const& preprocessed_build,
  std::shared_ptr<cudf::experimental::row::equality::preprocessed_table> const& preprocessed_probe,
  cudf::detail::multimap_type const& hash_table,
  join_kind join,
  bool has_nulls,
  null_equality compare_nulls,
  std::optional<std::size_t> output_size,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  // Use the output size directly if provided. Otherwise, compute the exact output size
  auto const probe_join_type =
    (join == cudf::detail::join_kind::FULL_JOIN) ? cudf::detail::join_kind::LEFT_JOIN : join;

  std::size_t const join_size = output_size ? *output_size
                                            : compute_join_output_size(build_table,
                                                                       probe_table,
                                                                       preprocessed_build,
                                                                       preprocessed_probe,
                                                                       hash_table,
                                                                       probe_join_type,
                                                                       has_nulls,
                                                                       compare_nulls,
                                                                       stream);

  // If output size is zero, return immediately
  if (join_size == 0) {
    return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                     std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
  }

  auto left_indices  = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);
  auto right_indices = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);
  cudf::experimental::prefetch::detail::prefetch("hash_join", *left_indices, stream);
  cudf::experimental::prefetch::detail::prefetch("hash_join", *right_indices, stream);

  auto const probe_nulls = cudf::nullate::DYNAMIC{has_nulls};
  auto const row_hash    = cudf::experimental::row::hash::row_hasher{preprocessed_probe};
  auto const iter =
    cudf::detail::make_counting_transform_iterator(0, pair_fn{row_hash.device_hasher(probe_nulls)});

  cudf::size_type const probe_table_num_rows = probe_table.num_rows();

  auto const out_probe_begin =
    thrust::make_transform_output_iterator(left_indices->begin(), output_fn{});
  auto const out_build_begin =
    thrust::make_transform_output_iterator(right_indices->begin(), output_fn{});

  auto const row_comparator =
    cudf::experimental::row::equality::two_table_comparator{preprocessed_probe, preprocessed_build};
  auto const comparator_helper = [&](auto device_comparator) {
    auto equality = pair_equal{device_comparator};

    if (join == cudf::detail::join_kind::FULL_JOIN or join == cudf::detail::join_kind::LEFT_JOIN) {
      [[maybe_unused]] auto out_probe_end = hash_table
                                              .retrieve_outer(iter,
                                                              iter + probe_table_num_rows,
                                                              equality,
                                                              hash_table.hash_function(),
                                                              out_probe_begin,
                                                              out_build_begin,
                                                              stream.value())
                                              .first;

      if (join == cudf::detail::join_kind::FULL_JOIN) {
        auto const actual_size = cuda::std::distance(out_probe_begin, out_probe_end);
        left_indices->resize(actual_size, stream);
        right_indices->resize(actual_size, stream);
      }
    } else {
      hash_table.retrieve(iter,
                          iter + probe_table_num_rows,
                          equality,
                          hash_table.hash_function(),
                          out_probe_begin,
                          out_build_begin,
                          stream.value());
    }
  };

  if (cudf::detail::has_nested_columns(probe_table)) {
    auto const device_comparator = row_comparator.equal_to<true>(probe_nulls, compare_nulls);
    comparator_helper(device_comparator);
  } else {
    auto const device_comparator = row_comparator.equal_to<false>(probe_nulls, compare_nulls);
    comparator_helper(device_comparator);
  }

  return std::pair(std::move(left_indices), std::move(right_indices));
}

/**
 * @brief Probes the `hash_table` built from `build_table` for tuples in `probe_table` twice,
 * and returns the output size of a full join operation between `build_table` and `probe_table`.
 * TODO: this is a temporary solution as part of `full_join_size`. To be refactored during
 * cuco integration.
 *
 * @param build_table Table of build side columns to join
 * @param probe_table Table of probe side columns to join
 * @param preprocessed_build shared_ptr to cudf::experimental::row::equality::preprocessed_table
 * for build_table
 * @param preprocessed_probe shared_ptr to cudf::experimental::row::equality::preprocessed_table
 * for probe_table
 * @param hash_table Hash table built from `build_table`
 * @param has_nulls Flag to denote if build or probe tables have nested nulls
 * @param compare_nulls Controls whether null join-key values should match or not
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the intermediate vectors
 *
 * @return Output size of full join.
 */
std::size_t get_full_join_size(
  cudf::table_view const& build_table,
  cudf::table_view const& probe_table,
  std::shared_ptr<cudf::experimental::row::equality::preprocessed_table> const& preprocessed_build,
  std::shared_ptr<cudf::experimental::row::equality::preprocessed_table> const& preprocessed_probe,
  cudf::detail::multimap_type const& hash_table,
  bool has_nulls,
  null_equality compare_nulls,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  std::size_t join_size = compute_join_output_size(build_table,
                                                   probe_table,
                                                   preprocessed_build,
                                                   preprocessed_probe,
                                                   hash_table,
                                                   cudf::detail::join_kind::LEFT_JOIN,
                                                   has_nulls,
                                                   compare_nulls,
                                                   stream);

  // If output size is zero, return immediately
  if (join_size == 0) { return join_size; }

  auto right_indices = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);

  auto const probe_nulls = cudf::nullate::DYNAMIC{has_nulls};
  auto const row_hash    = cudf::experimental::row::hash::row_hasher{preprocessed_probe};
  auto const iter =
    cudf::detail::make_counting_transform_iterator(0, pair_fn{row_hash.device_hasher(probe_nulls)});

  cudf::size_type const probe_table_num_rows = probe_table.num_rows();

  auto const out_build_begin =
    thrust::make_transform_output_iterator(right_indices->begin(), output_fn{});

  auto const row_comparator =
    cudf::experimental::row::equality::two_table_comparator{preprocessed_probe, preprocessed_build};
  auto const comparator_helper = [&](auto device_comparator) {
    auto equality = pair_equal{device_comparator};
    hash_table.retrieve_outer(iter,
                              iter + probe_table_num_rows,
                              equality,
                              hash_table.hash_function(),
                              thrust::make_discard_iterator(),
                              out_build_begin,
                              stream.value());
  };
  if (cudf::detail::has_nested_columns(probe_table)) {
    auto const device_comparator = row_comparator.equal_to<true>(probe_nulls, compare_nulls);
    comparator_helper(device_comparator);
  } else {
    auto const device_comparator = row_comparator.equal_to<false>(probe_nulls, compare_nulls);
    comparator_helper(device_comparator);
  }

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
                                                 cuda::std::identity());
  }
  return join_size + left_join_complement_size;
}
}  // namespace

template <typename Hasher>
hash_join<Hasher>::hash_join(cudf::table_view const& build,
                             bool has_nulls,
                             cudf::null_equality compare_nulls,
                             rmm::cuda_stream_view stream)
  : _has_nulls{has_nulls},
    _is_empty{build.num_rows() == 0},
    _nulls_equal{compare_nulls},
    _hash_table{cuco::extent{static_cast<size_t>(build.num_rows())},
                CUCO_DESIRED_LOAD_FACTOR,
                cuco::empty_key{cuco::pair{std::numeric_limits<hash_value_type>::max(),
                                           cudf::detail::JoinNoneValue}},
                {},
                {},
                {},
                {},
                cudf::detail::cuco_allocator<char>{rmm::mr::polymorphic_allocator<char>{}, stream},
                stream.value()},
    _build{build},
    _preprocessed_build{
      cudf::experimental::row::equality::preprocessed_table::create(_build, stream)}
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(0 != build.num_columns(), "Hash join build table is empty");

  if (_is_empty) { return; }

  // TODO: to be replaced by `build_join_hash_table` during mixed-join migration
  auto const row_hash = experimental::row::hash::row_hasher{_preprocessed_build};
  auto const iter     = cudf::detail::make_counting_transform_iterator(
    0, pair_fn{row_hash.device_hasher(nullate::DYNAMIC{_has_nulls})});

  size_type const build_table_num_rows{build.num_rows()};
  if (_nulls_equal == cudf::null_equality::EQUAL or (not nullable(build))) {
    _hash_table.insert_async(iter, iter + build_table_num_rows, stream.value());
  } else {
    auto const row_bitmask =
      cudf::detail::bitmask_and(build, stream, cudf::get_current_device_resource_ref()).first;
    auto const stencil = thrust::counting_iterator<size_type>{0};
    auto const pred    = row_is_valid{reinterpret_cast<bitmask_type const*>(row_bitmask.data())};

    // insert valid rows
    _hash_table.insert_if_async(iter, iter + build_table_num_rows, stencil, pred, stream.value());
  }
}

template <typename Hasher>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<Hasher>::inner_join(cudf::table_view const& probe,
                              std::optional<std::size_t> output_size,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();
  return compute_hash_join(probe, cudf::detail::join_kind::INNER_JOIN, output_size, stream, mr);
}

template <typename Hasher>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<Hasher>::left_join(cudf::table_view const& probe,
                             std::optional<std::size_t> output_size,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();
  return compute_hash_join(probe, cudf::detail::join_kind::LEFT_JOIN, output_size, stream, mr);
}

template <typename Hasher>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<Hasher>::full_join(cudf::table_view const& probe,
                             std::optional<std::size_t> output_size,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();
  return compute_hash_join(probe, cudf::detail::join_kind::FULL_JOIN, output_size, stream, mr);
}

template <typename Hasher>
std::size_t hash_join<Hasher>::inner_join_size(cudf::table_view const& probe,
                                               rmm::cuda_stream_view stream) const
{
  CUDF_FUNC_RANGE();

  // Return directly if build table is empty
  if (_is_empty) { return 0; }

  CUDF_EXPECTS(_has_nulls || !cudf::has_nested_nulls(probe),
               "Probe table has nulls while build table was not hashed with null check.");

  auto const preprocessed_probe =
    cudf::experimental::row::equality::preprocessed_table::create(probe, stream);

  return cudf::detail::compute_join_output_size(_build,
                                                probe,
                                                _preprocessed_build,
                                                preprocessed_probe,
                                                _hash_table,
                                                cudf::detail::join_kind::INNER_JOIN,
                                                _has_nulls,
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

  CUDF_EXPECTS(_has_nulls || !cudf::has_nested_nulls(probe),
               "Probe table has nulls while build table was not hashed with null check.");

  auto const preprocessed_probe =
    cudf::experimental::row::equality::preprocessed_table::create(probe, stream);

  return cudf::detail::compute_join_output_size(_build,
                                                probe,
                                                _preprocessed_build,
                                                preprocessed_probe,
                                                _hash_table,
                                                cudf::detail::join_kind::LEFT_JOIN,
                                                _has_nulls,
                                                _nulls_equal,
                                                stream);
}

template <typename Hasher>
std::size_t hash_join<Hasher>::full_join_size(cudf::table_view const& probe,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();

  // Trivial left join case - exit early
  if (_is_empty) { return probe.num_rows(); }

  CUDF_EXPECTS(_has_nulls || !cudf::has_nested_nulls(probe),
               "Probe table has nulls while build table was not hashed with null check.");

  auto const preprocessed_probe =
    cudf::experimental::row::equality::preprocessed_table::create(probe, stream);

  return cudf::detail::get_full_join_size(_build,
                                          probe,
                                          _preprocessed_build,
                                          preprocessed_probe,
                                          _hash_table,
                                          _has_nulls,
                                          _nulls_equal,
                                          stream,
                                          mr);
}

template <typename Hasher>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<Hasher>::probe_join_indices(cudf::table_view const& probe_table,
                                      cudf::detail::join_kind join,
                                      std::optional<std::size_t> output_size,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr) const
{
  // Trivial left join case - exit early
  if (_is_empty and join != cudf::detail::join_kind::INNER_JOIN) {
    return get_trivial_left_join_indices(probe_table, stream, mr);
  }

  CUDF_EXPECTS(!_is_empty, "Hash table of hash join is null.");

  CUDF_EXPECTS(_has_nulls || !cudf::has_nested_nulls(probe_table),
               "Probe table has nulls while build table was not hashed with null check.");

  auto const preprocessed_probe =
    cudf::experimental::row::equality::preprocessed_table::create(probe_table, stream);
  auto join_indices = cudf::detail::probe_join_hash_table(_build,
                                                          probe_table,
                                                          _preprocessed_build,
                                                          preprocessed_probe,
                                                          _hash_table,
                                                          join,
                                                          _has_nulls,
                                                          _nulls_equal,
                                                          output_size,
                                                          stream,
                                                          mr);

  if (join == cudf::detail::join_kind::FULL_JOIN) {
    auto complement_indices = detail::get_left_join_indices_complement(
      join_indices.second, probe_table.num_rows(), _build.num_rows(), stream, mr);
    join_indices = detail::concatenate_vector_pairs(join_indices, complement_indices, stream);
  }
  return join_indices;
}

template <typename Hasher>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<Hasher>::compute_hash_join(cudf::table_view const& probe,
                                     cudf::detail::join_kind join,
                                     std::optional<std::size_t> output_size,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) const
{
  CUDF_EXPECTS(0 != probe.num_columns(), "Hash join probe table is empty");

  CUDF_EXPECTS(_build.num_columns() == probe.num_columns(),
               "Mismatch in number of columns to be joined on");

  CUDF_EXPECTS(_has_nulls || !cudf::has_nested_nulls(probe),
               "Probe table has nulls while build table was not hashed with null check.");

  if (is_trivial_join(probe, _build, join)) {
    return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                     std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
  }

  CUDF_EXPECTS(cudf::have_same_types(_build, probe),
               "Mismatch in joining column data types",
               cudf::data_type_error);

  return probe_join_indices(probe, join, output_size, stream, mr);
}
}  // namespace detail

hash_join::~hash_join() = default;

hash_join::hash_join(cudf::table_view const& build,
                     null_equality compare_nulls,
                     rmm::cuda_stream_view stream)
  // If we cannot know beforehand about null existence then let's assume that there are nulls.
  : hash_join(build, nullable_join::YES, compare_nulls, stream)
{
}

hash_join::hash_join(cudf::table_view const& build,
                     nullable_join has_nulls,
                     null_equality compare_nulls,
                     rmm::cuda_stream_view stream)
  : _impl{std::make_unique<impl_type const>(
      build, has_nulls == nullable_join::YES, compare_nulls, stream)}
{
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::inner_join(cudf::table_view const& probe,
                      std::optional<std::size_t> output_size,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr) const
{
  return _impl->inner_join(probe, output_size, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::left_join(cudf::table_view const& probe,
                     std::optional<std::size_t> output_size,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr) const
{
  return _impl->left_join(probe, output_size, stream, mr);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join::full_join(cudf::table_view const& probe,
                     std::optional<std::size_t> output_size,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr) const
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
                                      rmm::device_async_resource_ref mr) const
{
  return _impl->full_join_size(probe, stream, mr);
}

}  // namespace cudf
