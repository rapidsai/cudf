/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "hash_join_helpers.cuh"

#include <cudf/copying.hpp>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/fill.h>
#include <thrust/iterator/transform_output_iterator.h>

#include <memory>

namespace cudf {
namespace detail {
namespace {

bool is_trivial_join(table_view const& left, table_view const& right, join_kind join_type)
{
  if (left.is_empty() || right.is_empty()) { return true; }
  if ((join_kind::LEFT_JOIN == join_type) && (0 == left.num_rows())) { return true; }
  if ((join_kind::INNER_JOIN == join_type) && ((0 == left.num_rows()) || (0 == right.num_rows()))) {
    return true;
  }
  if ((join_kind::LEFT_SEMI_JOIN == join_type) && (0 == right.num_rows())) { return true; }
  if ((join_kind::LEFT_SEMI_JOIN == join_type || join_kind::LEFT_ANTI_JOIN == join_type) &&
      (0 == left.num_rows())) {
    return true;
  }
  return false;
}

}  // namespace

template <typename Hasher>
hash_join<Hasher>::hash_join(cudf::table_view const& build,
                             bool has_nulls,
                             cudf::null_equality compare_nulls,
                             double load_factor,
                             rmm::cuda_stream_view stream)
  : _has_nulls(has_nulls),
    _is_empty{build.num_rows() == 0},
    _nulls_equal{compare_nulls},
    _hash_table{
      cuco::extent{static_cast<size_t>(build.num_rows())},
      load_factor,
      cuco::empty_key{cuco::pair{std::numeric_limits<hash_value_type>::max(), cudf::JoinNoMatch}},
      {},
      {},
      {},
      {},
      rmm::mr::polymorphic_allocator<char>{},
      stream.value()},
    _build{build},
    _preprocessed_build{cudf::detail::row::equality::preprocessed_table::create(_build, stream)}
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(0 != build.num_columns(), "Hash join build table is empty", std::invalid_argument);
  CUDF_EXPECTS(load_factor > 0 && load_factor <= 1,
               "Invalid load factor: must be greater than 0 and less than or equal to 1.",
               std::invalid_argument);

  if (_is_empty) { return; }

  auto const row_bitmask =
    cudf::detail::bitmask_and(build, stream, cudf::get_current_device_resource_ref()).first;
  cudf::detail::build_hash_join(_build,
                                _preprocessed_build,
                                _hash_table,
                                _has_nulls,
                                _nulls_equal,
                                reinterpret_cast<bitmask_type const*>(row_bitmask.data()),
                                stream);
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
  return compute_hash_join(probe, join_kind::INNER_JOIN, output_size, stream, mr);
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
  return compute_hash_join(probe, join_kind::LEFT_JOIN, output_size, stream, mr);
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
  return compute_hash_join(probe, join_kind::FULL_JOIN, output_size, stream, mr);
}

template <typename Hasher>
std::size_t hash_join<Hasher>::inner_join_size(cudf::table_view const& probe,
                                               rmm::cuda_stream_view stream) const
{
  CUDF_FUNC_RANGE();

  if (_is_empty) { return 0; }

  CUDF_EXPECTS(_has_nulls || !cudf::has_nested_nulls(probe),
               "Probe table has nulls while build table was not hashed with null check.",
               std::invalid_argument);

  auto const preprocessed_probe =
    cudf::detail::row::equality::preprocessed_table::create(probe, stream);

  return cudf::detail::compute_join_output_size(_build,
                                                probe,
                                                _preprocessed_build,
                                                preprocessed_probe,
                                                _hash_table,
                                                join_kind::INNER_JOIN,
                                                _has_nulls,
                                                _nulls_equal,
                                                stream);
}

template <typename Hasher>
std::size_t hash_join<Hasher>::left_join_size(cudf::table_view const& probe,
                                              rmm::cuda_stream_view stream) const
{
  CUDF_FUNC_RANGE();

  if (_is_empty) { return probe.num_rows(); }

  CUDF_EXPECTS(_has_nulls || !cudf::has_nested_nulls(probe),
               "Probe table has nulls while build table was not hashed with null check.",
               std::invalid_argument);

  auto const preprocessed_probe =
    cudf::detail::row::equality::preprocessed_table::create(probe, stream);

  return cudf::detail::compute_join_output_size(_build,
                                                probe,
                                                _preprocessed_build,
                                                preprocessed_probe,
                                                _hash_table,
                                                join_kind::LEFT_JOIN,
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

  if (_is_empty) { return probe.num_rows(); }

  CUDF_EXPECTS(_has_nulls || !cudf::has_nested_nulls(probe),
               "Probe table has nulls while build table was not hashed with null check.",
               std::invalid_argument);

  auto const preprocessed_probe =
    cudf::detail::row::equality::preprocessed_table::create(probe, stream);

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
template <typename OutputIterator>
void hash_join<Hasher>::compute_match_counts(cudf::table_view const& probe,
                                             OutputIterator output_iter,
                                             rmm::cuda_stream_view stream) const
{
  CUDF_EXPECTS(_has_nulls || !cudf::has_nested_nulls(probe),
               "Probe table has nulls while build table was not hashed with null check.",
               std::invalid_argument);

  auto const preprocessed_probe =
    cudf::detail::row::equality::preprocessed_table::create(probe, stream);
  auto const probe_nulls          = cudf::nullate::DYNAMIC{_has_nulls};
  auto const probe_table_num_rows = probe.num_rows();

  auto compute_counts = [&](auto equality, auto d_hasher) {
    auto const iter = cudf::detail::make_counting_transform_iterator(0, pair_fn{d_hasher});
    _hash_table.count_each(iter,
                           iter + probe_table_num_rows,
                           equality,
                           _hash_table.hash_function(),
                           output_iter,
                           stream.value());
  };

  if (cudf::detail::is_primitive_row_op_compatible(_build)) {
    auto const d_hasher = cudf::detail::row::primitive::row_hasher{probe_nulls, preprocessed_probe};
    auto const d_equal  = cudf::detail::row::primitive::row_equality_comparator{
      probe_nulls, preprocessed_probe, _preprocessed_build, _nulls_equal};
    compute_counts(primitive_pair_equal{d_equal}, d_hasher);
  } else {
    auto const d_hasher =
      cudf::detail::row::hash::row_hasher{preprocessed_probe}.device_hasher(probe_nulls);
    auto const row_comparator =
      cudf::detail::row::equality::two_table_comparator{preprocessed_probe, _preprocessed_build};
    auto const d_equal = row_comparator.equal_to<false>(probe_nulls, _nulls_equal);
    compute_counts(pair_equal{d_equal}, d_hasher);
  }
}

template <typename Hasher>
cudf::join_match_context hash_join<Hasher>::inner_join_match_context(
  cudf::table_view const& probe,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  cudf::scoped_range range{"hash_join::inner_join_match_context"};

  auto match_counts =
    std::make_unique<rmm::device_uvector<size_type>>(probe.num_rows(), stream, mr);

  if (_is_empty) {
    thrust::fill(rmm::exec_policy_nosync(stream), match_counts->begin(), match_counts->end(), 0);
  } else {
    compute_match_counts(probe, match_counts->begin(), stream);
  }

  return cudf::join_match_context{probe, std::move(match_counts)};
}

template <typename Hasher>
cudf::join_match_context hash_join<Hasher>::left_join_match_context(
  cudf::table_view const& probe,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  cudf::scoped_range range{"hash_join::left_join_match_context"};

  auto match_counts =
    std::make_unique<rmm::device_uvector<size_type>>(probe.num_rows(), stream, mr);

  if (_is_empty) {
    thrust::fill(rmm::exec_policy_nosync(stream), match_counts->begin(), match_counts->end(), 1);
  } else {
    auto transform = [] __device__(size_type count) { return count == 0 ? 1 : count; };
    auto transformed_output =
      thrust::make_transform_output_iterator(match_counts->begin(), transform);
    compute_match_counts(probe, transformed_output, stream);
  }

  return cudf::join_match_context{probe, std::move(match_counts)};
}

template <typename Hasher>
cudf::join_match_context hash_join<Hasher>::full_join_match_context(
  cudf::table_view const& probe,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  cudf::scoped_range range{"hash_join::full_join_match_context"};

  auto match_counts =
    std::make_unique<rmm::device_uvector<size_type>>(probe.num_rows(), stream, mr);

  if (_is_empty) {
    thrust::fill(rmm::exec_policy_nosync(stream), match_counts->begin(), match_counts->end(), 1);
  } else {
    auto transform = [] __device__(size_type count) { return count == 0 ? 1 : count; };
    auto transformed_output =
      thrust::make_transform_output_iterator(match_counts->begin(), transform);
    compute_match_counts(probe, transformed_output, stream);
  }

  return cudf::join_match_context{probe, std::move(match_counts)};
}

template <typename Hasher>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<Hasher>::probe_join_indices(cudf::table_view const& probe_table,
                                      cudf::join_kind join,
                                      std::optional<std::size_t> output_size,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr) const
{
  if (_is_empty and join != join_kind::INNER_JOIN) {
    return get_trivial_left_join_indices(probe_table, stream, mr);
  }

  CUDF_EXPECTS(!_is_empty, "Hash table of hash join is null.");

  CUDF_EXPECTS(_has_nulls || !cudf::has_nested_nulls(probe_table),
               "Probe table has nulls while build table was not hashed with null check.",
               std::invalid_argument);

  auto const preprocessed_probe =
    cudf::detail::row::equality::preprocessed_table::create(probe_table, stream);
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

  if (join == join_kind::FULL_JOIN) {
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
                                     cudf::join_kind join,
                                     std::optional<std::size_t> output_size,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) const
{
  CUDF_EXPECTS(0 != probe.num_columns(), "Hash join probe table is empty", std::invalid_argument);

  CUDF_EXPECTS(_build.num_columns() == probe.num_columns(),
               "Mismatch in number of columns to be joined on",
               std::invalid_argument);

  CUDF_EXPECTS(_has_nulls || !cudf::has_nested_nulls(probe),
               "Probe table has nulls while build table was not hashed with null check.",
               std::invalid_argument);

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
  : hash_join(
      build, nullable_join::YES, compare_nulls, cudf::detail::CUCO_DESIRED_LOAD_FACTOR, stream)
{
}

hash_join::hash_join(cudf::table_view const& build,
                     nullable_join has_nulls,
                     null_equality compare_nulls,
                     double load_factor,
                     rmm::cuda_stream_view stream)
  : _impl{std::make_unique<impl_type const>(
      build, has_nulls == nullable_join::YES, compare_nulls, load_factor, stream)}
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

cudf::join_match_context hash_join::inner_join_match_context(
  cudf::table_view const& probe,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  return _impl->inner_join_match_context(probe, stream, mr);
}

cudf::join_match_context hash_join::left_join_match_context(cudf::table_view const& probe,
                                                            rmm::cuda_stream_view stream,
                                                            rmm::device_async_resource_ref mr) const
{
  return _impl->left_join_match_context(probe, stream, mr);
}

cudf::join_match_context hash_join::full_join_match_context(cudf::table_view const& probe,
                                                            rmm::cuda_stream_view stream,
                                                            rmm::device_async_resource_ref mr) const
{
  return _impl->full_join_match_context(probe, stream, mr);
}

}  // namespace cudf
