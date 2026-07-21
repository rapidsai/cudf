/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join_common_utils.cuh"

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/join/distinct_filtered_join.cuh>
#include <cudf/detail/join/filtered_join.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/primitive_row_operators.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>
#include <rmm/resource_ref.hpp>

#include <cuco/bucket_storage.cuh>
#include <cuco/detail/open_addressing/kernels.cuh>
#include <cuco/extent.cuh>
#include <cuco/operator.hpp>
#include <cuco/static_set_ref.cuh>
#include <cuda/iterator>
#include <thrust/sequence.h>

#include <memory>

namespace cudf {
namespace detail {
namespace {

/**
 * @brief Build a row bitmask for the input table.
 *
 * The output bitmask will have invalid bits corresponding to the input rows having nulls (at
 * any nested level) and vice versa.
 *
 * @param input The input table
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return A pair of pointer to the output bitmask and the buffer containing the bitmask
 */
std::pair<rmm::device_buffer, bitmask_type const*> build_row_bitmask(table_view const& input,
                                                                     rmm::cuda_stream_view stream)
{
  auto const nullable_columns = get_nullable_columns(input);
  CUDF_EXPECTS(nullable_columns.size() > 0,
               "The input table has nulls thus it should have nullable columns.");

  // If there are more than one nullable column, we compute `bitmask_and` of their null masks.
  // Otherwise, we have only one nullable column and can use its null mask directly.
  if (nullable_columns.size() > 1) {
    auto row_bitmask =
      cudf::detail::bitmask_and(
        table_view{nullable_columns}, stream, cudf::get_current_device_resource_ref())
        .first;
    auto const row_bitmask_ptr = static_cast<bitmask_type const*>(row_bitmask.data());
    return std::pair(std::move(row_bitmask), row_bitmask_ptr);
  }

  return std::pair(rmm::device_buffer{0, stream}, nullable_columns.front().null_mask());
}

struct gather_mask {
  join_kind kind;
  device_span<bool const> flagged;
  __device__ bool operator()(size_type idx) const noexcept
  {
    return flagged[idx] == (kind == join_kind::LEFT_SEMI_JOIN);
  }
};

}  // namespace

auto filtered_join::compute_bucket_storage_size(cudf::table_view tbl, double load_factor)
{
  auto const size_with_primitive_probe = static_cast<std::size_t>(
    cuco::make_valid_extent<primitive_probing_scheme, storage_type, std::size_t>(tbl.num_rows(),
                                                                                 load_factor));
  auto const size_with_nested_probe = static_cast<std::size_t>(
    cuco::make_valid_extent<nested_probing_scheme, storage_type, std::size_t>(tbl.num_rows(),
                                                                              load_factor));
  auto const size_with_simple_probe = static_cast<std::size_t>(
    cuco::make_valid_extent<simple_probing_scheme, storage_type, std::size_t>(tbl.num_rows(),
                                                                              load_factor));
  return std::max({size_with_primitive_probe, size_with_nested_probe, size_with_simple_probe});
}

template <int32_t CGSize, typename Ref>
void filtered_join::insert_right_table(Ref const& insert_ref, rmm::cuda_stream_view stream)
{
  cudf::scoped_range range{"distinct_filtered_join::insert_right_table"};
  auto insert = [&]<typename Iterator>(Iterator right_iter) {
    // Build hash table by inserting all rows from the right table
    auto const grid_size = cuco::detail::grid_size(_right.num_rows(), CGSize);

    if (cudf::has_nested_nulls(_right) && _nulls_equal == null_equality::UNEQUAL) {
      auto const bitmask_buffer_and_ptr = build_row_bitmask(_right, stream);
      auto const row_bitmask_ptr        = bitmask_buffer_and_ptr.second;
      cuco::detail::open_addressing_ns::insert_if_n<CGSize, cuco::detail::default_block_size()>
        <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
          right_iter,
          _right.num_rows(),
          cuda::counting_iterator<size_type>{0},
          row_is_valid{row_bitmask_ptr},
          insert_ref);
      CUDF_CUDA_TRY(cudaGetLastError());
    } else {
      cuco::detail::open_addressing_ns::insert_if_n<CGSize, cuco::detail::default_block_size()>
        <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
          right_iter,
          _right.num_rows(),
          cuda::constant_iterator<bool>{true},
          cuda::std::identity{},
          insert_ref);
      CUDF_CUDA_TRY(cudaGetLastError());
    }
  };

  // Any mismatch in nullate between left and right row operators results in UB. Ideally, nullate
  // should be determined by the logical OR of left nulls and right nulls. However, since we do not
  // know if the left has nulls apriori, we set nullate::DYNAMIC{true} (in the case of primitive
  // row operators) and nullate::YES (in the case of non-primitive row operators) to ensure both
  // left and right row operators use consistent null handling.
  if (is_primitive_row_op_compatible(_right)) {
    auto const d_right_hasher = primitive_row_hasher{nullate::DYNAMIC{true}, _preprocessed_right};
    auto const right_iter     = cudf::detail::make_counting_transform_iterator(
      size_type{0}, key_pair_fn<lhs_index_type, primitive_row_hasher>{d_right_hasher});

    insert(right_iter);
  } else {
    auto const d_right_hasher =
      cudf::detail::row::hash::row_hasher{_preprocessed_right}.device_hasher(nullate::YES{});
    auto const right_iter = cudf::detail::make_counting_transform_iterator(
      size_type{0}, key_pair_fn<lhs_index_type, row_hasher>{d_right_hasher});

    insert(right_iter);
  }
}

template <int32_t CGSize, typename Ref>
std::unique_ptr<rmm::device_uvector<cudf::size_type>> distinct_filtered_join::query_right_table(
  cudf::table_view const& left,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> preprocessed_left,
  join_kind kind,
  Ref query_ref,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  cudf::scoped_range range{"distinct_filtered_join::query_right_table"};
  auto const left_has_nulls = has_nested_nulls(left);

  auto query_set = [this,
                    left,
                    left_has_nulls,
                    query_ref,
                    stream]<typename InputLeftIterator, typename OutputContainsIterator>(
                     InputLeftIterator left_iter, OutputContainsIterator contains_iter) {
    auto const grid_size = cuco::detail::grid_size(left.num_rows(), CGSize);
    if (left_has_nulls && _nulls_equal == null_equality::UNEQUAL) {
      auto const bitmask_buffer_and_ptr = build_row_bitmask(left, stream);
      auto const row_bitmask_ptr        = bitmask_buffer_and_ptr.second;
      cuco::detail::open_addressing_ns::contains_if_n<CGSize, cuco::detail::default_block_size()>
        <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
          left_iter,
          left.num_rows(),
          cuda::counting_iterator<size_type>{0},
          row_is_valid{row_bitmask_ptr},
          contains_iter,
          query_ref);
      CUDF_CUDA_TRY(cudaGetLastError());
    } else {
      cuco::detail::open_addressing_ns::contains_if_n<CGSize, cuco::detail::default_block_size()>
        <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
          left_iter,
          left.num_rows(),
          cuda::constant_iterator<bool>{true},
          cuda::std::identity{},
          contains_iter,
          query_ref);
      CUDF_CUDA_TRY(cudaGetLastError());
    }
  };

  auto contains_map = rmm::device_uvector<bool>(left.num_rows(), stream);
  if (is_primitive_row_op_compatible(_right)) {
    auto const d_left_hasher = primitive_row_hasher{nullate::DYNAMIC{true}, preprocessed_left};
    auto const left_iter     = cudf::detail::make_counting_transform_iterator(
      size_type{0}, key_pair_fn<rhs_index_type, primitive_row_hasher>{d_left_hasher});

    query_set(left_iter, contains_map.begin());
  } else {
    auto const d_left_hasher =
      cudf::detail::row::hash::row_hasher{preprocessed_left}.device_hasher(nullate::YES{});
    auto const left_iter = cudf::detail::make_counting_transform_iterator(
      size_type{0}, key_pair_fn<rhs_index_type, row_hasher>{d_left_hasher});

    query_set(left_iter, contains_map.begin());
  }
  rmm::device_uvector<size_type> gather_map(left.num_rows(), stream, mr);
  auto gather_map_end =
    thrust::copy_if(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    cuda::counting_iterator<size_type>{0},
                    cuda::counting_iterator<size_type>{left.num_rows()},
                    gather_map.begin(),
                    gather_mask{kind, contains_map});
  gather_map.resize(cuda::std::distance(gather_map.begin(), gather_map_end), stream);
  return std::make_unique<rmm::device_uvector<size_type>>(std::move(gather_map));
}

filtered_join::filtered_join(cudf::table_view const& right,
                             cudf::null_equality compare_nulls,
                             double load_factor,
                             rmm::cuda_stream_view stream,
                             cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : _right_props{right_properties{cudf::has_nested_columns(right)}},
    _nulls_equal{compare_nulls},
    _right{right},
    _preprocessed_right{cudf::detail::row::equality::preprocessed_table::create(_right, stream)},
    _bucket_storage{cuco::extent<std::size_t>{compute_bucket_storage_size(right, load_factor)},
                    rmm::mr::polymorphic_allocator<char>{std::move(mr)},
                    stream.value()}
{
  if (_right.num_rows() == 0) return;
  _bucket_storage.initialize(empty_sentinel_key, stream);
}

distinct_filtered_join::distinct_filtered_join(
  cudf::table_view const& right,
  cudf::null_equality compare_nulls,
  double load_factor,
  rmm::cuda_stream_view stream,
  cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : filtered_join(right, compare_nulls, load_factor, stream, std::move(mr))
{
  cudf::scoped_range range{"distinct_filtered_join::distinct_filtered_join"};
  if (_right.num_rows() == 0) return;
  // Any mismatch in nullate between left and right row operators results in UB. Ideally, nullate
  // should be determined by the logical OR of left nulls and right nulls. However, since we do not
  // know if the left has nulls apriori, we set nullate::DYNAMIC{true} (in the case of primitive
  // row operators) and nullate::YES (in the case of non-primitive row operators) to ensure both
  // left and right row operators use consistent null handling.
  if (is_primitive_row_op_compatible(right)) {
    auto const d_right_comparator = primitive_row_comparator{
      nullate::DYNAMIC{true}, _preprocessed_right, _preprocessed_right, compare_nulls};
    cuco::static_set_ref set_ref{empty_sentinel_key,
                                 insertion_adapter{d_right_comparator},
                                 primitive_probing_scheme{},
                                 cuco::thread_scope_device,
                                 _bucket_storage.ref()};
    auto insert_ref = set_ref.rebind_operators(cuco::insert);
    insert_right_table<primitive_probing_scheme::cg_size>(insert_ref, stream);
  } else if (_right_props.has_nested_columns) {
    auto const d_right_comparator =
      cudf::detail::row::equality::self_comparator{_preprocessed_right}.equal_to<true>(
        nullate::YES{},
        compare_nulls,
        cudf::detail::row::equality::nan_equal_physical_equality_comparator{});
    cuco::static_set_ref set_ref{empty_sentinel_key,
                                 insertion_adapter{d_right_comparator},
                                 nested_probing_scheme{},
                                 cuco::thread_scope_device,
                                 _bucket_storage.ref()};
    auto insert_ref = set_ref.rebind_operators(cuco::insert);
    insert_right_table<nested_probing_scheme::cg_size>(insert_ref, stream);
  } else {
    auto const d_right_comparator =
      cudf::detail::row::equality::self_comparator{_preprocessed_right}.equal_to<false>(
        nullate::YES{},
        compare_nulls,
        cudf::detail::row::equality::nan_equal_physical_equality_comparator{});
    cuco::static_set_ref set_ref{empty_sentinel_key,
                                 insertion_adapter{d_right_comparator},
                                 simple_probing_scheme{},
                                 cuco::thread_scope_device,
                                 _bucket_storage.ref()};
    auto insert_ref = set_ref.rebind_operators(cuco::insert);
    insert_right_table<simple_probing_scheme::cg_size>(insert_ref, stream);
  }
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> distinct_filtered_join::semi_anti_join(
  cudf::table_view const& left,
  join_kind kind,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  cudf::scoped_range range{"distinct_filtered_join::semi_anti_join"};

  auto const preprocessed_left = [&left, stream] {
    cudf::scoped_range range{"distinct_filtered_join::semi_anti_join::preprocessed_left"};
    return cudf::detail::row::equality::preprocessed_table::create(left, stream);
  }();

  if (is_primitive_row_op_compatible(_right)) {
    auto const d_right_left_comparator = primitive_row_comparator{
      nullate::DYNAMIC{true}, _preprocessed_right, preprocessed_left, _nulls_equal};

    cuco::static_set_ref set_ref{empty_sentinel_key,
                                 comparator_adapter{d_right_left_comparator},
                                 primitive_probing_scheme{},
                                 cuco::thread_scope_device,
                                 _bucket_storage.ref()};
    auto query_ref = set_ref.rebind_operators(cuco::op::contains);
    return query_right_table<primitive_probing_scheme::cg_size>(
      left, preprocessed_left, kind, query_ref, stream, mr);
  } else {
    auto const d_right_left_comparator =
      cudf::detail::row::equality::two_table_comparator{_preprocessed_right, preprocessed_left};

    if (_right_props.has_nested_columns) {
      auto d_right_left_nan_comparator = d_right_left_comparator.equal_to<true>(
        nullate::YES{},
        _nulls_equal,
        cudf::detail::row::equality::nan_equal_physical_equality_comparator{});
      cuco::static_set_ref set_ref{empty_sentinel_key,
                                   comparator_adapter{d_right_left_nan_comparator},
                                   nested_probing_scheme{},
                                   cuco::thread_scope_device,
                                   _bucket_storage.ref()};
      auto query_ref = set_ref.rebind_operators(cuco::op::contains);
      return query_right_table<nested_probing_scheme::cg_size>(
        left, preprocessed_left, kind, query_ref, stream, mr);
    } else {
      auto d_right_left_nan_comparator = d_right_left_comparator.equal_to<false>(
        nullate::YES{},
        _nulls_equal,
        cudf::detail::row::equality::nan_equal_physical_equality_comparator{});
      cuco::static_set_ref set_ref{empty_sentinel_key,
                                   comparator_adapter{d_right_left_nan_comparator},
                                   simple_probing_scheme{},
                                   cuco::thread_scope_device,
                                   _bucket_storage.ref()};
      auto query_ref = set_ref.rebind_operators(cuco::op::contains);
      return query_right_table<simple_probing_scheme::cg_size>(
        left, preprocessed_left, kind, query_ref, stream, mr);
    }
  }
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> distinct_filtered_join::semi_join(
  cudf::table_view const& left, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  // Early return for empty right or left table
  if (_right.num_rows() == 0 || left.num_rows() == 0) {
    return std::make_unique<rmm::device_uvector<cudf::size_type>>(0, stream, mr);
  }

  return semi_anti_join(left, join_kind::LEFT_SEMI_JOIN, stream, mr);
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> distinct_filtered_join::anti_join(
  cudf::table_view const& left, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  // Early return for empty left table
  if (left.num_rows() == 0) {
    return std::make_unique<rmm::device_uvector<cudf::size_type>>(0, stream, mr);
  }
  if (_right.num_rows() == 0) {
    auto result =
      std::make_unique<rmm::device_uvector<cudf::size_type>>(left.num_rows(), stream, mr);
    thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     result->begin(),
                     result->end());
    return result;
  }

  return semi_anti_join(left, join_kind::LEFT_ANTI_JOIN, stream, mr);
}

}  // namespace detail

filtered_join::~filtered_join() = default;

filtered_join::filtered_join(cudf::table_view const& build,
                             null_equality compare_nulls,
                             double load_factor,
                             rmm::cuda_stream_view stream,
                             cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : _impl{std::make_unique<cudf::detail::distinct_filtered_join>(
      build, compare_nulls, load_factor, stream, std::move(mr))}
{
}

filtered_join::filtered_join(cudf::table_view const& build,
                             null_equality compare_nulls,
                             rmm::cuda_stream_view stream,
                             cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : filtered_join(
      build, compare_nulls, cudf::detail::CUCO_DESIRED_LOAD_FACTOR, stream, std::move(mr))
{
}

std::unique_ptr<rmm::device_uvector<size_type>> filtered_join::semi_join(
  cudf::table_view const& probe,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  return _impl->semi_join(probe, stream, mr);
}

std::unique_ptr<rmm::device_uvector<size_type>> filtered_join::anti_join(
  cudf::table_view const& probe,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  return _impl->anti_join(probe, stream, mr);
}

}  // namespace cudf
