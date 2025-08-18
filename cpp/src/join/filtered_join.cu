/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include "cuco/utility/cuda_thread_scope.cuh"
#include "join_common_utils.cuh"
#include "thrust/iterator/counting_iterator.h"

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/join/filtered_join.cuh>
#include <cudf/detail/join/join.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cuco/bucket_storage.cuh>
#include <cuco/detail/open_addressing/kernels.cuh>
#include <cuco/extent.cuh>
#include <cuco/operator.hpp>
#include <cuco/static_set_ref.cuh>
#include <cuco/types.cuh>

#include <algorithm>
#include <limits>
#include <memory>

namespace cudf {
namespace detail {
namespace {

auto compute_bucket_storage_size(cudf::table_view tbl, double load_factor)
{
  return std::max({static_cast<cudf::size_type>(
                     cuco::make_valid_extent<filtered_join::primitive_probing_scheme,
                                             filtered_join::storage_type,
                                             cudf::size_type>(tbl.num_rows(), load_factor)),
                   static_cast<cudf::size_type>(
                     cuco::make_valid_extent<filtered_join::nested_probing_scheme,
                                             filtered_join::storage_type,
                                             cudf::size_type>(tbl.num_rows(), load_factor)),
                   static_cast<cudf::size_type>(
                     cuco::make_valid_extent<filtered_join::simple_probing_scheme,
                                             filtered_join::storage_type,
                                             cudf::size_type>(tbl.num_rows(), load_factor))});
}

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

}  // namespace

filtered_join::filtered_join(cudf::table_view const& build,
                             null_equality compare_nulls,
                             rmm::cuda_stream_view stream)
  // If we cannot know beforehand about null existence then let's assume that there are nulls.
  : filtered_join(build, compare_nulls, cudf::detail::CUCO_DESIRED_LOAD_FACTOR, stream)
{
}

filtered_join::filtered_join(cudf::table_view const& build,
                             null_equality compare_nulls,
                             double load_factor,
                             rmm::cuda_stream_view stream)
  : _build_has_nulls{cudf::has_nested_nulls(build)},
    _nulls_equal{compare_nulls},
    _build{build},
    _preprocessed_build{
      cudf::experimental::row::equality::preprocessed_table::create(_build, stream)},
    _bucket_storage{cuco::extent<cudf::size_type>{compute_bucket_storage_size(build, load_factor)},
                    cuco_allocator<char>{rmm::mr::polymorphic_allocator<char>{}, stream.value()}}
{
  auto const build_has_floating_point =
    std::any_of(_build.begin(), _build.end(), [](auto const& col) {
      return cudf::is_floating_point(col.type());
    });
  auto empty_sentinel_key = cuco::empty_key{
    cuco::pair{std::numeric_limits<hash_value_type>::max(), rhs_index_type{JoinNoneValue}}};
  _bucket_storage.initialize(empty_sentinel_key);

  if (cudf::is_primitive_row_op_compatible(_build) && !build_has_floating_point) {
    auto const d_build_hasher =
      primitive_row_hasher{nullate::DYNAMIC{_build_has_nulls}, _preprocessed_build};
    auto const d_build_comparator = cudf::row::primitive::row_equality_comparator{
      nullate::DYNAMIC{_build_has_nulls}, _preprocessed_build, _preprocessed_build, _nulls_equal};

    cuco::static_set_ref set_ref{empty_sentinel_key,
                                 comparator_adapter{d_build_comparator},
                                 primitive_probing_scheme{},
                                 cuco::thread_scope_device,
                                 _bucket_storage.ref()};
    auto insert_ref = set_ref.rebind_operators(cuco::insert);

    // Build hash table by inserting all rows from build table
    auto const grid_size =
      cuco::detail::grid_size(_build.num_rows(), primitive_probing_scheme::cg_size);
    auto const build_iter = cudf::detail::make_counting_transform_iterator(
      size_type{0}, keys_adapter<rhs_index_type, primitive_row_hasher>{d_build_hasher});

    if (_build_has_nulls && _nulls_equal == null_equality::UNEQUAL) {
      auto const bitmask_buffer_and_ptr = build_row_bitmask(_build, stream);
      auto const row_bitmask_ptr        = bitmask_buffer_and_ptr.second;

      cuco::detail::open_addressing_ns::insert_if_n<primitive_probing_scheme::cg_size,
                                                    cuco::detail::default_block_size()>
        <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
          build_iter,
          _build.num_rows(),
          thrust::counting_iterator<size_type>{0},
          row_is_valid{row_bitmask_ptr},
          insert_ref);
    } else {
      cuco::detail::open_addressing_ns::insert_if_n<primitive_probing_scheme::cg_size,
                                                    cuco::detail::default_block_size()>
        <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
          build_iter,
          _build.num_rows(),
          thrust::constant_iterator<bool>{true},
          cuda::std::identity{},
          insert_ref);
    }
  } else {
    auto const build_has_nested_columns = cudf::has_nested_columns(_build);

    auto const d_build_hasher =
      cudf::experimental::row::hash::row_hasher{_preprocessed_build}.device_hasher(
        nullate::DYNAMIC(_build_has_nulls));
    auto const d_build_comparator =
      cudf::experimental::row::equality::self_comparator{_preprocessed_build};

    auto const build_iter = cudf::detail::make_counting_transform_iterator(
      size_type{0}, keys_adapter<rhs_index_type, row_hasher>{d_build_hasher});

    if (build_has_nested_columns) {
      auto d_build_nan_comparator = d_build_comparator.equal_to<true>(
        nullate::DYNAMIC{_build_has_nulls},
        _nulls_equal,
        cudf::experimental::row::equality::nan_equal_physical_equality_comparator{});
      cuco::static_set_ref set_ref{empty_sentinel_key,
                                   comparator_adapter{d_build_nan_comparator},
                                   nested_probing_scheme{},
                                   cuco::thread_scope_device,
                                   _bucket_storage.ref()};
      auto insert_ref = set_ref.rebind_operators(cuco::insert);

      auto const grid_size =
        cuco::detail::grid_size(_build.num_rows(), nested_probing_scheme::cg_size);

      if (_build_has_nulls && _nulls_equal == null_equality::UNEQUAL) {
        auto const bitmask_buffer_and_ptr = build_row_bitmask(build, stream);
        auto const row_bitmask_ptr        = bitmask_buffer_and_ptr.second;

        cuco::detail::open_addressing_ns::insert_if_n<nested_probing_scheme::cg_size,
                                                      cuco::detail::default_block_size()>
          <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
            build_iter,
            _build.num_rows(),
            thrust::counting_iterator<size_type>{0},
            row_is_valid{row_bitmask_ptr},
            insert_ref);
      } else {
        cuco::detail::open_addressing_ns::insert_if_n<nested_probing_scheme::cg_size,
                                                      cuco::detail::default_block_size()>
          <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
            build_iter,
            _build.num_rows(),
            thrust::constant_iterator<bool>{true},
            cuda::std::identity{},
            insert_ref);
      }
    } else {
      auto d_build_nan_comparator = d_build_comparator.equal_to<false>(
        nullate::DYNAMIC{_build_has_nulls},
        _nulls_equal,
        cudf::experimental::row::equality::nan_equal_physical_equality_comparator{});
      cuco::static_set_ref set_ref{empty_sentinel_key,
                                   comparator_adapter{d_build_nan_comparator},
                                   simple_probing_scheme{},
                                   cuco::thread_scope_device,
                                   _bucket_storage.ref()};
      auto insert_ref = set_ref.rebind_operators(cuco::insert);

      auto const grid_size =
        cuco::detail::grid_size(_build.num_rows(), simple_probing_scheme::cg_size);

      if (_build_has_nulls && _nulls_equal == null_equality::UNEQUAL) {
        auto const bitmask_buffer_and_ptr = build_row_bitmask(build, stream);
        auto const row_bitmask_ptr        = bitmask_buffer_and_ptr.second;

        cuco::detail::open_addressing_ns::insert_if_n<simple_probing_scheme::cg_size,
                                                      cuco::detail::default_block_size()>
          <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
            build_iter,
            _build.num_rows(),
            thrust::counting_iterator<size_type>{0},
            row_is_valid{row_bitmask_ptr},
            insert_ref);
      } else {
        cuco::detail::open_addressing_ns::insert_if_n<simple_probing_scheme::cg_size,
                                                      cuco::detail::default_block_size()>
          <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
            build_iter,
            _build.num_rows(),
            thrust::constant_iterator<bool>{true},
            cuda::std::identity{},
            insert_ref);
      }
    }
  }
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> filtered_join::semi_join(
  cudf::table_view const& probe, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  auto const probe_has_nulls = has_nested_nulls(probe);
  auto const has_any_nulls   = probe_has_nulls || _build_has_nulls;

  auto empty_sentinel_key = cuco::empty_key{
    cuco::pair{std::numeric_limits<hash_value_type>::max(), rhs_index_type{JoinNoneValue}}};
  auto const preprocessed_probe =
    cudf::experimental::row::equality::preprocessed_table::create(probe, stream);

  auto const build_has_floating_point =
    std::any_of(_build.begin(), _build.end(), [](auto const& col) {
      return cudf::is_floating_point(col.type());
    });

  auto contained = rmm::device_uvector<bool>(probe.num_rows(), stream);

  if (cudf::is_primitive_row_op_compatible(_build) && !build_has_floating_point) {
    auto const d_probe_hasher =
      primitive_row_hasher{nullate::DYNAMIC{has_any_nulls}, preprocessed_probe};
    auto const d_probe_build_comparator = cudf::row::primitive::row_equality_comparator{
      nullate::DYNAMIC{has_any_nulls}, preprocessed_probe, _preprocessed_build, _nulls_equal};

    cuco::static_set_ref set_ref{empty_sentinel_key,
                                 comparator_adapter{d_probe_build_comparator},
                                 primitive_probing_scheme{},
                                 cuco::thread_scope_device,
                                 _bucket_storage.ref()};
    auto contains_ref = set_ref.rebind_operators(cuco::op::contains);

    auto const grid_size =
      cuco::detail::grid_size(probe.num_rows(), primitive_probing_scheme::cg_size);
    auto const probe_iter = cudf::detail::make_counting_transform_iterator(
      size_type{0}, keys_adapter<lhs_index_type, primitive_row_hasher>{d_probe_hasher});

    if (probe_has_nulls && _nulls_equal == null_equality::UNEQUAL) {
      auto const bitmask_buffer_and_ptr = build_row_bitmask(probe, stream);
      auto const row_bitmask_ptr        = bitmask_buffer_and_ptr.second;

      cuco::detail::open_addressing_ns::contains_if_n<primitive_probing_scheme::cg_size,
                                                      cuco::detail::default_block_size()>
        <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
          probe_iter,
          probe.num_rows(),
          thrust::counting_iterator<size_type>{0},
          row_is_valid{row_bitmask_ptr},
          contained.begin(),
          contains_ref);
    } else {
      cuco::detail::open_addressing_ns::contains_if_n<primitive_probing_scheme::cg_size,
                                                      cuco::detail::default_block_size()>
        <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
          probe_iter,
          probe.num_rows(),
          thrust::constant_iterator<bool>{true},
          cuda::std::identity{},
          contained.begin(),
          contains_ref);
    }
  } else {
    auto const build_has_nested_columns = cudf::has_nested_columns(_build);

    auto const d_probe_hasher =
      cudf::experimental::row::hash::row_hasher{preprocessed_probe}.device_hasher(
        nullate::DYNAMIC(has_any_nulls));
    auto const d_probe_build_comparator = cudf::experimental::row::equality::two_table_comparator{
      preprocessed_probe, _preprocessed_build};

    auto const probe_iter = cudf::detail::make_counting_transform_iterator(
      size_type{0}, keys_adapter<lhs_index_type, row_hasher>{d_probe_hasher});

    if (build_has_nested_columns) {
      auto d_probe_build_nan_comparator = d_probe_build_comparator.equal_to<true>(
        nullate::DYNAMIC{has_any_nulls},
        _nulls_equal,
        cudf::experimental::row::equality::nan_equal_physical_equality_comparator{});
      cuco::static_set_ref set_ref{empty_sentinel_key,
                                   comparator_adapter{d_probe_build_nan_comparator},
                                   nested_probing_scheme{},
                                   cuco::thread_scope_device,
                                   _bucket_storage.ref()};
      auto contains_ref = set_ref.rebind_operators(cuco::op::contains);

      auto const grid_size =
        cuco::detail::grid_size(probe.num_rows(), nested_probing_scheme::cg_size);

      if (probe_has_nulls && _nulls_equal == null_equality::UNEQUAL) {
        auto const bitmask_buffer_and_ptr = build_row_bitmask(probe, stream);
        auto const row_bitmask_ptr        = bitmask_buffer_and_ptr.second;

        cuco::detail::open_addressing_ns::contains_if_n<nested_probing_scheme::cg_size,
                                                        cuco::detail::default_block_size()>
          <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
            probe_iter,
            probe.num_rows(),
            thrust::counting_iterator<size_type>{0},
            row_is_valid{row_bitmask_ptr},
            contained.begin(),
            contains_ref);
      } else {
        cuco::detail::open_addressing_ns::contains_if_n<nested_probing_scheme::cg_size,
                                                        cuco::detail::default_block_size()>
          <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
            probe_iter,
            probe.num_rows(),
            thrust::constant_iterator<bool>{true},
            cuda::std::identity{},
            contained.begin(),
            contains_ref);
      }
    } else {
      auto d_probe_build_nan_comparator = d_probe_build_comparator.equal_to<false>(
        nullate::DYNAMIC{has_any_nulls},
        _nulls_equal,
        cudf::experimental::row::equality::nan_equal_physical_equality_comparator{});
      cuco::static_set_ref set_ref{empty_sentinel_key,
                                   comparator_adapter{d_probe_build_nan_comparator},
                                   simple_probing_scheme{},
                                   cuco::thread_scope_device,
                                   _bucket_storage.ref()};
      auto contains_ref = set_ref.rebind_operators(cuco::op::contains);

      auto const grid_size =
        cuco::detail::grid_size(probe.num_rows(), simple_probing_scheme::cg_size);

      if (probe_has_nulls && _nulls_equal == null_equality::UNEQUAL) {
        auto const bitmask_buffer_and_ptr = build_row_bitmask(probe, stream);
        auto const row_bitmask_ptr        = bitmask_buffer_and_ptr.second;

        cuco::detail::open_addressing_ns::contains_if_n<simple_probing_scheme::cg_size,
                                                        cuco::detail::default_block_size()>
          <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
            probe_iter,
            probe.num_rows(),
            thrust::counting_iterator<size_type>{0},
            row_is_valid{row_bitmask_ptr},
            contained.begin(),
            contains_ref);
      } else {
        cuco::detail::open_addressing_ns::contains_if_n<simple_probing_scheme::cg_size,
                                                        cuco::detail::default_block_size()>
          <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
            probe_iter,
            probe.num_rows(),
            thrust::constant_iterator<bool>{true},
            cuda::std::identity{},
            contained.begin(),
            contains_ref);
      }
    }
  }

  auto gather_map     = detail::make_zeroed_device_uvector<size_type>(probe.num_rows(), stream, mr);
  auto gather_map_end = thrust::copy_if(rmm::exec_policy(stream),
                                        thrust::counting_iterator<size_type>(0),
                                        thrust::counting_iterator<size_type>(probe.num_rows()),
                                        gather_map.begin(),
                                        [d_flagged = contained.begin()] __device__(
                                          size_type const idx) { return *(d_flagged + idx) == 1; });

  gather_map.resize(cuda::std::distance(gather_map.begin(), gather_map_end), stream);
  return std::make_unique<rmm::device_uvector<size_type>>(std::move(gather_map));
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> filtered_join::anti_join(
  cudf::table_view const& probe, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  auto const probe_has_nulls = has_nested_nulls(probe);
  auto empty_sentinel_key    = cuco::empty_key{
    cuco::pair{std::numeric_limits<hash_value_type>::max(), rhs_index_type{JoinNoneValue}}};
  auto const preprocessed_probe =
    cudf::experimental::row::equality::preprocessed_table::create(probe, stream);

  auto const build_has_floating_point =
    std::any_of(_build.begin(), _build.end(), [](auto const& col) {
      return cudf::is_floating_point(col.type());
    });

  auto contained = rmm::device_uvector<bool>(probe.num_rows(), stream);

  if (cudf::is_primitive_row_op_compatible(_build) && !build_has_floating_point) {
    auto const d_probe_hasher =
      primitive_row_hasher{nullate::DYNAMIC{probe_has_nulls}, preprocessed_probe};
    auto const d_probe_build_comparator = cudf::row::primitive::row_equality_comparator{
      nullate::DYNAMIC{probe_has_nulls}, preprocessed_probe, _preprocessed_build, _nulls_equal};

    cuco::static_set_ref set_ref{empty_sentinel_key,
                                 comparator_adapter{d_probe_build_comparator},
                                 primitive_probing_scheme{},
                                 cuco::thread_scope_device,
                                 _bucket_storage.ref()};
    auto contains_ref = set_ref.rebind_operators(cuco::op::contains);

    auto const grid_size =
      cuco::detail::grid_size(probe.num_rows(), primitive_probing_scheme::cg_size);
    auto const probe_iter = cudf::detail::make_counting_transform_iterator(
      size_type{0}, keys_adapter<lhs_index_type, primitive_row_hasher>{d_probe_hasher});

    if (probe_has_nulls && _nulls_equal == null_equality::UNEQUAL) {
      auto const bitmask_buffer_and_ptr = build_row_bitmask(probe, stream);
      auto const row_bitmask_ptr        = bitmask_buffer_and_ptr.second;

      cuco::detail::open_addressing_ns::contains_if_n<primitive_probing_scheme::cg_size,
                                                      cuco::detail::default_block_size()>
        <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
          probe_iter,
          probe.num_rows(),
          thrust::counting_iterator<size_type>{0},
          row_is_valid{row_bitmask_ptr},
          contained.begin(),
          contains_ref);
    } else {
      cuco::detail::open_addressing_ns::contains_if_n<primitive_probing_scheme::cg_size,
                                                      cuco::detail::default_block_size()>
        <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
          probe_iter,
          probe.num_rows(),
          thrust::constant_iterator<bool>{true},
          cuda::std::identity{},
          contained.begin(),
          contains_ref);
    }
  } else {
    auto const build_has_nested_columns = cudf::has_nested_columns(_build);

    auto const d_probe_hasher =
      cudf::experimental::row::hash::row_hasher{preprocessed_probe}.device_hasher(
        nullate::DYNAMIC(probe_has_nulls));
    auto const d_probe_build_comparator = cudf::experimental::row::equality::two_table_comparator{
      preprocessed_probe, _preprocessed_build};

    auto const probe_iter = cudf::detail::make_counting_transform_iterator(
      size_type{0}, keys_adapter<lhs_index_type, row_hasher>{d_probe_hasher});

    if (build_has_nested_columns) {
      auto d_probe_build_nan_comparator = d_probe_build_comparator.equal_to<true>(
        nullate::DYNAMIC{probe_has_nulls},
        _nulls_equal,
        cudf::experimental::row::equality::nan_equal_physical_equality_comparator{});
      cuco::static_set_ref set_ref{empty_sentinel_key,
                                   comparator_adapter{d_probe_build_nan_comparator},
                                   nested_probing_scheme{},
                                   cuco::thread_scope_device,
                                   _bucket_storage.ref()};
      auto contains_ref = set_ref.rebind_operators(cuco::op::contains);

      auto const grid_size =
        cuco::detail::grid_size(probe.num_rows(), nested_probing_scheme::cg_size);

      if (probe_has_nulls && _nulls_equal == null_equality::UNEQUAL) {
        auto const bitmask_buffer_and_ptr = build_row_bitmask(probe, stream);
        auto const row_bitmask_ptr        = bitmask_buffer_and_ptr.second;

        cuco::detail::open_addressing_ns::contains_if_n<nested_probing_scheme::cg_size,
                                                        cuco::detail::default_block_size()>
          <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
            probe_iter,
            probe.num_rows(),
            thrust::counting_iterator<size_type>{0},
            row_is_valid{row_bitmask_ptr},
            contained.begin(),
            contains_ref);
      } else {
        cuco::detail::open_addressing_ns::contains_if_n<nested_probing_scheme::cg_size,
                                                        cuco::detail::default_block_size()>
          <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
            probe_iter,
            probe.num_rows(),
            thrust::constant_iterator<bool>{true},
            cuda::std::identity{},
            contained.begin(),
            contains_ref);
      }
    } else {
      auto d_probe_build_nan_comparator = d_probe_build_comparator.equal_to<false>(
        nullate::DYNAMIC{probe_has_nulls},
        _nulls_equal,
        cudf::experimental::row::equality::nan_equal_physical_equality_comparator{});
      cuco::static_set_ref set_ref{empty_sentinel_key,
                                   comparator_adapter{d_probe_build_nan_comparator},
                                   simple_probing_scheme{},
                                   cuco::thread_scope_device,
                                   _bucket_storage.ref()};
      auto contains_ref = set_ref.rebind_operators(cuco::op::contains);

      auto const grid_size =
        cuco::detail::grid_size(probe.num_rows(), simple_probing_scheme::cg_size);

      if (probe_has_nulls && _nulls_equal == null_equality::UNEQUAL) {
        auto const bitmask_buffer_and_ptr = build_row_bitmask(probe, stream);
        auto const row_bitmask_ptr        = bitmask_buffer_and_ptr.second;

        cuco::detail::open_addressing_ns::contains_if_n<simple_probing_scheme::cg_size,
                                                        cuco::detail::default_block_size()>
          <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
            probe_iter,
            probe.num_rows(),
            thrust::counting_iterator<size_type>{0},
            row_is_valid{row_bitmask_ptr},
            contained.begin(),
            contains_ref);
      } else {
        cuco::detail::open_addressing_ns::contains_if_n<simple_probing_scheme::cg_size,
                                                        cuco::detail::default_block_size()>
          <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
            probe_iter,
            probe.num_rows(),
            thrust::constant_iterator<bool>{true},
            cuda::std::identity{},
            contained.begin(),
            contains_ref);
      }
    }
  }

  auto gather_map     = detail::make_zeroed_device_uvector<size_type>(probe.num_rows(), stream, mr);
  auto gather_map_end = thrust::copy_if(rmm::exec_policy(stream),
                                        thrust::counting_iterator<size_type>(0),
                                        thrust::counting_iterator<size_type>(probe.num_rows()),
                                        gather_map.begin(),
                                        [d_flagged = contained.begin()] __device__(
                                          size_type const idx) { return *(d_flagged + idx) == 0; });

  gather_map.resize(cuda::std::distance(gather_map.begin(), gather_map_end), stream);
  return std::make_unique<rmm::device_uvector<size_type>>(std::move(gather_map));
}

}  // namespace detail

filtered_join::~filtered_join() = default;

filtered_join::filtered_join(cudf::table_view const& build,
                             null_equality compare_nulls,
                             double load_factor,
                             rmm::cuda_stream_view stream)
  : _impl{std::make_unique<impl_type>(build, compare_nulls, load_factor, stream)}
{
}

filtered_join::filtered_join(cudf::table_view const& build,
                             null_equality compare_nulls,
                             rmm::cuda_stream_view stream)
  : filtered_join(build, compare_nulls, cudf::detail::CUCO_DESIRED_LOAD_FACTOR, stream)
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
