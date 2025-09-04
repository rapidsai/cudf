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

#include "join_common_utils.cuh"

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/join/filtered_join.cuh>
#include <cudf/detail/join/join.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cuco/bucket_storage.cuh>
#include <cuco/detail/open_addressing/kernels.cuh>
#include <cuco/detail/storage/counter_storage.cuh>
#include <cuco/extent.cuh>
#include <cuco/operator.hpp>
#include <cuco/static_multiset_ref.cuh>
#include <cuco/static_set_ref.cuh>
#include <cuco/types.cuh>
#include <cuco/utility/cuda_thread_scope.cuh>
#include <cuda/std/iterator>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>

#include <algorithm>
#include <memory>
#include <type_traits>

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
  __device__ bool operator()(size_type idx)
  {
    return flagged[idx] == (kind == join_kind::LEFT_SEMI_JOIN);
  }
};

}  // namespace

auto filtered_join::compute_bucket_storage_size(cudf::table_view tbl, double load_factor)
{
  return std::max(
    {static_cast<cudf::size_type>(
       cuco::make_valid_extent<primitive_probing_scheme, storage_type, cudf::size_type>(
         tbl.num_rows(), load_factor)),
     static_cast<cudf::size_type>(
       cuco::make_valid_extent<nested_probing_scheme, storage_type, cudf::size_type>(tbl.num_rows(),
                                                                                     load_factor)),
     static_cast<cudf::size_type>(
       cuco::make_valid_extent<simple_probing_scheme, storage_type, cudf::size_type>(
         tbl.num_rows(), load_factor))});
}

template <int32_t CGSize, typename Ref>
void filtered_join::insert_build_table(Ref const& insert_ref, rmm::cuda_stream_view stream)
{
  cudf::scoped_range range{"filtered_join_with_set::insert_build_table"};
  auto insert = [&]<typename Iterator>(Iterator build_iter) {
    // Build hash table by inserting all rows from build table
    auto const grid_size = cuco::detail::grid_size(_build.num_rows(), CGSize);

    if (_build_props.has_nulls && _nulls_equal == null_equality::UNEQUAL) {
      auto const bitmask_buffer_and_ptr = build_row_bitmask(_build, stream);
      auto const row_bitmask_ptr        = bitmask_buffer_and_ptr.second;
      cuco::detail::open_addressing_ns::insert_if_n<CGSize, cuco::detail::default_block_size()>
        <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
          build_iter,
          _build.num_rows(),
          thrust::counting_iterator<size_type>{0},
          row_is_valid{row_bitmask_ptr},
          insert_ref);
    } else {
      cuco::detail::open_addressing_ns::insert_if_n<CGSize, cuco::detail::default_block_size()>
        <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
          build_iter,
          _build.num_rows(),
          thrust::constant_iterator<bool>{true},
          cuda::std::identity{},
          insert_ref);
    }
  };

  if (cudf::is_primitive_row_op_compatible(_build) && !_build_props.has_floating_point) {
    auto const d_build_hasher =
      primitive_row_hasher{nullate::DYNAMIC{_build_props.has_nulls}, _preprocessed_build};
    auto const build_iter = cudf::detail::make_counting_transform_iterator(
      size_type{0}, key_pair_fn<lhs_index_type, primitive_row_hasher>{d_build_hasher});

    insert(build_iter);
  } else {
    auto const d_build_hasher =
      cudf::experimental::row::hash::row_hasher{_preprocessed_build}.device_hasher(
        nullate::DYNAMIC(_build_props.has_nulls));
    auto const build_iter = cudf::detail::make_counting_transform_iterator(
      size_type{0}, key_pair_fn<lhs_index_type, row_hasher>{d_build_hasher});

    insert(build_iter);
  }
}

template <int32_t CGSize, typename Ref>
std::unique_ptr<rmm::device_uvector<cudf::size_type>> filtered_join_with_set::query_build_table(
  cudf::table_view const& probe,
  std::shared_ptr<cudf::experimental::row::equality::preprocessed_table> preprocessed_probe,
  join_kind kind,
  Ref query_ref,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  cudf::scoped_range range{"filtered_join_with_set::query_build_table"};
  auto const probe_has_nulls = has_nested_nulls(probe);
  auto const has_any_nulls   = probe_has_nulls || _build_props.has_nulls;

  auto query_set = [this,
                    probe,
                    probe_has_nulls,
                    query_ref,
                    stream]<typename InputProbeIterator, typename OutputContainsIterator>(
                     InputProbeIterator probe_iter, OutputContainsIterator contains_iter) {
    auto const grid_size = cuco::detail::grid_size(probe.num_rows(), CGSize);
    if (probe_has_nulls && _nulls_equal == null_equality::UNEQUAL) {
      auto const bitmask_buffer_and_ptr = build_row_bitmask(probe, stream);
      auto const row_bitmask_ptr        = bitmask_buffer_and_ptr.second;
      cuco::detail::open_addressing_ns::contains_if_n<CGSize, cuco::detail::default_block_size()>
        <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
          probe_iter,
          probe.num_rows(),
          thrust::counting_iterator<size_type>{0},
          row_is_valid{row_bitmask_ptr},
          contains_iter,
          query_ref);
    } else {
      cuco::detail::open_addressing_ns::contains_if_n<CGSize, cuco::detail::default_block_size()>
        <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
          probe_iter,
          probe.num_rows(),
          thrust::constant_iterator<bool>{true},
          cuda::std::identity{},
          contains_iter,
          query_ref);
    }
  };

  auto contains_map = rmm::device_uvector<bool>(probe.num_rows(), stream);
  if (cudf::is_primitive_row_op_compatible(_build) && !_build_props.has_floating_point) {
    auto const d_probe_hasher =
      primitive_row_hasher{nullate::DYNAMIC{has_any_nulls}, preprocessed_probe};
    auto const probe_iter = cudf::detail::make_counting_transform_iterator(
      size_type{0}, key_pair_fn<rhs_index_type, primitive_row_hasher>{d_probe_hasher});

    query_set(probe_iter, contains_map.begin());
  } else {
    auto const d_probe_hasher =
      cudf::experimental::row::hash::row_hasher{preprocessed_probe}.device_hasher(
        nullate::DYNAMIC(has_any_nulls));
    auto const probe_iter = cudf::detail::make_counting_transform_iterator(
      size_type{0}, key_pair_fn<rhs_index_type, row_hasher>{d_probe_hasher});

    query_set(probe_iter, contains_map.begin());
  }
  rmm::device_uvector<size_type> gather_map(probe.num_rows(), stream, mr);
  auto gather_map_end = thrust::copy_if(rmm::exec_policy(stream),
                                        thrust::counting_iterator<size_type>(0),
                                        thrust::counting_iterator<size_type>(probe.num_rows()),
                                        gather_map.begin(),
                                        gather_mask{kind, contains_map});
  gather_map.resize(cuda::std::distance(gather_map.begin(), gather_map_end), stream);
  return std::make_unique<rmm::device_uvector<size_type>>(std::move(gather_map));
}

filtered_join::filtered_join(cudf::table_view const& build,
                             cudf::null_equality compare_nulls,
                             double load_factor,
                             rmm::cuda_stream_view stream)
  : _build_props{build_properties{
      cudf::has_nested_nulls(build),
      std::any_of(build.begin(),
                  build.end(),
                  [](auto const& col) { return cudf::is_floating_point(col.type()); }),
      cudf::has_nested_columns(build)}},
    _nulls_equal{compare_nulls},
    _build{build},
    _preprocessed_build{
      cudf::experimental::row::equality::preprocessed_table::create(_build, stream)},
    _bucket_storage{cuco::extent<cudf::size_type>{compute_bucket_storage_size(build, load_factor)},
                    cuco_allocator<char>{rmm::mr::polymorphic_allocator<char>{}, stream.value()}}
{
  _bucket_storage.initialize(empty_sentinel_key, stream);
}

filtered_join_with_set::filtered_join_with_set(cudf::table_view const& build,
                                               cudf::null_equality compare_nulls,
                                               double load_factor,
                                               rmm::cuda_stream_view stream)
  : filtered_join(build, compare_nulls, load_factor, stream)
{
  cudf::scoped_range range{"filtered_join_with_set::filtered_join_with_set"};
  if (cudf::is_primitive_row_op_compatible(build) && !_build_props.has_floating_point) {
    auto const d_build_comparator =
      primitive_row_comparator{nullate::DYNAMIC{_build_props.has_nulls},
                               _preprocessed_build,
                               _preprocessed_build,
                               compare_nulls};
    cuco::static_set_ref set_ref{empty_sentinel_key,
                                 insertion_adapter{d_build_comparator},
                                 primitive_probing_scheme{},
                                 cuco::thread_scope_device,
                                 _bucket_storage.ref()};
    auto insert_ref = set_ref.rebind_operators(cuco::insert);
    insert_build_table<primitive_probing_scheme::cg_size>(insert_ref, stream);
  } else if (_build_props.has_nested_columns) {
    auto const d_build_comparator =
      cudf::experimental::row::equality::self_comparator{_preprocessed_build}.equal_to<true>(
        nullate::DYNAMIC{_build_props.has_nulls},
        compare_nulls,
        cudf::experimental::row::equality::nan_equal_physical_equality_comparator{});
    cuco::static_set_ref set_ref{empty_sentinel_key,
                                 insertion_adapter{d_build_comparator},
                                 nested_probing_scheme{},
                                 cuco::thread_scope_device,
                                 _bucket_storage.ref()};
    auto insert_ref = set_ref.rebind_operators(cuco::insert);
    insert_build_table<nested_probing_scheme::cg_size>(insert_ref, stream);
  } else {
    auto const d_build_comparator =
      cudf::experimental::row::equality::self_comparator{_preprocessed_build}.equal_to<false>(
        nullate::DYNAMIC{_build_props.has_nulls},
        compare_nulls,
        cudf::experimental::row::equality::nan_equal_physical_equality_comparator{});
    cuco::static_set_ref set_ref{empty_sentinel_key,
                                 insertion_adapter{d_build_comparator},
                                 simple_probing_scheme{},
                                 cuco::thread_scope_device,
                                 _bucket_storage.ref()};
    auto insert_ref = set_ref.rebind_operators(cuco::insert);
    insert_build_table<simple_probing_scheme::cg_size>(insert_ref, stream);
  }
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> filtered_join_with_set::semi_anti_join(
  cudf::table_view const& probe,
  join_kind kind,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  cudf::scoped_range range{"filtered_join_with_set::semi_anti_join"};
  auto const has_any_nulls = has_nested_nulls(probe) || _build_props.has_nulls;

  nvtxRangePushA("filtered_join_with_set::semi_anti_join::preprocessed_probe");
  auto const preprocessed_probe =
    cudf::experimental::row::equality::preprocessed_table::create(probe, stream);
  nvtxRangePop();

  if (cudf::is_primitive_row_op_compatible(_build) && !_build_props.has_floating_point) {
    auto const d_build_probe_comparator = primitive_row_comparator{
      nullate::DYNAMIC{has_any_nulls}, _preprocessed_build, preprocessed_probe, _nulls_equal};

    cuco::static_set_ref set_ref{empty_sentinel_key,
                                 comparator_adapter{d_build_probe_comparator},
                                 primitive_probing_scheme{},
                                 cuco::thread_scope_device,
                                 _bucket_storage.ref()};
    auto query_ref = set_ref.rebind_operators(cuco::op::contains);
    return query_build_table<primitive_probing_scheme::cg_size>(
      probe, preprocessed_probe, kind, query_ref, stream, mr);
  } else {
    auto const d_build_probe_comparator = cudf::experimental::row::equality::two_table_comparator{
      _preprocessed_build, preprocessed_probe};

    if (_build_props.has_nested_columns) {
      auto d_build_probe_nan_comparator = d_build_probe_comparator.equal_to<true>(
        nullate::DYNAMIC{has_any_nulls},
        _nulls_equal,
        cudf::experimental::row::equality::nan_equal_physical_equality_comparator{});
      cuco::static_set_ref set_ref{empty_sentinel_key,
                                   comparator_adapter{d_build_probe_nan_comparator},
                                   nested_probing_scheme{},
                                   cuco::thread_scope_device,
                                   _bucket_storage.ref()};
      auto query_ref = set_ref.rebind_operators(cuco::op::contains);
      return query_build_table<nested_probing_scheme::cg_size>(
        probe, preprocessed_probe, kind, query_ref, stream, mr);
    } else {
      auto d_build_probe_nan_comparator = d_build_probe_comparator.equal_to<false>(
        nullate::DYNAMIC{has_any_nulls},
        _nulls_equal,
        cudf::experimental::row::equality::nan_equal_physical_equality_comparator{});
      cuco::static_set_ref set_ref{empty_sentinel_key,
                                   comparator_adapter{d_build_probe_nan_comparator},
                                   simple_probing_scheme{},
                                   cuco::thread_scope_device,
                                   _bucket_storage.ref()};
      auto query_ref = set_ref.rebind_operators(cuco::op::contains);
      return query_build_table<simple_probing_scheme::cg_size>(
        probe, preprocessed_probe, kind, query_ref, stream, mr);
    }
  }
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> filtered_join_with_set::semi_join(
  cudf::table_view const& probe, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  return semi_anti_join(probe, join_kind::LEFT_SEMI_JOIN, stream, mr);
}

std::unique_ptr<rmm::device_uvector<cudf::size_type>> filtered_join_with_set::anti_join(
  cudf::table_view const& probe, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  return semi_anti_join(probe, join_kind::LEFT_ANTI_JOIN, stream, mr);
}

}  // namespace detail

filtered_join::~filtered_join() = default;

filtered_join::filtered_join(cudf::table_view const& build,
                             null_equality compare_nulls,
                             set_as_build_table reuse_tbl,
                             double load_factor,
                             rmm::cuda_stream_view stream)
  : _reuse_tbl{reuse_tbl}, _impl{std::make_unique<cudf::detail::filtered_join_with_set>(
    build, compare_nulls, load_factor, stream)}
{
}

filtered_join::filtered_join(cudf::table_view const& build,
                             null_equality compare_nulls,
                             set_as_build_table reuse_tbl,
                             rmm::cuda_stream_view stream)
  : filtered_join(build, compare_nulls, reuse_tbl, cudf::detail::CUCO_DESIRED_LOAD_FACTOR, stream)
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
