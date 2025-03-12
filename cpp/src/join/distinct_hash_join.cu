/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
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
#include "join_common_utils.hpp"

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/join/distinct_hash_join.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/join.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cooperative_groups.h>
#include <cub/block/block_scan.cuh>
#include <cuco/static_set.cuh>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>

#include <cstddef>
#include <limits>
#include <memory>
#include <utility>

namespace cudf {
namespace detail {
namespace {

bool constexpr has_nulls = true;  ///< Always has nulls

/**
 * @brief Device functor to create a pair of {hash_value, row_index} for a given row.
 */
template <typename T>
class build_keys_fn {
  using hasher =
    cudf::experimental::row::hash::device_row_hasher<cudf::hashing::detail::default_hash,
                                                     cudf::nullate::DYNAMIC>;

 public:
  CUDF_HOST_DEVICE constexpr build_keys_fn(hasher const& hash) : _hash{hash} {}

  __device__ __forceinline__ auto operator()(size_type i) const noexcept
  {
    return cuco::pair{_hash(i), T{i}};
  }

 private:
  hasher _hash;
};

/**
 * @brief Device output transform functor to construct `size_type` with `cuco::pair<hash_value_type,
 * rhs_index_type>`
 */
struct output_fn {
  __device__ constexpr cudf::size_type operator()(
    cuco::pair<hash_value_type, rhs_index_type> const& x) const
  {
    return static_cast<cudf::size_type>(x.second);
  }
};
}  // namespace

distinct_hash_join::distinct_hash_join(cudf::table_view const& build,
                                       cudf::null_equality compare_nulls,
                                       rmm::cuda_stream_view stream)
  : _has_nested_columns{cudf::has_nested_columns(build)},
    _nulls_equal{compare_nulls},
    _build{build},
    _preprocessed_build{
      cudf::experimental::row::equality::preprocessed_table::create(_build, stream)},
    _hash_table{build.num_rows(),
                CUCO_DESIRED_LOAD_FACTOR,
                cuco::empty_key{cuco::pair{std::numeric_limits<hash_value_type>::max(),
                                           rhs_index_type{JoinNoneValue}}},
                always_not_equal{},
                {},
                cuco::thread_scope_device,
                cuco_storage_type{},
                cudf::detail::cuco_allocator<char>{rmm::mr::polymorphic_allocator<char>{}, stream},
                stream.value()}
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(0 != this->_build.num_columns(), "Hash join build table is empty");

  if (this->_build.num_rows() == 0) { return; }

  auto const row_hasher = experimental::row::hash::row_hasher{this->_preprocessed_build};
  auto const d_hasher   = row_hasher.device_hasher(nullate::DYNAMIC{has_nulls});

  auto const iter =
    cudf::detail::make_counting_transform_iterator(0, build_keys_fn<rhs_index_type>{d_hasher});

  size_type const build_table_num_rows{build.num_rows()};
  if (this->_nulls_equal == cudf::null_equality::EQUAL or (not cudf::nullable(this->_build))) {
    this->_hash_table.insert_async(iter, iter + build_table_num_rows, stream.value());
  } else {
    auto stencil = thrust::counting_iterator<size_type>{0};
    auto const row_bitmask =
      cudf::detail::bitmask_and(this->_build, stream, cudf::get_current_device_resource_ref())
        .first;
    auto const pred =
      cudf::detail::row_is_valid{reinterpret_cast<bitmask_type const*>(row_bitmask.data())};

    // insert valid rows
    this->_hash_table.insert_if_async(
      iter, iter + build_table_num_rows, stencil, pred, stream.value());
  }
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
distinct_hash_join::inner_join(cudf::table_view const& probe,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr) const
{
  cudf::scoped_range range{"distinct_hash_join::inner_join"};

  size_type const probe_table_num_rows{probe.num_rows()};

  // If output size is zero, return immediately
  if (probe_table_num_rows == 0) {
    return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                     std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
  }

  auto preprocessed_probe =
    cudf::experimental::row::equality::preprocessed_table::create(probe, stream);
  auto const two_table_equal = cudf::experimental::row::equality::two_table_comparator(
    preprocessed_probe, _preprocessed_build);

  auto build_indices =
    std::make_unique<rmm::device_uvector<size_type>>(probe_table_num_rows, stream, mr);
  auto probe_indices =
    std::make_unique<rmm::device_uvector<size_type>>(probe_table_num_rows, stream, mr);

  auto const probe_row_hasher = cudf::experimental::row::hash::row_hasher{preprocessed_probe};
  auto const d_probe_hasher   = probe_row_hasher.device_hasher(nullate::DYNAMIC{has_nulls});
  auto const iter             = cudf::detail::make_counting_transform_iterator(
    0, build_keys_fn<lhs_index_type>{d_probe_hasher});

  auto found_indices = rmm::device_uvector<size_type>(probe_table_num_rows, stream);
  auto const found_begin =
    thrust::make_transform_output_iterator(found_indices.begin(), output_fn{});

  auto const comparator_helper = [&](auto device_comparator) {
    // If `idx` is within the range `[0, probe_table_num_rows)` and `found_indices[idx]` is not
    // equal to `JoinNoneValue`, then `idx` has a match in the hash set.
    if (this->_nulls_equal == cudf::null_equality::EQUAL or (not cudf::nullable(probe))) {
      this->_hash_table.find_async(iter,
                                   iter + probe_table_num_rows,
                                   comparator_adapter{device_comparator},
                                   hasher{},
                                   found_begin,
                                   stream.value());
    } else {
      auto stencil = thrust::counting_iterator<size_type>{0};
      auto const row_bitmask =
        cudf::detail::bitmask_and(probe, stream, cudf::get_current_device_resource_ref()).first;
      auto const pred =
        cudf::detail::row_is_valid{reinterpret_cast<bitmask_type const*>(row_bitmask.data())};

      this->_hash_table.find_if_async(iter,
                                      iter + probe_table_num_rows,
                                      stencil,
                                      pred,
                                      comparator_adapter{device_comparator},
                                      hasher{},
                                      found_begin,
                                      stream.value());
    }
  };

  if (_has_nested_columns) {
    auto const device_comparator =
      two_table_equal.equal_to<true>(nullate::DYNAMIC{has_nulls}, _nulls_equal);
    comparator_helper(device_comparator);
  } else {
    auto const device_comparator =
      two_table_equal.equal_to<false>(nullate::DYNAMIC{has_nulls}, _nulls_equal);
    comparator_helper(device_comparator);
  }

  auto const tuple_iter = cudf::detail::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<thrust::tuple<size_type, size_type>>(
      [found_iter = found_indices.begin()] __device__(size_type idx) {
        return thrust::tuple{*(found_iter + idx), idx};
      }));
  auto const output_begin =
    thrust::make_zip_iterator(build_indices->begin(), probe_indices->begin());
  auto const output_end =
    thrust::copy_if(rmm::exec_policy_nosync(stream),
                    tuple_iter,
                    tuple_iter + probe_table_num_rows,
                    found_indices.begin(),
                    output_begin,
                    cuda::proclaim_return_type<bool>(
                      [] __device__(size_type idx) { return idx != JoinNoneValue; }));
  auto const actual_size = std::distance(output_begin, output_end);

  build_indices->resize(actual_size, stream);
  probe_indices->resize(actual_size, stream);

  return {std::move(probe_indices), std::move(build_indices)};
}

std::unique_ptr<rmm::device_uvector<size_type>> distinct_hash_join::left_join(
  cudf::table_view const& probe,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  cudf::scoped_range range{"distinct_hash_join::left_join"};

  size_type const probe_table_num_rows{probe.num_rows()};

  // If output size is zero, return empty
  if (probe_table_num_rows == 0) {
    return std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr);
  }

  auto build_indices =
    std::make_unique<rmm::device_uvector<size_type>>(probe_table_num_rows, stream, mr);

  // If build table is empty, return probe table
  if (this->_build.num_rows() == 0) {
    thrust::fill(
      rmm::exec_policy_nosync(stream), build_indices->begin(), build_indices->end(), JoinNoneValue);
  } else {
    auto preprocessed_probe =
      cudf::experimental::row::equality::preprocessed_table::create(probe, stream);
    auto const two_table_equal = cudf::experimental::row::equality::two_table_comparator(
      preprocessed_probe, _preprocessed_build);

    auto const probe_row_hasher = cudf::experimental::row::hash::row_hasher{preprocessed_probe};
    auto const d_probe_hasher   = probe_row_hasher.device_hasher(nullate::DYNAMIC{has_nulls});
    auto const iter             = cudf::detail::make_counting_transform_iterator(
      0, build_keys_fn<lhs_index_type>{d_probe_hasher});

    auto const output_begin =
      thrust::make_transform_output_iterator(build_indices->begin(), output_fn{});
    auto const comparator_helper = [&](auto device_comparator) {
      if (this->_nulls_equal == cudf::null_equality::EQUAL or (not cudf::nullable(probe))) {
        this->_hash_table.find_async(iter,
                                     iter + probe_table_num_rows,
                                     comparator_adapter{device_comparator},
                                     hasher{},
                                     output_begin,
                                     stream.value());
      } else {
        auto stencil = thrust::counting_iterator<size_type>{0};
        auto const row_bitmask =
          cudf::detail::bitmask_and(probe, stream, cudf::get_current_device_resource_ref()).first;
        auto const pred =
          cudf::detail::row_is_valid{reinterpret_cast<bitmask_type const*>(row_bitmask.data())};

        this->_hash_table.find_if_async(iter,
                                        iter + probe_table_num_rows,
                                        stencil,
                                        pred,
                                        comparator_adapter{device_comparator},
                                        hasher{},
                                        output_begin,
                                        stream.value());
      }
    };

    if (_has_nested_columns) {
      auto const device_comparator =
        two_table_equal.equal_to<true>(nullate::DYNAMIC{has_nulls}, _nulls_equal);
      comparator_helper(device_comparator);
    } else {
      auto const device_comparator =
        two_table_equal.equal_to<false>(nullate::DYNAMIC{has_nulls}, _nulls_equal);
      comparator_helper(device_comparator);
    }
  }

  return build_indices;
}
}  // namespace detail

distinct_hash_join::~distinct_hash_join() = default;

distinct_hash_join::distinct_hash_join(cudf::table_view const& build,
                                       null_equality compare_nulls,
                                       rmm::cuda_stream_view stream)
  : _impl{std::make_unique<impl_type>(build, compare_nulls, stream)}
{
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
distinct_hash_join::inner_join(cudf::table_view const& probe,
                               rmm::cuda_stream_view stream,
                               rmm::device_async_resource_ref mr) const
{
  return _impl->inner_join(probe, stream, mr);
}

std::unique_ptr<rmm::device_uvector<size_type>> distinct_hash_join::left_join(
  cudf::table_view const& probe,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr) const
{
  return _impl->left_join(probe, stream, mr);
}
}  // namespace cudf
