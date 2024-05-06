/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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
#include <cudf/detail/distinct_hash_join.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/join.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cooperative_groups.h>
#include <cub/block/block_scan.cuh>
#include <cuco/static_set.cuh>
#include <thrust/fill.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/sequence.h>

#include <cstddef>
#include <limits>
#include <memory>
#include <utility>

namespace cudf {
namespace detail {
namespace {

template <cudf::has_nested HasNested>
auto prepare_device_equal(
  std::shared_ptr<cudf::experimental::row::equality::preprocessed_table> build,
  std::shared_ptr<cudf::experimental::row::equality::preprocessed_table> probe,
  bool has_nulls,
  cudf::null_equality compare_nulls)
{
  auto const two_table_equal =
    cudf::experimental::row::equality::two_table_comparator(build, probe);
  return comparator_adapter{two_table_equal.equal_to<HasNested == cudf::has_nested::YES>(
    nullate::DYNAMIC{has_nulls}, compare_nulls)};
}

/**
 * @brief Device functor to create a pair of {hash_value, row_index} for a given row.
 *
 * @tparam Hasher The type of internal hasher to compute row hash.
 */
template <typename Hasher, typename T>
class build_keys_fn {
 public:
  CUDF_HOST_DEVICE build_keys_fn(Hasher const& hash) : _hash{hash} {}

  __device__ __forceinline__ auto operator()(size_type i) const noexcept
  {
    return cuco::pair{_hash(i), T{i}};
  }

 private:
  Hasher _hash;
};

/**
 * @brief Device output transform functor to construct `size_type` with `cuco::pair<hash_value_type,
 * lhs_index_type>` or `cuco::pair<hash_value_type, rhs_index_type>`
 */
struct output_fn {
  __device__ constexpr cudf::size_type operator()(
    cuco::pair<hash_value_type, lhs_index_type> const& x) const
  {
    return static_cast<cudf::size_type>(x.second);
  }
  __device__ constexpr cudf::size_type operator()(
    cuco::pair<hash_value_type, rhs_index_type> const& x) const
  {
    return static_cast<cudf::size_type>(x.second);
  }
};
}  // namespace

template <cudf::has_nested HasNested>
distinct_hash_join<HasNested>::distinct_hash_join(cudf::table_view const& build,
                                                  cudf::table_view const& probe,
                                                  bool has_nulls,
                                                  cudf::null_equality compare_nulls,
                                                  rmm::cuda_stream_view stream)
  : _has_nulls{has_nulls},
    _nulls_equal{compare_nulls},
    _build{build},
    _probe{probe},
    _preprocessed_build{
      cudf::experimental::row::equality::preprocessed_table::create(_build, stream)},
    _preprocessed_probe{
      cudf::experimental::row::equality::preprocessed_table::create(_probe, stream)},
    _hash_table{build.num_rows(),
                CUCO_DESIRED_LOAD_FACTOR,
                cuco::empty_key{cuco::pair{std::numeric_limits<hash_value_type>::max(),
                                           lhs_index_type{JoinNoneValue}}},
                prepare_device_equal<HasNested>(
                  _preprocessed_build, _preprocessed_probe, has_nulls, compare_nulls),
                {},
                cuco::thread_scope_device,
                cuco_storage_type{},
                cudf::detail::cuco_allocator{stream},
                stream.value()}
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(0 != this->_build.num_columns(), "Hash join build table is empty");

  if (this->_build.num_rows() == 0) { return; }

  auto const row_hasher = experimental::row::hash::row_hasher{this->_preprocessed_build};
  auto const d_hasher   = row_hasher.device_hasher(nullate::DYNAMIC{this->_has_nulls});

  auto const iter = cudf::detail::make_counting_transform_iterator(
    0, build_keys_fn<decltype(d_hasher), lhs_index_type>{d_hasher});

  size_type const build_table_num_rows{build.num_rows()};
  if (this->_nulls_equal == cudf::null_equality::EQUAL or (not cudf::nullable(this->_build))) {
    this->_hash_table.insert_async(iter, iter + build_table_num_rows, stream.value());
  } else {
    auto stencil = thrust::counting_iterator<size_type>{0};
    auto const row_bitmask =
      cudf::detail::bitmask_and(this->_build, stream, rmm::mr::get_current_device_resource()).first;
    auto const pred =
      cudf::detail::row_is_valid{reinterpret_cast<bitmask_type const*>(row_bitmask.data())};

    // insert valid rows
    this->_hash_table.insert_if_async(
      iter, iter + build_table_num_rows, stencil, pred, stream.value());
  }
}

template <cudf::has_nested HasNested>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
distinct_hash_join<HasNested>::inner_join(rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr) const
{
  cudf::scoped_range range{"distinct_hash_join::inner_join"};

  size_type const probe_table_num_rows{this->_probe.num_rows()};

  // If output size is zero, return immediately
  if (probe_table_num_rows == 0) {
    return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                     std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
  }

  auto build_indices =
    std::make_unique<rmm::device_uvector<size_type>>(probe_table_num_rows, stream, mr);
  auto probe_indices =
    std::make_unique<rmm::device_uvector<size_type>>(probe_table_num_rows, stream, mr);

  auto const probe_row_hasher =
    cudf::experimental::row::hash::row_hasher{this->_preprocessed_probe};
  auto const d_probe_hasher = probe_row_hasher.device_hasher(nullate::DYNAMIC{this->_has_nulls});
  auto const iter           = cudf::detail::make_counting_transform_iterator(
    0, build_keys_fn<decltype(d_probe_hasher), rhs_index_type>{d_probe_hasher});

  auto const build_indices_begin =
    thrust::make_transform_output_iterator(build_indices->begin(), output_fn{});
  auto const probe_indices_begin =
    thrust::make_transform_output_iterator(probe_indices->begin(), output_fn{});

  auto const [probe_indices_end, _] = this->_hash_table.retrieve(
    iter, iter + probe_table_num_rows, probe_indices_begin, build_indices_begin, stream.value());

  auto const actual_size = std::distance(probe_indices_begin, probe_indices_end);
  build_indices->resize(actual_size, stream);
  probe_indices->resize(actual_size, stream);

  return {std::move(build_indices), std::move(probe_indices)};
}

template <cudf::has_nested HasNested>
std::unique_ptr<rmm::device_uvector<size_type>> distinct_hash_join<HasNested>::left_join(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  cudf::scoped_range range{"distinct_hash_join::left_join"};

  size_type const probe_table_num_rows{this->_probe.num_rows()};

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
    auto const probe_row_hasher =
      cudf::experimental::row::hash::row_hasher{this->_preprocessed_probe};
    auto const d_probe_hasher = probe_row_hasher.device_hasher(nullate::DYNAMIC{this->_has_nulls});
    auto const iter           = cudf::detail::make_counting_transform_iterator(
      0, build_keys_fn<decltype(d_probe_hasher), rhs_index_type>{d_probe_hasher});

    auto const output_begin =
      thrust::make_transform_output_iterator(build_indices->begin(), output_fn{});
    // TODO conditional find for nulls once `cuco::static_set::find_if` is added
    this->_hash_table.find_async(iter, iter + probe_table_num_rows, output_begin, stream.value());
  }

  return build_indices;
}
}  // namespace detail

template <>
distinct_hash_join<cudf::has_nested::YES>::~distinct_hash_join() = default;

template <>
distinct_hash_join<cudf::has_nested::NO>::~distinct_hash_join() = default;

template <>
distinct_hash_join<cudf::has_nested::YES>::distinct_hash_join(cudf::table_view const& build,
                                                              cudf::table_view const& probe,
                                                              nullable_join has_nulls,
                                                              null_equality compare_nulls,
                                                              rmm::cuda_stream_view stream)
  : _impl{std::make_unique<impl_type>(
      build, probe, has_nulls == nullable_join::YES, compare_nulls, stream)}
{
}

template <>
distinct_hash_join<cudf::has_nested::NO>::distinct_hash_join(cudf::table_view const& build,
                                                             cudf::table_view const& probe,
                                                             nullable_join has_nulls,
                                                             null_equality compare_nulls,
                                                             rmm::cuda_stream_view stream)
  : _impl{std::make_unique<impl_type>(
      build, probe, has_nulls == nullable_join::YES, compare_nulls, stream)}
{
}

template <>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
distinct_hash_join<cudf::has_nested::YES>::inner_join(rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr) const
{
  return _impl->inner_join(stream, mr);
}

template <>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
distinct_hash_join<cudf::has_nested::NO>::inner_join(rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr) const
{
  return _impl->inner_join(stream, mr);
}

template <>
std::unique_ptr<rmm::device_uvector<size_type>>
distinct_hash_join<cudf::has_nested::YES>::left_join(rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr) const
{
  return _impl->left_join(stream, mr);
}

template <>
std::unique_ptr<rmm::device_uvector<size_type>> distinct_hash_join<cudf::has_nested::NO>::left_join(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  return _impl->left_join(stream, mr);
}
}  // namespace cudf
