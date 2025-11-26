/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "join_common_utils.cuh"

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/join/distinct_hash_join.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/row_operator/primitive_row_operators.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/join/distinct_hash_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cooperative_groups.h>
#include <cub/block/block_scan.cuh>
#include <cuco/static_set.cuh>
#include <cuda/std/tuple>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sequence.h>

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
class primitive_keys_fn {
  using hasher = cudf::detail::row::primitive::row_hasher<>;

 public:
  CUDF_HOST_DEVICE constexpr primitive_keys_fn(hasher const& hash) : _hash{hash} {}

  __device__ __forceinline__ auto operator()(size_type i) const noexcept
  {
    return cuco::pair{_hash(i), T{i}};
  }

 private:
  hasher _hash;
};

/**
 * @brief Device functor to create a pair of {hash_value, row_index} for a given row.
 */
template <typename T>
class build_keys_fn {
  using hasher = cudf::detail::row::hash::device_row_hasher<cudf::hashing::detail::default_hash,
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

/**
 * @brief Find matching rows in the hash table
 *
 * @tparam IterType Type of the iterator over hash values
 * @tparam EqualType Type of the equality comparator
 * @param hash_table The hash table to search in
 * @param iter Iterator over hash values
 * @param d_equal Equality comparator
 * @param probe The probe table
 * @param hasher Hash function
 * @param nulls_equal Null equality setting
 * @param found_begin Output iterator for found indices
 * @param stream CUDA stream
 */
template <typename HashTableType,
          typename IterType,
          typename EqualType,
          typename Hasher,
          typename FoundIterator>
void find_matches_in_hash_table(HashTableType const& hash_table,
                                IterType iter,
                                EqualType const& d_equal,
                                cudf::table_view const& probe,
                                Hasher hasher,
                                cudf::null_equality nulls_equal,
                                FoundIterator found_begin,
                                rmm::cuda_stream_view stream)
{
  auto const probe_table_num_rows = probe.num_rows();
  // If `idx` is within the range `[0, probe_table_num_rows)` and `found_indices[idx]` is not
  // equal to `cudf::JoinNoMatch`, then `idx` has a match in the hash set.
  if (nulls_equal == cudf::null_equality::EQUAL or (not cudf::nullable(probe))) {
    hash_table.find_async(
      iter, iter + probe_table_num_rows, d_equal, hasher, found_begin, stream.value());
  } else {
    auto stencil = thrust::counting_iterator<size_type>{0};
    auto const row_bitmask =
      cudf::detail::bitmask_and(probe, stream, cudf::get_current_device_resource_ref()).first;
    auto const pred =
      cudf::detail::row_is_valid{reinterpret_cast<bitmask_type const*>(row_bitmask.data())};

    hash_table.find_if_async(iter,
                             iter + probe_table_num_rows,
                             stencil,
                             pred,
                             d_equal,
                             hasher,
                             found_begin,
                             stream.value());
  }
}

}  // namespace

distinct_hash_join::distinct_hash_join(cudf::table_view const& build,
                                       cudf::null_equality compare_nulls,
                                       rmm::cuda_stream_view stream)
  : distinct_hash_join{build, compare_nulls, CUCO_DESIRED_LOAD_FACTOR, stream}
{
}

distinct_hash_join::distinct_hash_join(cudf::table_view const& build,
                                       cudf::null_equality compare_nulls,
                                       double load_factor,
                                       rmm::cuda_stream_view stream)
  : _has_nested_columns{cudf::has_nested_columns(build)},
    _nulls_equal{compare_nulls},
    _build{build},
    _preprocessed_build{cudf::detail::row::equality::preprocessed_table::create(_build, stream)},
    _hash_table{cuco::extent{static_cast<std::size_t>(build.num_rows())},
                load_factor,
                cuco::empty_key{cuco::pair{std::numeric_limits<hash_value_type>::max(),
                                           rhs_index_type{cudf::JoinNoMatch}}},
                always_not_equal{},
                {},
                cuco::thread_scope_device,
                cuco_storage_type{},
                rmm::mr::polymorphic_allocator<char>{},
                stream.value()}
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(0 != this->_build.num_columns(), "Hash join build table is empty");
  CUDF_EXPECTS(load_factor > 0 && load_factor <= 1,
               "Invalid load factor: must be greater than 0 and less than or equal to 1.",
               std::invalid_argument);

  size_type const build_table_num_rows{_build.num_rows()};

  if (build_table_num_rows == 0) { return; }

  auto const build_hash_table = [&](auto iter) {
    if (this->_nulls_equal == cudf::null_equality::EQUAL or (not cudf::nullable(build))) {
      this->_hash_table.insert_async(iter, iter + build_table_num_rows, stream.value());
    } else {
      auto stencil = thrust::counting_iterator<size_type>{0};
      auto const row_bitmask =
        cudf::detail::bitmask_and(_build, stream, cudf::get_current_device_resource_ref()).first;
      auto const pred =
        cudf::detail::row_is_valid{reinterpret_cast<bitmask_type const*>(row_bitmask.data())};

      // insert valid rows
      this->_hash_table.insert_if_async(
        iter, iter + build_table_num_rows, stencil, pred, stream.value());
    }
  };

  if (cudf::detail::is_primitive_row_op_compatible(_build)) {
    auto const d_hasher = cudf::detail::row::primitive::row_hasher{nullate::DYNAMIC{has_nulls},
                                                                   this->_preprocessed_build};

    auto const iter = cudf::detail::make_counting_transform_iterator(
      0, primitive_keys_fn<rhs_index_type>{d_hasher});

    build_hash_table(iter);
  } else {
    auto const row_hasher = detail::row::hash::row_hasher{this->_preprocessed_build};
    auto const d_hasher   = row_hasher.device_hasher(nullate::DYNAMIC{has_nulls});

    auto const iter =
      cudf::detail::make_counting_transform_iterator(0, build_keys_fn<rhs_index_type>{d_hasher});

    build_hash_table(iter);
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

  auto build_indices =
    std::make_unique<rmm::device_uvector<size_type>>(probe_table_num_rows, stream, mr);
  auto probe_indices =
    std::make_unique<rmm::device_uvector<size_type>>(probe_table_num_rows, stream, mr);

  auto found_indices = rmm::device_uvector<size_type>(probe_table_num_rows, stream);
  auto const found_begin =
    thrust::make_transform_output_iterator(found_indices.begin(), output_fn{});

  auto preprocessed_probe = cudf::detail::row::equality::preprocessed_table::create(probe, stream);
  if (cudf::detail::is_primitive_row_op_compatible(_build)) {
    auto const d_hasher =
      cudf::detail::row::primitive::row_hasher{nullate::DYNAMIC{has_nulls}, preprocessed_probe};
    auto const d_equal = cudf::detail::row::primitive::row_equality_comparator{
      nullate::DYNAMIC{has_nulls}, preprocessed_probe, _preprocessed_build, _nulls_equal};
    auto const iter = cudf::detail::make_counting_transform_iterator(
      0, primitive_keys_fn<lhs_index_type>{d_hasher});

    find_matches_in_hash_table(this->_hash_table,
                               iter,
                               primitive_comparator_adapter{d_equal},
                               probe,
                               hasher{},
                               _nulls_equal,
                               found_begin,
                               stream);
  } else {
    auto const two_table_equal =
      cudf::detail::row::equality::two_table_comparator(preprocessed_probe, _preprocessed_build);

    auto const probe_row_hasher = cudf::detail::row::hash::row_hasher{preprocessed_probe};
    auto const d_probe_hasher   = probe_row_hasher.device_hasher(nullate::DYNAMIC{has_nulls});
    auto const iter             = cudf::detail::make_counting_transform_iterator(
      0, build_keys_fn<lhs_index_type>{d_probe_hasher});

    if (_has_nested_columns) {
      auto const device_comparator =
        two_table_equal.equal_to<true>(nullate::DYNAMIC{has_nulls}, _nulls_equal);
      find_matches_in_hash_table(this->_hash_table,
                                 iter,
                                 comparator_adapter{device_comparator},
                                 probe,
                                 hasher{},
                                 _nulls_equal,
                                 found_begin,
                                 stream);
    } else {
      auto const device_comparator =
        two_table_equal.equal_to<false>(nullate::DYNAMIC{has_nulls}, _nulls_equal);
      find_matches_in_hash_table(this->_hash_table,
                                 iter,
                                 comparator_adapter{device_comparator},
                                 probe,
                                 hasher{},
                                 _nulls_equal,
                                 found_begin,
                                 stream);
    }
  }

  auto const tuple_iter = cudf::detail::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<cuda::std::tuple<size_type, size_type>>(
      [found_iter = found_indices.begin()] __device__(size_type idx) {
        return cuda::std::tuple{*(found_iter + idx), idx};
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
                      [] __device__(size_type idx) { return idx != cudf::JoinNoMatch; }));
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
  auto const output_begin =
    thrust::make_transform_output_iterator(build_indices->begin(), output_fn{});

  auto preprocessed_probe = cudf::detail::row::equality::preprocessed_table::create(probe, stream);

  if (cudf::detail::is_primitive_row_op_compatible(_build)) {
    auto const d_hasher =
      cudf::detail::row::primitive::row_hasher{nullate::DYNAMIC{has_nulls}, preprocessed_probe};
    auto const d_equal = cudf::detail::row::primitive::row_equality_comparator{
      nullate::DYNAMIC{has_nulls}, preprocessed_probe, _preprocessed_build, _nulls_equal};

    auto const iter = cudf::detail::make_counting_transform_iterator(
      0, primitive_keys_fn<lhs_index_type>{d_hasher});

    find_matches_in_hash_table(this->_hash_table,
                               iter,
                               primitive_comparator_adapter{d_equal},
                               probe,
                               hasher{},
                               _nulls_equal,
                               output_begin,
                               stream);
  } else {
    // If build table is empty, return probe table
    if (this->_build.num_rows() == 0) {
      thrust::fill(rmm::exec_policy_nosync(stream),
                   build_indices->begin(),
                   build_indices->end(),
                   cudf::JoinNoMatch);
    } else {
      auto const two_table_equal =
        cudf::detail::row::equality::two_table_comparator(preprocessed_probe, _preprocessed_build);

      auto const probe_row_hasher = cudf::detail::row::hash::row_hasher{preprocessed_probe};
      auto const d_probe_hasher   = probe_row_hasher.device_hasher(nullate::DYNAMIC{has_nulls});
      auto const iter             = cudf::detail::make_counting_transform_iterator(
        0, build_keys_fn<lhs_index_type>{d_probe_hasher});

      if (_has_nested_columns) {
        auto const device_comparator =
          two_table_equal.equal_to<true>(nullate::DYNAMIC{has_nulls}, _nulls_equal);
        find_matches_in_hash_table(this->_hash_table,
                                   iter,
                                   comparator_adapter{device_comparator},
                                   probe,
                                   hasher{},
                                   _nulls_equal,
                                   output_begin,
                                   stream);
      } else {
        auto const device_comparator =
          two_table_equal.equal_to<false>(nullate::DYNAMIC{has_nulls}, _nulls_equal);
        find_matches_in_hash_table(this->_hash_table,
                                   iter,
                                   comparator_adapter{device_comparator},
                                   probe,
                                   hasher{},
                                   _nulls_equal,
                                   output_begin,
                                   stream);
      }
    }
  }
  return build_indices;
}
}  // namespace detail

distinct_hash_join::~distinct_hash_join() = default;

distinct_hash_join::distinct_hash_join(cudf::table_view const& build,
                                       null_equality compare_nulls,
                                       double load_factor,
                                       rmm::cuda_stream_view stream)
  : _impl{std::make_unique<impl_type>(build, compare_nulls, load_factor, stream)}
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
