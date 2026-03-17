/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../join_common_utils.cuh"
#include "common.cuh"

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/mr/polymorphic_allocator.hpp>

#include <limits>
#include <memory>

namespace cudf::detail {

void build_hash_join(
  cudf::table_view const& build,
  std::shared_ptr<detail::row::equality::preprocessed_table> const& preprocessed_build,
  cudf::detail::hash_table_t& hash_table,
  bool has_nested_nulls,
  null_equality nulls_equal,
  [[maybe_unused]] bitmask_type const* bitmask,
  rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(0 != build.num_columns(), "Selected build dataset is empty", std::invalid_argument);
  CUDF_EXPECTS(0 != build.num_rows(), "Build side table has no rows", std::invalid_argument);

  auto insert_rows = [&](auto const& build, auto const& d_hasher) {
    auto const iter = cudf::detail::make_counting_transform_iterator(0, pair_fn{d_hasher});

    if (nulls_equal == cudf::null_equality::EQUAL or not nullable(build)) {
      hash_table.insert(iter, iter + build.num_rows(), stream.value());
    } else {
      auto const stencil = thrust::counting_iterator<size_type>{0};
      auto const pred    = row_is_valid{bitmask};

      hash_table.insert_if(iter, iter + build.num_rows(), stencil, pred, stream.value());
    }
  };

  auto const nulls = nullate::DYNAMIC{has_nested_nulls};

  if (cudf::detail::is_primitive_row_op_compatible(build)) {
    auto const d_hasher = cudf::detail::row::primitive::row_hasher{nulls, preprocessed_build};

    insert_rows(build, d_hasher);
  } else {
    auto const row_hash = detail::row::hash::row_hasher{preprocessed_build};
    auto const d_hasher = row_hash.device_hasher(nulls);

    insert_rows(build, d_hasher);
  }
}

template <typename Hasher>
hash_join<Hasher>::hash_join(cudf::table_view const& build,
                             bool has_nulls,
                             cudf::null_equality compare_nulls,
                             rmm::cuda_stream_view stream)
  : hash_join{build, has_nulls, compare_nulls, CUCO_DESIRED_LOAD_FACTOR, stream}
{
}

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

using hasher_type = cudf::hashing::detail::MurmurHash3_x86_32<cudf::hash_value_type>;

template hash_join<hasher_type>::hash_join(cudf::table_view const& build,
                                           bool has_nulls,
                                           cudf::null_equality compare_nulls,
                                           rmm::cuda_stream_view stream);

template hash_join<hasher_type>::hash_join(cudf::table_view const& build,
                                           bool has_nulls,
                                           cudf::null_equality compare_nulls,
                                           double load_factor,
                                           rmm::cuda_stream_view stream);

}  // namespace cudf::detail
