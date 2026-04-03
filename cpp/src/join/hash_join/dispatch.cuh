/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "common.cuh"

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/row_operator/primitive_row_operators.cuh>

#include <cuco/pair.cuh>

#include <memory>
#include <utility>

namespace cudf::detail {

/**
 * @brief Equality comparator for cuco hash table probing with row-level equality.
 */
template <typename Equal>
class pair_equal {
 public:
  pair_equal(Equal check_row_equality) : _check_row_equality{std::move(check_row_equality)} {}

  __device__ __forceinline__ bool operator()(
    cuco::pair<hash_value_type, size_type> const& lhs,
    cuco::pair<hash_value_type, size_type> const& rhs) const noexcept
  {
    using detail::row::lhs_index_type;
    using detail::row::rhs_index_type;

    return lhs.first == rhs.first and
           _check_row_equality(lhs_index_type{lhs.second}, rhs_index_type{rhs.second});
  }

 private:
  Equal _check_row_equality;
};

/**
 * @brief Extracts the build-side row index from a cuco hash table slot.
 */
struct output_fn {
  __device__ constexpr cudf::size_type operator()(
    cuco::pair<hash_value_type, cudf::size_type> const& slot) const
  {
    return slot.second;
  }
};

/**
 * @brief Equality comparator for cuco hash table probing with primitive row equality.
 */
class primitive_pair_equal {
 public:
  primitive_pair_equal(cudf::detail::row::primitive::row_equality_comparator check_row_equality)
    : _check_row_equality{std::move(check_row_equality)}
  {
  }

  __device__ __forceinline__ bool operator()(
    cuco::pair<hash_value_type, size_type> const& lhs,
    cuco::pair<hash_value_type, size_type> const& rhs) const noexcept
  {
    return lhs.first == rhs.first and _check_row_equality(lhs.second, rhs.second);
  }

 private:
  cudf::detail::row::primitive::row_equality_comparator _check_row_equality;
};

template <typename Fn>
decltype(auto) dispatch_join_comparator(
  table_view const& build_table,
  table_view const& probe_table,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_build,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_probe,
  bool has_nulls,
  null_equality compare_nulls,
  Fn&& fn)
{
  auto const probe_nulls = cudf::nullate::DYNAMIC{has_nulls};

  if (cudf::detail::is_primitive_row_op_compatible(build_table)) {
    auto const d_hasher = cudf::detail::row::primitive::row_hasher{probe_nulls, preprocessed_probe};
    auto const d_equal  = cudf::detail::row::primitive::row_equality_comparator{
      probe_nulls, preprocessed_probe, preprocessed_build, compare_nulls};
    return std::forward<Fn>(fn)(primitive_pair_equal{d_equal}, d_hasher);
  }

  auto const d_hasher =
    cudf::detail::row::hash::row_hasher{preprocessed_probe}.device_hasher(probe_nulls);
  auto const row_comparator =
    cudf::detail::row::equality::two_table_comparator{preprocessed_probe, preprocessed_build};

  if (cudf::detail::has_nested_columns(probe_table)) {
    auto const d_equal = row_comparator.equal_to<true>(probe_nulls, compare_nulls);
    return std::forward<Fn>(fn)(pair_equal{d_equal}, d_hasher);
  }

  auto const d_equal = row_comparator.equal_to<false>(probe_nulls, compare_nulls);
  return std::forward<Fn>(fn)(pair_equal{d_equal}, d_hasher);
}

}  // namespace cudf::detail
