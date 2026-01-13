/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/hashing/detail/xxhash_64.cuh>
#include <cudf/reduction/approx_distinct_count.hpp>
#include <cudf/reduction/detail/approx_distinct_count.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/hyperloglog.cuh>
#include <cuco/hyperloglog_ref.cuh>
#include <cuda/functional>
#include <thrust/iterator/counting_iterator.h>

namespace cudf {
namespace detail {

namespace {

/**
 * @brief Device functor to check if a row is valid using a bitmask
 */
struct row_is_valid {
  bitmask_type const* row_bitmask;

  __device__ bool operator()(cudf::size_type row_idx) const noexcept
  {
    return cudf::bit_is_set(row_bitmask, row_idx);
  }
};

/**
 * @brief Device functor that returns NULL_HASH for rows containing null or NaN
 */
template <typename Hasher>
struct nan_to_null_hasher {
  Hasher base_hasher;
  table_device_view d_table;

  __device__ hash_value_type operator()(cudf::size_type row_idx) const noexcept
  {
    constexpr auto null_hash = cuda::std::numeric_limits<hash_value_type>::max();

    for (cudf::size_type col_idx = 0; col_idx < d_table.num_columns(); ++col_idx) {
      auto const& col = d_table.column(col_idx);

      if (col.nullable() && col.is_null_nocheck(row_idx)) { return null_hash; }

      auto const type = col.type();
      if (type.id() == type_id::FLOAT32) {
        if (cuda::std::isnan(col.element<float>(row_idx))) { return null_hash; }
      } else if (type.id() == type_id::FLOAT64) {
        if (cuda::std::isnan(col.element<double>(row_idx))) { return null_hash; }
      }
    }

    return base_hasher(row_idx);
  }
};

/**
 * @brief Predicate to check if a row has any NaN values (for use with bitmask filtering)
 */
struct check_nans_predicate {
  table_device_view d_table;
  bitmask_type const* row_bitmask;

  __device__ bool operator()(cudf::size_type row_idx) const noexcept
  {
    if (row_bitmask != nullptr && !cudf::bit_is_set(row_bitmask, row_idx)) { return false; }

    for (cudf::size_type col_idx = 0; col_idx < d_table.num_columns(); ++col_idx) {
      auto const& col = d_table.column(col_idx);
      auto const type = col.type();
      if (type.id() == type_id::FLOAT32) {
        if (cuda::std::isnan(col.element<float>(row_idx))) { return false; }
      } else if (type.id() == type_id::FLOAT64) {
        if (cuda::std::isnan(col.element<double>(row_idx))) { return false; }
      }
    }
    return true;
  }
};

}  // namespace

approx_distinct_count::~approx_distinct_count() = default;

approx_distinct_count::approx_distinct_count(table_view const& input,
                                             std::int32_t precision,
                                             null_policy null_handling,
                                             nan_policy nan_handling,
                                             rmm::cuda_stream_view stream)
  : _impl{cuco::precision{precision},
          cuda::std::identity{},
          rmm::mr::polymorphic_allocator<cuda::std::byte>{},
          stream},
    _null_handling{null_handling},
    _nan_handling{nan_handling}
{
  auto const num_rows = input.num_rows();
  if (num_rows == 0) { return; }

  add(input, stream);
}

approx_distinct_count::approx_distinct_count(cuda::std::span<cuda::std::byte> sketch_span,
                                             std::int32_t precision,
                                             null_policy null_handling,
                                             nan_policy nan_handling,
                                             rmm::cuda_stream_view stream)
  : _impl{cuco::precision{precision},
          cuda::std::identity{},
          rmm::mr::polymorphic_allocator<cuda::std::byte>{},
          stream},
    _null_handling{null_handling},
    _nan_handling{nan_handling}
{
  auto sketch_ref = hll_type::ref_type<>{sketch_span, cuda::std::identity{}};
  _impl.merge_async(sketch_ref, stream);
}

void approx_distinct_count::add(table_view const& input, rmm::cuda_stream_view stream)
{
  auto const num_rows = input.num_rows();
  if (num_rows == 0) { return; }

  auto const has_nulls = nullate::DYNAMIC{cudf::has_nested_nulls(input)};
  auto const preprocessed_input =
    cudf::detail::row::hash::preprocessed_table::create(input, stream);
  auto const row_hasher = cudf::detail::row::hash::row_hasher(preprocessed_input);
  auto const hash_key   = row_hasher.device_hasher<cudf::hashing::detail::XXHash_64>(has_nulls);

  if (_null_handling == null_policy::INCLUDE) {
    if (_nan_handling == nan_policy::NAN_IS_NULL) {
      // Include nulls and treat NaN as null - use custom hasher that maps NaN to NULL_HASH
      auto const d_table    = table_device_view::create(input, stream);
      auto const nan_hasher = nan_to_null_hasher{hash_key, *d_table};
      auto const hash_iter  = cudf::detail::make_counting_transform_iterator(0, nan_hasher);
      _impl.add_async(hash_iter, hash_iter + num_rows, stream);
    } else {
      auto const hash_iter = cudf::detail::make_counting_transform_iterator(0, hash_key);
      _impl.add_async(hash_iter, hash_iter + num_rows, stream);
    }
  } else {
    // Exclude nulls
    auto const hash_iter = cudf::detail::make_counting_transform_iterator(0, hash_key);
    auto const stencil   = thrust::counting_iterator{0};

    if (_nan_handling == nan_policy::NAN_IS_VALID) {
      if (!has_nulls) {
        _impl.add_async(hash_iter, hash_iter + num_rows, stream);
      } else {
        auto const row_bitmask =
          cudf::detail::bitmask_and(input, stream, cudf::get_current_device_resource_ref()).first;
        auto const pred = row_is_valid{static_cast<bitmask_type const*>(row_bitmask.data())};
        _impl.add_if_async(hash_iter, hash_iter + num_rows, stencil, pred, stream);
      }
    } else {
      auto const d_table = table_device_view::create(input, stream);
      if (!has_nulls) {
        auto const pred = check_nans_predicate{*d_table, nullptr};
        _impl.add_if_async(hash_iter, hash_iter + num_rows, stencil, pred, stream);
      } else {
        auto const row_bitmask =
          cudf::detail::bitmask_and(input, stream, cudf::get_current_device_resource_ref()).first;
        auto const bitmask_ptr = static_cast<bitmask_type const*>(row_bitmask.data());
        auto const pred        = check_nans_predicate{*d_table, bitmask_ptr};
        _impl.add_if_async(hash_iter, hash_iter + num_rows, stencil, pred, stream);
      }
    }
  }
}

void approx_distinct_count::merge(approx_distinct_count const& other, rmm::cuda_stream_view stream)
{
  // Validate policies match
  CUDF_EXPECTS(_null_handling == other._null_handling,
               "Cannot merge sketches with different null handling policies",
               std::invalid_argument);
  CUDF_EXPECTS(_nan_handling == other._nan_handling,
               "Cannot merge sketches with different NaN handling policies",
               std::invalid_argument);

  _impl.merge_async(other._impl, stream);
}

void approx_distinct_count::merge(cuda::std::span<cuda::std::byte> sketch_span,
                                  rmm::cuda_stream_view stream)
{
  auto other_ref = hll_type::ref_type<>{sketch_span, cuda::std::identity{}};
  _impl.merge_async(other_ref, stream);
}

std::size_t approx_distinct_count::estimate(rmm::cuda_stream_view stream) const
{
  return _impl.estimate(stream);
}

cuda::std::span<cuda::std::byte> approx_distinct_count::sketch() noexcept { return _impl.sketch(); }

null_policy approx_distinct_count::null_handling() const noexcept { return _null_handling; }

nan_policy approx_distinct_count::nan_handling() const noexcept { return _nan_handling; }

}  // namespace detail

approx_distinct_count::~approx_distinct_count() = default;

approx_distinct_count::approx_distinct_count(table_view const& input,
                                             std::int32_t precision,
                                             null_policy null_handling,
                                             nan_policy nan_handling,
                                             rmm::cuda_stream_view stream)
  : _impl(std::make_unique<impl_type>(input, precision, null_handling, nan_handling, stream))
{
}

approx_distinct_count::approx_distinct_count(cuda::std::span<cuda::std::byte> sketch_span,
                                             std::int32_t precision,
                                             null_policy null_handling,
                                             nan_policy nan_handling,
                                             rmm::cuda_stream_view stream)
  : _impl(std::make_unique<impl_type>(sketch_span, precision, null_handling, nan_handling, stream))
{
}

void approx_distinct_count::add(table_view const& input, rmm::cuda_stream_view stream)
{
  _impl->add(input, stream);
}

void approx_distinct_count::merge(approx_distinct_count const& other, rmm::cuda_stream_view stream)
{
  _impl->merge(*other._impl, stream);
}

void approx_distinct_count::merge(cuda::std::span<cuda::std::byte> sketch_span,
                                  rmm::cuda_stream_view stream)
{
  _impl->merge(sketch_span, stream);
}

std::size_t approx_distinct_count::estimate(rmm::cuda_stream_view stream) const
{
  return _impl->estimate(stream);
}

cuda::std::span<cuda::std::byte> approx_distinct_count::sketch() noexcept
{
  return _impl->sketch();
}

null_policy approx_distinct_count::null_handling() const noexcept { return _impl->null_handling(); }

nan_policy approx_distinct_count::nan_handling() const noexcept { return _impl->nan_handling(); }

}  // namespace cudf
