/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/join/join.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/row_operator/primitive_row_operators.cuh>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/bucket_storage.cuh>
#include <cuco/extent.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/pair.cuh>
#include <cuco/probing_scheme.cuh>
#include <cuco/types.cuh>
#include <cuda/std/limits>

#include <memory>

namespace cudf::detail {

using cudf::detail::row::lhs_index_type;
using cudf::detail::row::rhs_index_type;

static constexpr hash_value_type mark_bit_mask = hash_value_type{1}
                                                 << (sizeof(hash_value_type) * 8 - 1);

CUDF_HOST_DEVICE constexpr hash_value_type set_mark(hash_value_type value) noexcept
{
  return value | mark_bit_mask;
}

CUDF_HOST_DEVICE constexpr hash_value_type unset_mark(hash_value_type value) noexcept
{
  return value & ~mark_bit_mask;
}

CUDF_HOST_DEVICE constexpr bool is_marked(hash_value_type value) noexcept
{
  return (value & mark_bit_mask) != 0;
}

struct masked_hash_fn {
  template <typename T>
  __device__ constexpr hash_value_type operator()(
    cuco::pair<hash_value_type, T> const& key) const noexcept
  {
    return unset_mark(key.first);
  }
};

struct secondary_hash_fn {
  uint32_t _seed{0};

  CUDF_HOST_DEVICE secondary_hash_fn() = default;
  CUDF_HOST_DEVICE secondary_hash_fn(uint32_t seed) : _seed{seed} {}

  template <typename T>
  CUDF_HOST_DEVICE auto operator()(cuco::pair<hash_value_type, T> const& key) const noexcept
  {
    return cuco::xxhash_32<hash_value_type>{_seed}(unset_mark(key.first));
  }
};

template <typename T, typename Hasher>
struct masked_key_fn {
  CUDF_HOST_DEVICE constexpr masked_key_fn(Hasher const& hasher) : _hasher{hasher} {}

  __device__ __forceinline__ auto operator()(size_type i) const noexcept
  {
    return cuco::pair{unset_mark(_hasher(i)), T{i}};
  }

 private:
  Hasher _hasher;
};

template <typename IndexType>
struct hash_pair_fn {
  hash_value_type const* _hashes;

  __device__ __forceinline__ auto operator()(size_type i) const noexcept
  {
    return cuco::pair{_hashes[i], IndexType{i}};
  }
};

template <typename Equal>
struct masked_comparator_fn {
  masked_comparator_fn(Equal const& d_equal) : _d_equal{d_equal} {}

  __device__ constexpr auto operator()(
    cuco::pair<hash_value_type, lhs_index_type> const& lhs,
    cuco::pair<hash_value_type, lhs_index_type> const& rhs) const noexcept
  {
    if (unset_mark(lhs.first) != unset_mark(rhs.first)) { return false; }
    return _d_equal(lhs.second, rhs.second);
  }

  __device__ constexpr auto operator()(
    cuco::pair<hash_value_type, rhs_index_type> const& probe,
    cuco::pair<hash_value_type, lhs_index_type> const& build) const noexcept
  {
    if (unset_mark(probe.first) != unset_mark(build.first)) { return false; }
    return _d_equal(build.second, probe.second);
  }

 private:
  Equal _d_equal;
};

template <typename T>
struct insertion_adapter {
  insertion_adapter(T const&) {}
  __device__ constexpr bool operator()(
    cuco::pair<hash_value_type, lhs_index_type> const&,
    cuco::pair<hash_value_type, lhs_index_type> const&) const noexcept
  {
    return false;
  }
};

using masked_probing_scheme = cuco::double_hashing<1, masked_hash_fn, secondary_hash_fn>;

using mark_key_type = cuco::pair<hash_value_type, lhs_index_type>;

using mark_storage_type = cuco::
  bucket_storage<mark_key_type, 1, cuco::extent<std::size_t>, rmm::mr::polymorphic_allocator<char>>;

static constexpr auto masked_empty_sentinel =
  cuco::empty_key{cuco::pair{unset_mark(cuda::std::numeric_limits<hash_value_type>::max()),
                             lhs_index_type{cudf::JoinNoMatch}}};

class mark_join {
 public:
  mark_join(cudf::table_view const& build,
            cudf::null_equality compare_nulls,
            double load_factor,
            rmm::cuda_stream_view stream);

  std::unique_ptr<rmm::device_uvector<cudf::size_type>> semi_join(
    cudf::table_view const& probe, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);

  std::unique_ptr<rmm::device_uvector<cudf::size_type>> anti_join(
    cudf::table_view const& probe, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);

 private:
  using primitive_row_hasher =
    cudf::detail::row::primitive::row_hasher<cudf::hashing::detail::default_hash>;
  using primitive_row_comparator = cudf::detail::row::primitive::row_equality_comparator;
  using row_hasher =
    cudf::detail::row::hash::device_row_hasher<cudf::hashing::detail::default_hash, nullate::YES>;

  bool _has_nested_columns;
  cudf::table_view _build;
  cudf::null_equality _nulls_equal;
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> _preprocessed_build;
  mark_storage_type _bucket_storage;
  cudf::size_type _num_build_inserted{0};

  [[nodiscard]] cudf::size_type num_build_inserted() const { return _num_build_inserted; }

  std::unique_ptr<rmm::device_uvector<cudf::size_type>> semi_anti_join(
    cudf::table_view const& probe,
    join_kind kind,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

  template <typename Comparator>
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> mark_probe_and_retrieve(
    cudf::table_view const& probe,
    std::shared_ptr<cudf::detail::row::equality::preprocessed_table> preprocessed_probe,
    join_kind kind,
    Comparator comparator,
    rmm::cuda_stream_view stream,
    rmm::device_async_resource_ref mr);

  void clear_marks(rmm::cuda_stream_view stream);
};

}  // namespace cudf::detail
