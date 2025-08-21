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
#pragma once

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/join/join.hpp>
#include <cudf/table/experimental/row_operators.cuh>
#include <cudf/table/primitive_row_operators.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cuco/bucket_storage.cuh>
#include <cuco/extent.cuh>
#include <cuda/std/type_traits>
#include <cuco/types.cuh>
#include <cuco/static_set_ref.cuh>
#include <cuco/static_multiset_ref.cuh>

// Forward declaration
namespace cudf::experimental::row::equality {
class preprocessed_table;
}

namespace CUDF_EXPORT cudf {
namespace detail {

using cudf::experimental::row::lhs_index_type;
using cudf::experimental::row::rhs_index_type;

class filtered_join {
 public:
  struct build_properties {
    bool _has_nulls;                   ///< True if nested nulls are present in build table
    bool _has_floating_point;
    bool _has_nested_columns;
  };

  struct insertion_adapter {
    __device__ constexpr bool operator()(
      cuco::pair<hash_value_type, lhs_index_type> const&,
      cuco::pair<hash_value_type, lhs_index_type> const&) const noexcept
    {
      return false;
    }
  };

  struct hasher_adapter {
    template <typename T>
    __device__ constexpr hash_value_type operator()(
      cuco::pair<hash_value_type, T> const& key) const noexcept
    {
      return key.first;
    }
  };

  template <typename T, typename Hasher>
  struct keys_adapter {
    CUDF_HOST_DEVICE constexpr keys_adapter(Hasher const& hasher) : _hasher{hasher} {}

    __device__ __forceinline__ auto operator()(size_type i) const noexcept
    {
      return cuco::pair{_hasher(i), T{i}};
    }

   private:
    Hasher _hasher;
  };

  template <typename Equal>
  struct comparator_adapter {
    comparator_adapter(Equal const& d_equal) : _d_equal{d_equal} {}

    __device__ constexpr auto operator()(
      cuco::pair<hash_value_type, lhs_index_type> const& lhs,
      cuco::pair<hash_value_type, rhs_index_type> const& rhs) const noexcept
    {
      if (lhs.first != rhs.first) { return false; }
      return _d_equal(lhs.second, rhs.second);
    }

    __device__ constexpr auto operator()(
      cuco::pair<hash_value_type, rhs_index_type> const& rhs,
      cuco::pair<hash_value_type, lhs_index_type> const& lhs) const noexcept
    {
      if (lhs.first != rhs.first) { return false; }
      return _d_equal(lhs.second, rhs.second);
    }

   private:
    Equal _d_equal;
  };

  struct output_adapter {
    __device__ constexpr cudf::size_type operator()(
      cuco::pair<hash_value_type, lhs_index_type> const& x) const
    {
      return static_cast<cudf::size_type>(x.second);
    }
  };

  using key = cuco::pair<hash_value_type, lhs_index_type>;
  using storage_type =
    cuco::bucket_storage<key,
                         1,  /// fixing bucket size to be 1 i.e each thread handles one slot
                         cuco::extent<cudf::size_type>,
                         cudf::detail::cuco_allocator<char>>;

  using primitive_row_hasher =
    cudf::row::primitive::row_hasher<cudf::hashing::detail::default_hash>;
  using primitive_probing_scheme = cuco::linear_probing<1, hasher_adapter>;
  using primitive_row_comparator = cudf::row::primitive::row_equality_comparator;

  using row_hasher =
    cudf::experimental::row::hash::device_row_hasher<cudf::hashing::detail::default_hash,
                                                     nullate::DYNAMIC>;
  using nested_probing_scheme = cuco::linear_probing<4, hasher_adapter>;
  using simple_probing_scheme = cuco::linear_probing<1, hasher_adapter>;
  using row_comparator        = cudf::experimental::row::equality::device_row_comparator<
           true,
           cudf::nullate::DYNAMIC,
           cudf::experimental::row::equality::nan_equal_physical_equality_comparator>;


  storage_type _bucket_storage;
  static constexpr auto empty_sentinel_key = cuco::empty_key{
    cuco::pair{std::numeric_limits<hash_value_type>::max(), lhs_index_type{JoinNoneValue}}};
  build_properties _build_props;
  cudf::table_view _build;                 ///< input table to build the hash map
  cudf::null_equality const _nulls_equal;  ///< whether to consider nulls as equal
  std::shared_ptr<cudf::experimental::row::equality::preprocessed_table>
    _preprocessed_build;  ///< input table preprocssed for row operators

 public:
  auto compute_bucket_storage_size(cudf::table_view tbl, double load_factor);

 public:
  filtered_join(cudf::table_view const& build,
                cudf::null_equality compare_nulls,
                double load_factor,
                rmm::cuda_stream_view stream);

  template <int32_t CGSize, typename Ref>
  void insert_build_table(Ref const &insert_ref, rmm::cuda_stream_view stream);

  virtual std::unique_ptr<rmm::device_uvector<cudf::size_type>> semi_join(
    cudf::table_view const& probe, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) = 0;
};

class filtered_join_with_multiset : public filtered_join {
 public:
  filtered_join_with_multiset(cudf::table_view const& build,
                        cudf::null_equality compare_nulls,
                        double load_factor,
                        rmm::cuda_stream_view stream);

  std::unique_ptr<rmm::device_uvector<cudf::size_type>> semi_join(
    cudf::table_view const& probe, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) override;
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> anti_join(
    cudf::table_view const& probe, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);
  template <int32_t CGSize, typename Ref>
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> query_build_table(
      cudf::table_view const &probe,
      std::shared_ptr<cudf::experimental::row::equality::preprocessed_table> preprocessed_probe,
      join_kind kind,
      Ref query_ref,
      rmm::cuda_stream_view stream, 
      rmm::device_async_resource_ref mr);
};

class filtered_join_with_set : public filtered_join {
 public:
  filtered_join_with_set(cudf::table_view const& build,
                        cudf::null_equality compare_nulls,
                        double load_factor,
                        rmm::cuda_stream_view stream);

  std::unique_ptr<rmm::device_uvector<cudf::size_type>> semi_join(
    cudf::table_view const& probe, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> anti_join(
    cudf::table_view const& probe, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);
  template <int32_t CGSize, typename Ref>
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> query_build_table(
      cudf::table_view const &probe,
      std::shared_ptr<cudf::experimental::row::equality::preprocessed_table> preprocessed_probe,
      join_kind kind,
      Ref query_ref,
      rmm::cuda_stream_view stream, 
      rmm::device_async_resource_ref mr);
};

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
