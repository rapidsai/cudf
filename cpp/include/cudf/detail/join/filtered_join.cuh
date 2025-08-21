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

/*
template<bool ReuseLeftTable>
struct filtered_join {
 public:
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
  using primitive_insert_ref_type = typename std::conditional_t<
    ReuseLeftTable,
    cuco::static_set_ref<key, cuco::thread_scope_device, insertion_adapter, primitive_probing_scheme, storage_type::ref_type, cuco::insert_tag>,
    cuco::static_multiset_ref<key, cuco::thread_scope_device, insertion_adapter, primitive_probing_scheme, storage_type::ref_type, cuco::insert_tag>>;

  using row_hasher =
    cudf::experimental::row::hash::device_row_hasher<cudf::hashing::detail::default_hash,
                                                     nullate::DYNAMIC>;
  using nested_probing_scheme = cuco::linear_probing<4, hasher_adapter>;
  using simple_probing_scheme = cuco::linear_probing<1, hasher_adapter>;
  using row_comparator        = cudf::experimental::row::equality::device_row_comparator<
           true,
           cudf::nullate::DYNAMIC,
           cudf::experimental::row::equality::nan_equal_physical_equality_comparator>;
  using nested_insert_ref_type = typename std::conditional_t<
    ReuseLeftTable,
    cuco::static_set_ref<key, cuco::thread_scope_device, insertion_adapter, nested_probing_scheme, storage_type::ref_type, cuco::insert_tag>,
    cuco::static_multiset_ref<key, cuco::thread_scope_device, insertion_adapter, nested_probing_scheme, storage_type::ref_type, cuco::insert_tag>>;
  using simple_insert_ref_type = typename std::conditional_t<
    ReuseLeftTable,
    cuco::static_set_ref<key, cuco::thread_scope_device, insertion_adapter, simple_probing_scheme, storage_type::ref_type, cuco::insert_tag>,
    cuco::static_multiset_ref<key, cuco::thread_scope_device, insertion_adapter, simple_probing_scheme, storage_type::ref_type, cuco::insert_tag>>;

  filtered_join()                                = delete;
  ~filtered_join()                               = default;
  filtered_join(filtered_join const&)            = delete;
  filtered_join(filtered_join&&)                 = delete;
  filtered_join& operator=(filtered_join const&) = delete;
  filtered_join& operator=(filtered_join&&)      = delete;

 private:
  build_properties _build_props;
  cudf::null_equality const _nulls_equal;  ///< whether to consider nulls as equal
  cudf::table_view _build;                 ///< input table to build the hash map
  std::shared_ptr<cudf::experimental::row::equality::preprocessed_table>
    _preprocessed_build;  ///< input table preprocssed for row operators
  storage_type _bucket_storage;
  static constexpr cuco::pair<hash_value_type, lhs_index_type> empty_sentinel_key = cuco::empty_key{
    cuco::pair{std::numeric_limits<hash_value_type>::max(), lhs_index_type{JoinNoneValue}}};

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
        bitmask_and(
          table_view{nullable_columns}, stream, cudf::get_current_device_resource_ref())
          .first;
      auto const row_bitmask_ptr = static_cast<bitmask_type const*>(row_bitmask.data());
      return std::pair(std::move(row_bitmask), row_bitmask_ptr);
    }

    return std::pair(rmm::device_buffer{0, stream}, nullable_columns.front().null_mask());
  }

  void insert_build_table(auto insert_ref, rmm::cuda_stream_view stream) {
    auto insert = [&]<typename Iterator>(Iterator build_iter) {
      // Build hash table by inserting all rows from build table
      auto const cg_size = insert_ref.probing_scheme().cg_size;
      auto const grid_size =
        cuco::detail::grid_size(_build.num_rows(), cg_size);

      if (_build_props._has_nulls && _nulls_equal == null_equality::UNEQUAL) {
        auto const bitmask_buffer_and_ptr = build_row_bitmask(_build, stream);
        auto const row_bitmask_ptr        = bitmask_buffer_and_ptr.second;
        cuco::detail::open_addressing_ns::insert_if_n<cg_size,
                                                      cuco::detail::default_block_size()>
          <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
            build_iter,
            _build.num_rows(),
            thrust::counting_iterator<size_type>{0},
            row_is_valid{row_bitmask_ptr},
            insert_ref);
      } else {
        cuco::detail::open_addressing_ns::insert_if_n<cg_size,
                                                      cuco::detail::default_block_size()>
          <<<grid_size, cuco::detail::default_block_size(), 0, stream.value()>>>(
            build_iter,
            build.num_rows(),
            thrust::constant_iterator<bool>{true},
            cuda::std::identity{},
            insert_ref);
      }
    };

    if (cudf::is_primitive_row_op_compatible(_build) && !_build_has_floating_point) {
      auto const d_build_hasher =
        primitive_row_hasher{nullate::DYNAMIC{_build_has_nulls}, _preprocessed_build};
      auto const build_iter = cudf::detail::make_counting_transform_iterator(
        size_type{0}, keys_adapter<lhs_index_type, primitive_row_hasher>{d_build_hasher});

      cuco::static_multiset_ref set_ref{empty_sentinel_key,
                                   insertion_adapter{},
                                   primitive_probing_scheme{},
                                   cuco::thread_scope_device,
                                   _bucket_storage.ref()};
      auto insert_ref = set_ref.rebind_operators(cuco::insert);
      insert(build_iter, insert_ref);
    } else {
      auto const build_has_nested_columns = cudf::has_nested_columns(_build);

      auto const d_build_hasher =
        cudf::experimental::row::hash::row_hasher{_preprocessed_build}.device_hasher(
          nullate::DYNAMIC(_build_has_nulls));
      auto const build_iter = cudf::detail::make_counting_transform_iterator(
        size_type{0}, keys_adapter<lhs_index_type, row_hasher>{d_build_hasher});

      if (build_has_nested_columns) {
        cuco::static_multiset_ref set_ref{empty_sentinel_key,
                                     insertion_adapter{},
                                     nested_probing_scheme{},
                                     cuco::thread_scope_device,
                                     _bucket_storage.ref()};
        auto insert_ref = set_ref.rebind_operators(cuco::insert);
        insert(build_iter, insert_ref);
      } else {
        cuco::static_multiset_ref set_ref{empty_sentinel_key,
                                     insertion_adapter{},
                                     simple_probing_scheme{},
                                     cuco::thread_scope_device,
                                     _bucket_storage.ref()};
        auto insert_ref = set_ref.rebind_operators(cuco::insert);
        insert(build_iter, insert_ref);
      }
    }

  }

 public:
  filtered_join(cudf::table_view const& build,
                cudf::null_equality compare_nulls,
                rmm::cuda_stream_view stream) : filtered_join(build, compare_nulls, CUCO_DESIRED_LOAD_FACTOR, stream) {}

  filtered_join(cudf::table_view const& build,
                cudf::null_equality compare_nulls,
                double load_factor,
                rmm::cuda_stream_view stream)
  : _build_props{build_properties{cudf::has_nested_nulls(build), std::any_of(build.begin(), build.end(), [](auto const& col) {
      return cudf::is_floating_point(col.type());
      }), cudf::has_nested_columns(build)}},
    _nulls_equal{compare_nulls},
    _build{build},
    _preprocessed_build{
      cudf::experimental::row::equality::preprocessed_table::create(_build, stream)},
    _bucket_storage{cuco::extent<cudf::size_type>{compute_bucket_storage_size(build, load_factor)},
                    cuco_allocator<char>{rmm::mr::polymorphic_allocator<char>{}, stream.value()}},
  {
    _bucket_storage.initialize(empty_sentinel_key);

    if (cudf::is_primitive_row_op_compatible(_build) && !_build_props._has_floating_point) {
      primitive_insert_ref_type set_ref{empty_sentinel_key,
                                  insertion_adapter{},
                                  primitive_probing_scheme{},
                                  cuco::thread_scope_device,
                                  _bucket_storage.ref()};
      auto insert_ref = set_ref.rebind_operators(cuco::insert);
      insert_build_table(insert_ref, stream);
    } else if (_build_props._has_nested_columns) {
      nested_insert_ref_type set_ref{empty_sentinel_key,
                                    insertion_adapter{},
                                    nested_probing_scheme{},
                                    cuco::thread_scope_device,
                                    _bucket_storage.ref()};
      auto insert_ref = set_ref.rebind_operators(cuco::insert);
      insert_build_table(insert_ref, stream);
    } else {
      simple_insert_ref_type set_ref{empty_sentinel_key,
                                    insertion_adapter{},
                                    simple_probing_scheme{},
                                    cuco::thread_scope_device,
                                    _bucket_storage.ref()};
      auto insert_ref = set_ref.rebind_operators(cuco::insert);
      insert_build_table(insert_ref, stream);
    }
  }

  std::unique_ptr<rmm::device_uvector<cudf::size_type>> semi_join(
    cudf::table_view const& probe, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);
  std::unique_ptr<rmm::device_uvector<cudf::size_type>> anti_join(
    cudf::table_view const& probe, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr);
};
*/

}  // namespace detail
}  // namespace CUDF_EXPORT cudf
