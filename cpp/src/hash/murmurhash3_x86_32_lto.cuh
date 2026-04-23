/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuda/std/__type_traits/is_same.h"
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/column/column_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/hashing.hpp>
#include <cudf/structs/structs_column_device_view.cuh>
#include <cudf/lists/lists_column_device_view.cuh>

namespace cudf::hashing::detail {
using result_type = hash_value_type;

template <typename Nullate>
class element_hasher {
 public:

  /**
   * @brief Constructs an element_hasher object.
   *
   * @param nulls Indicates whether to check for nulls
   * @param seed  The seed to use for the hash function
   * @param null_hash The hash value to use for nulls
   */
  __device__ element_hasher(
    Nullate nulls,
    result_type seed      = DEFAULT_HASH_SEED,
    result_type null_hash = cuda::std::numeric_limits<result_type>::max()) noexcept
    : _check_nulls(nulls), _seed(seed), _null_hash(null_hash)
  {
  }

  /**
   * @brief Returns the hash value of the given element.
   *
   * @tparam T The type of the element to hash
   * @param col The column to hash
   * @param row_index The index of the row to hash
   * @return The hash value of the given element
   */
  template <typename T>
  __device__ result_type operator()(column_device_view const& col,
                                    size_type row_index) const noexcept
    requires(column_device_view::has_element_accessor<T>())
  {
    if (_check_nulls && col.is_null(row_index)) { return _null_hash; }
    return MurmurHash3_x86_32<T>{_seed}(col.element<T>(row_index));
  }

  /**
   * @brief Returns the hash value of the given element.
   *
   * @tparam T The type of the element to hash
   * @param col The column to hash
   * @param row_index The index of the row to hash
   * @return The hash value of the given element
   */
  template <typename T>
  __device__ result_type operator()(column_device_view const& col,
                                    size_type row_index) const noexcept
    requires(not column_device_view::has_element_accessor<T>())
  {
    CUDF_UNREACHABLE("Unsupported type in hash.");
  }

  Nullate _check_nulls;
  // Assumes seeds are the same as the result type of the hash function
  result_type _seed;
  result_type _null_hash;
};

template <typename Nullate>
class element_hasher_adaptor {
  static constexpr result_type NULL_HASH     = cuda::std::numeric_limits<result_type>::max();
  static constexpr result_type NON_NULL_HASH = 0;

 public:
  __device__ element_hasher_adaptor(Nullate _check_nulls, result_type seed) noexcept
    : _element_hasher(_check_nulls, seed)
  {
  }

  template <typename T>
  __device__ result_type operator()(column_device_view const& col,
                                    size_type row_index) const noexcept
    requires(not cudf::is_nested<T>() and not cudf::is_dictionary<T>())
  {
    return _element_hasher.template operator()<T>(col, row_index);
  }

  template <typename T>
  __device__ result_type operator()(column_device_view const& col,
                                    size_type row_index) const noexcept
    requires(cudf::is_nested<T>() or cudf::is_dictionary<T>())
  {
    CUDF_UNREACHABLE("Can't get here yet");
  }

  //template <typename T>
  //__device__ result_type operator()(column_device_view const& col,
  //                                  size_type row_index) const noexcept
  //  requires(cudf::is_dictionary<T>())
  //{
  //  if constexpr (Nullate) {
  //    if (col.is_null(row_index)) { return NULL_HASH; }
  //  }
  //
  //  auto const keys = col.child(dictionary_column_view::keys_column_index);
  //  return type_dispatcher<dispatch_storage_type>(
  //    keys.type(),
  //    _element_hasher,
  //    keys,
  //    static_cast<size_type>(col.element<dictionary32>(row_index)));
  //}
  //
  //template <typename T>
  //__device__ result_type operator()(column_device_view const& col,
  //                                  size_type row_index) const noexcept
  //  requires(cudf::is_nested<T>())
  //{
  //  auto hash                   = result_type{0};
  //  column_device_view curr_col = col.slice(row_index, 1);
  //  while (curr_col.type().id() == type_id::STRUCT || curr_col.type().id() == type_id::LIST) {
  //    if constexpr (Nullate) {
  //      auto validity_it = detail::make_validity_iterator<true>(curr_col);
  //      hash             = detail::accumulate(
  //        validity_it, validity_it + curr_col.size(), hash, [](auto hash, auto is_valid) {
  //          return cudf::hashing::detail::hash_combine(hash,
  //                                                     is_valid ? NON_NULL_HASH : NULL_HASH);
  //        });
  //    }
  //    if (curr_col.type().id() == type_id::STRUCT) {
  //      if (curr_col.num_child_columns() == 0) { return hash; }
  //      curr_col = detail::structs_column_device_view(curr_col).get_sliced_child(0);
  //    } else if (curr_col.type().id() == type_id::LIST) {
  //      auto list_col   = detail::lists_column_device_view(curr_col);
  //      auto list_sizes = make_list_size_iterator(list_col);
  //      hash            = detail::accumulate(
  //        list_sizes, list_sizes + list_col.size(), hash, [](auto hash, auto size) {
  //          return cudf::hashing::detail::hash_combine(hash, MurmurHash3_x86_32<size_type>{}(size));
  //        });
  //      curr_col = list_col.get_sliced_child();
  //    }
  //  }
  //  for (int i = 0; i < curr_col.size(); ++i) {
  //    hash = cudf::hashing::detail::hash_combine(
  //      hash,
  //      type_dispatcher<dispatch_void_if_nested>(curr_col.type(), _element_hasher, curr_col, i));
  //  }
  //  return hash;
  //}

  element_hasher<Nullate> const _element_hasher;
};


template <typename Nullate, typename T>
__device__ hash_value_type hasher_impl(Nullate check_nulls, cudf::column_device_view col, uint32_t seed) {
  auto hasher = element_hasher_adaptor<Nullate>{check_nulls, seed};
  return hasher.template operator()<T>(col, threadIdx.x);
}

template <typename T>
__device__ hash_value_type hasher(cudf::column_device_view col, uint32_t seed, bool const nullable) {
  return hasher_impl<nullate::DYNAMIC, T>(nullate::DYNAMIC{nullable}, col, seed);
}
} // namespace cudf::hashing::detail
