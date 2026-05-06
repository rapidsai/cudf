/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/row_operator/common_utils.cuh>
#include <cudf/detail/utilities/accumulate.cuh>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/hashing.hpp>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/lists/list_device_view.cuh>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/structs/structs_column_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/std/limits>

namespace cudf::hashing::detail {

using result_type = hash_value_type;

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
    bool nulls,
    result_type seed      = DEFAULT_HASH_SEED,
    result_type null_hash = cuda::std::numeric_limits<result_type>::max()) noexcept
    : _check_nulls(nulls), _seed(seed), _null_hash(null_hash)
  {
  }

  template <typename T>
  __device__ result_type operator()(column_device_view const& col,
                                    size_type row_index) const noexcept
    requires(column_device_view::has_element_accessor<T>())
  {
    if (_check_nulls && col.is_null(row_index)) { return _null_hash; }
    return MurmurHash3_x86_32<T>{_seed}(col.element<T>(row_index));
  }

  template <typename T>
  __device__ result_type operator()(column_device_view const& col,
                                    size_type row_index) const noexcept
    requires(not column_device_view::has_element_accessor<T>())
  {
    CUDF_UNREACHABLE("Unsupported type in hash.");
  }

  bool _check_nulls;
  result_type _seed;
  result_type _null_hash;
};

class element_hasher_adaptor {
  static constexpr result_type NULL_HASH     = cuda::std::numeric_limits<result_type>::max();
  static constexpr result_type NON_NULL_HASH = 0;

 public:
  __device__ element_hasher_adaptor(bool check_nulls, result_type seed) noexcept
    : _element_hasher(check_nulls, seed), _check_nulls(check_nulls)
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
    requires(cudf::is_dictionary<T>())
  {
    if (_check_nulls && col.is_null(row_index)) { return NULL_HASH; }

    auto const keys = col.child(dictionary_column_view::keys_column_index);
    return type_dispatcher<dispatch_storage_type>(
      keys.type(),
      _element_hasher,
      keys,
      static_cast<size_type>(col.element<dictionary32>(row_index)));
  }

  template <typename T>
  __device__ result_type operator()(column_device_view const& col,
                                    size_type row_index) const noexcept
    requires(cudf::is_nested<T>())
  {
    auto hash                   = result_type{0};
    column_device_view curr_col = col.slice(row_index, 1);
    while (curr_col.type().id() == type_id::STRUCT || curr_col.type().id() == type_id::LIST) {
      if (_check_nulls) {
        auto validity_it = cudf::detail::make_validity_iterator<true>(curr_col);
        hash             = cudf::detail::accumulate(
          validity_it, validity_it + curr_col.size(), hash, [](auto h, auto is_valid) {
            return cudf::hashing::detail::hash_combine(h, is_valid ? NON_NULL_HASH : NULL_HASH);
          });
      }
      if (curr_col.type().id() == type_id::STRUCT) {
        if (curr_col.num_child_columns() == 0) { return hash; }
        curr_col = cudf::detail::structs_column_device_view(curr_col).get_sliced_child(0);
      } else if (curr_col.type().id() == type_id::LIST) {
        auto list_col   = cudf::detail::lists_column_device_view(curr_col);
        auto list_sizes = cudf::make_list_size_iterator(list_col);
        hash            = cudf::detail::accumulate(
          list_sizes, list_sizes + list_col.size(), hash, [](auto h, auto size) {
            return cudf::hashing::detail::hash_combine(h, MurmurHash3_x86_32<size_type>{}(size));
          });
        curr_col = list_col.get_sliced_child();
      }
    }
    for (int i = 0; i < curr_col.size(); ++i) {
      hash =
        cudf::hashing::detail::hash_combine(hash,
                                            type_dispatcher<cudf::detail::dispatch_void_if_nested>(
                                              curr_col.type(), _element_hasher, curr_col, i));
    }
    return hash;
  }

  element_hasher const _element_hasher;
  bool const _check_nulls;
};

template <typename T>
__device__ hash_value_type
hasher_impl(bool check_nulls, cudf::column_device_view col, uint32_t seed, size_type row_index)
{
  auto const hasher = element_hasher_adaptor{check_nulls, seed};
  return hasher.template operator()<T>(col, row_index);
}

template <typename T>
__device__ hash_value_type
hasher(cudf::column_device_view col, uint32_t seed, bool const nullable, size_type row_index)
{
  return hasher_impl<T>(nullable, col, seed, row_index);
}

}  // namespace cudf::hashing::detail
