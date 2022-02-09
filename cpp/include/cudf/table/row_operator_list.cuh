/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/detail/utilities/hash_functions.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <thrust/equal.h>
#include <thrust/swap.h>
#include <thrust/transform_reduce.h>

#include <limits>

namespace cudf {
namespace experimental {

#pragma nv_exec_check_disable
template <template <cudf::type_id> typename IdTypeMap = id_to_type_impl,
          typename Functor,
          typename... Ts>
CUDF_HOST_DEVICE __forceinline__ constexpr decltype(auto) type_dispatcher2(cudf::data_type dtype,
                                                                           Functor f,
                                                                           Ts&&... args)
{
  switch (dtype.id()) {
    case type_id::BOOL8:
      return f.template operator()<typename IdTypeMap<type_id::BOOL8>::type>(
        std::forward<Ts>(args)...);
    case type_id::INT8:
      return f.template operator()<typename IdTypeMap<type_id::INT8>::type>(
        std::forward<Ts>(args)...);
    case type_id::INT16:
      return f.template operator()<typename IdTypeMap<type_id::INT16>::type>(
        std::forward<Ts>(args)...);
    case type_id::INT32:
      return f.template operator()<typename IdTypeMap<type_id::INT32>::type>(
        std::forward<Ts>(args)...);
    case type_id::INT64:
      return f.template operator()<typename IdTypeMap<type_id::INT64>::type>(
        std::forward<Ts>(args)...);
    case type_id::UINT8:
      return f.template operator()<typename IdTypeMap<type_id::UINT8>::type>(
        std::forward<Ts>(args)...);
    case type_id::UINT16:
      return f.template operator()<typename IdTypeMap<type_id::UINT16>::type>(
        std::forward<Ts>(args)...);
    case type_id::UINT32:
      return f.template operator()<typename IdTypeMap<type_id::UINT32>::type>(
        std::forward<Ts>(args)...);
    case type_id::UINT64:
      return f.template operator()<typename IdTypeMap<type_id::UINT64>::type>(
        std::forward<Ts>(args)...);
    case type_id::FLOAT32:
      return f.template operator()<typename IdTypeMap<type_id::FLOAT32>::type>(
        std::forward<Ts>(args)...);
    case type_id::FLOAT64:
      return f.template operator()<typename IdTypeMap<type_id::FLOAT64>::type>(
        std::forward<Ts>(args)...);
    case type_id::STRING:
      return f.template operator()<typename IdTypeMap<type_id::STRING>::type>(
        std::forward<Ts>(args)...);
    case type_id::TIMESTAMP_DAYS:
      return f.template operator()<typename IdTypeMap<type_id::TIMESTAMP_DAYS>::type>(
        std::forward<Ts>(args)...);
    case type_id::TIMESTAMP_SECONDS:
      return f.template operator()<typename IdTypeMap<type_id::TIMESTAMP_SECONDS>::type>(
        std::forward<Ts>(args)...);
    case type_id::TIMESTAMP_MILLISECONDS:
      return f.template operator()<typename IdTypeMap<type_id::TIMESTAMP_MILLISECONDS>::type>(
        std::forward<Ts>(args)...);
    case type_id::TIMESTAMP_MICROSECONDS:
      return f.template operator()<typename IdTypeMap<type_id::TIMESTAMP_MICROSECONDS>::type>(
        std::forward<Ts>(args)...);
    case type_id::TIMESTAMP_NANOSECONDS:
      return f.template operator()<typename IdTypeMap<type_id::TIMESTAMP_NANOSECONDS>::type>(
        std::forward<Ts>(args)...);
    case type_id::DURATION_DAYS:
      return f.template operator()<typename IdTypeMap<type_id::DURATION_DAYS>::type>(
        std::forward<Ts>(args)...);
    case type_id::DURATION_SECONDS:
      return f.template operator()<typename IdTypeMap<type_id::DURATION_SECONDS>::type>(
        std::forward<Ts>(args)...);
    case type_id::DURATION_MILLISECONDS:
      return f.template operator()<typename IdTypeMap<type_id::DURATION_MILLISECONDS>::type>(
        std::forward<Ts>(args)...);
    case type_id::DURATION_MICROSECONDS:
      return f.template operator()<typename IdTypeMap<type_id::DURATION_MICROSECONDS>::type>(
        std::forward<Ts>(args)...);
    case type_id::DURATION_NANOSECONDS:
      return f.template operator()<typename IdTypeMap<type_id::DURATION_NANOSECONDS>::type>(
        std::forward<Ts>(args)...);
    case type_id::DICTIONARY32:
      return f.template operator()<typename IdTypeMap<type_id::DICTIONARY32>::type>(
        std::forward<Ts>(args)...);
    case type_id::DECIMAL32:
      return f.template operator()<typename IdTypeMap<type_id::DECIMAL32>::type>(
        std::forward<Ts>(args)...);
    case type_id::DECIMAL64:
      return f.template operator()<typename IdTypeMap<type_id::DECIMAL64>::type>(
        std::forward<Ts>(args)...);
    case type_id::DECIMAL128:
      return f.template operator()<typename IdTypeMap<type_id::DECIMAL128>::type>(
        std::forward<Ts>(args)...);
    default: {
#ifndef __CUDA_ARCH__
      CUDF_FAIL("Unsupported type_id.");
#else
      cudf_assert(false && "Unsupported type_id.");

      // The following code will never be reached, but the compiler generates a
      // warning if there isn't a return value.

      // Need to find out what the return type is in order to have a default
      // return value and solve the compiler warning for lack of a default
      // return
      using return_type = decltype(f.template operator()<int8_t>(std::forward<Ts>(args)...));
      return return_type();
#endif
    }
  }
}

/**
 * @brief Performs an equality comparison between two elements in two columns.
 *
 * @tparam Nullate A cudf::nullate type describing how to check for nulls.
 */
template <typename Nullate>
class element_equality_comparator {
 public:
  /**
   * @brief Construct type-dispatched function object for comparing equality
   * between two elements.
   *
   * @note `lhs` and `rhs` may be the same.
   *
   * @param has_nulls Indicates if either input column contains nulls.
   * @param lhs The column containing the first element
   * @param rhs The column containing the second element (may be the same as lhs)
   * @param nulls_are_equal Indicates if two null elements are treated as equivalent
   */
  __host__ __device__
  element_equality_comparator(Nullate has_nulls,
                              column_device_view lhs,
                              column_device_view rhs,
                              null_equality nulls_are_equal = null_equality::EQUAL)
    : lhs{lhs}, rhs{rhs}, nulls{has_nulls}, nulls_are_equal{nulls_are_equal}
  {
  }

  /**
   * @brief Compares the specified elements for equality.
   *
   * @param lhs_element_index The index of the first element
   * @param rhs_element_index The index of the second element
   * @return True if both lhs and rhs element are both nulls and `nulls_are_equal` is true, or equal
   */
  template <typename Element,
            std::enable_if_t<cudf::is_equality_comparable<Element, Element>()>* = nullptr>
  __device__ bool operator()(size_type lhs_element_index,
                             size_type rhs_element_index) const noexcept
  {
    if (nulls) {
      bool const lhs_is_null{lhs.is_null(lhs_element_index)};
      bool const rhs_is_null{rhs.is_null(rhs_element_index)};
      if (lhs_is_null and rhs_is_null) {
        return nulls_are_equal == null_equality::EQUAL;
      } else if (lhs_is_null != rhs_is_null) {
        return false;
      }
    }

    return equality_compare(lhs.element<Element>(lhs_element_index),
                            rhs.element<Element>(rhs_element_index));
  }

  template <typename Element,
            std::enable_if_t<not cudf::is_equality_comparable<Element, Element>() and
                             not std::is_same_v<Element, cudf::list_view>>* = nullptr>
  __device__ bool operator()(size_type lhs_element_index, size_type rhs_element_index)
  {
    cudf_assert(false && "Attempted to compare elements of uncomparable types.");
    return false;
  }

  template <typename Element,
            std::enable_if_t<not cudf::is_equality_comparable<Element, Element>() and
                             std::is_same_v<Element, cudf::list_view>>* = nullptr>
  __device__ bool operator()(size_type lhs_element_index, size_type rhs_element_index)
  {
    column_device_view lcol = lhs;
    column_device_view rcol = rhs;
    int l_start_off         = lhs_element_index;
    int r_start_off         = rhs_element_index;
    int l_end_off           = lhs_element_index + 1;
    int r_end_off           = rhs_element_index + 1;
    while (lcol.type().id() == type_id::LIST) {
      auto l_size = l_end_off - l_start_off;
      auto r_size = r_end_off - r_start_off;
      if (l_size != r_size) { return false; }

      auto l_off = lcol.child(lists_column_view::offsets_column_index);
      auto r_off = rcol.child(lists_column_view::offsets_column_index);
      for (int i = 0; i < l_size; ++i) {
        if (l_off.element<size_type>(l_start_off + i + 1) -
              l_off.element<size_type>(l_start_off + i) !=
            r_off.element<size_type>(r_start_off + i + 1) -
              r_off.element<size_type>(r_start_off + i))
          return false;
      }
      lcol        = lcol.child(lists_column_view::child_column_index);
      rcol        = rcol.child(lists_column_view::child_column_index);
      l_start_off = l_off.element<size_type>(l_start_off);
      r_start_off = r_off.element<size_type>(r_start_off);
      l_end_off   = l_off.element<size_type>(l_end_off);
      r_end_off   = r_off.element<size_type>(r_end_off);
    }
    auto l_size = l_end_off - l_start_off;
    auto r_size = r_end_off - r_start_off;
    if (l_size != r_size) { return false; }
    bool equal = true;
    for (int i = 0; i < l_size; ++i) {
      equal &= type_dispatcher2(lcol.type(),
                                element_equality_comparator{nulls, lcol, rcol, nulls_are_equal},
                                l_start_off + i,
                                r_start_off + i);
    }
    return equal;
  }

 private:
  column_device_view lhs;
  column_device_view rhs;
  Nullate nulls;
  null_equality nulls_are_equal;
};

template <typename Nullate>
class row_equality_comparator {
 public:
  row_equality_comparator(Nullate has_nulls,
                          table_device_view lhs,
                          table_device_view rhs,
                          null_equality nulls_are_equal = null_equality::EQUAL)
    : lhs{lhs}, rhs{rhs}, nulls{has_nulls}, nulls_are_equal{nulls_are_equal}
  {
    CUDF_EXPECTS(lhs.num_columns() == rhs.num_columns(), "Mismatched number of columns.");
  }

  __device__ bool operator()(size_type lhs_row_index, size_type rhs_row_index) const noexcept
  {
    auto equal_elements = [=](column_device_view l, column_device_view r) {
      return cudf::type_dispatcher(l.type(),
                                   element_equality_comparator{nulls, l, r, nulls_are_equal},
                                   lhs_row_index,
                                   rhs_row_index);
    };

    return thrust::equal(thrust::seq, lhs.begin(), lhs.end(), rhs.begin(), equal_elements);
  }

 private:
  table_device_view lhs;
  table_device_view rhs;
  Nullate nulls;
  null_equality nulls_are_equal;
};

/**
 * @brief Computes the hash value of an element in the given column.
 *
 * @tparam hash_function Hash functor to use for hashing elements.
 * @tparam Nullate A cudf::nullate type describing how to check for nulls.
 */
template <template <typename> class hash_function, typename Nullate>
class element_hasher {
 public:
  template <typename T,
            typename hash_combiner,
            CUDF_ENABLE_IF(column_device_view::has_element_accessor<T>())>
  __device__ hash_value_type operator()(column_device_view col,
                                        size_type row_index,
                                        hash_combiner const& hash_combine) const
  {
    if (has_nulls && col.is_null(row_index)) { return std::numeric_limits<hash_value_type>::max(); }
    return hash_function<T>{}(col.element<T>(row_index));
  }

  template <typename T,
            typename hash_combiner,
            CUDF_ENABLE_IF(not column_device_view::has_element_accessor<T>() and
                           not std::is_same_v<T, cudf::list_view>)>
  __device__ hash_value_type operator()(column_device_view col,
                                        size_type row_index,
                                        hash_combiner const& hash_combine) const
  {
    cudf_assert(false && "Unsupported type in hash.");
    return {};
  }

  template <typename T,
            typename hash_combiner,
            CUDF_ENABLE_IF(not column_device_view::has_element_accessor<T>() and
                           std::is_same_v<T, cudf::list_view>)>
  __device__ hash_value_type operator()(column_device_view col,
                                        size_type row_index,
                                        hash_combiner const& hash_combine) const
  {
    auto hash                   = hash_value_type{0};
    column_device_view curr_col = col;
    int start_off               = row_index;
    int end_off                 = row_index + 1;
    while (curr_col.type().id() == type_id::LIST) {
      auto size = end_off - start_off;

      auto offsets = curr_col.child(lists_column_view::offsets_column_index);
      for (int i = 0; i < size; ++i) {
        auto const child_size =
          offsets.element<size_type>(start_off + i + 1) - offsets.element<size_type>(start_off + i);
        hash = hash_combine(hash, hash_function<decltype(child_size)>{}(child_size));
      }
      curr_col  = curr_col.child(lists_column_view::child_column_index);
      start_off = offsets.element<size_type>(start_off);
      end_off   = offsets.element<size_type>(end_off);
    }
    auto size = end_off - start_off;
    hash      = hash_combine(hash, hash_function<decltype(size)>{}(size));
    for (int i = 0; i < size; ++i) {
      hash = hash_combine(hash,
                          type_dispatcher2(curr_col.type(),
                                           element_hasher<hash_function, Nullate>{has_nulls},
                                           curr_col,
                                           start_off + i,
                                           hash_combine));
    }
    // printf("hash %d\n", hash);
    return hash;
    // return hash_value_type{0};
  }

  Nullate has_nulls;
};

/**
 * @brief Computes the hash value of a row in the given table.
 *
 * @tparam hash_function Hash functor to use for hashing elements.
 * @tparam Nullate A cudf::nullate type describing how to check for nulls.
 */
template <template <typename> class hash_function, typename Nullate>
class row_hasher {
 public:
  row_hasher() = delete;
  CUDF_HOST_DEVICE row_hasher(Nullate has_nulls, table_device_view t)
    : _table{t}, _has_nulls{has_nulls}
  {
  }
  CUDF_HOST_DEVICE row_hasher(Nullate has_nulls, table_device_view t, uint32_t seed)
    : _table{t}, _seed(seed), _has_nulls{has_nulls}
  {
  }

  __device__ auto operator()(size_type row_index) const
  {
    auto hash_combiner = [](hash_value_type lhs, hash_value_type rhs) {
      return hash_function<hash_value_type>{}.hash_combine(lhs, rhs);
    };

    // Hash the first column w/ the seed
    auto const initial_hash = hash_combiner(
      hash_value_type{0},
      type_dispatcher<dispatch_storage_type>(_table.column(0).type(),
                                             // TODO: revert back to using seed
                                             element_hasher<hash_function, Nullate>{_has_nulls},
                                             _table.column(0),
                                             row_index,
                                             hash_combiner));

    // Hashes an element in a column
    auto hasher = [=](size_type column_index) {
      return cudf::type_dispatcher<dispatch_storage_type>(
        _table.column(column_index).type(),
        element_hasher<hash_function, Nullate>{_has_nulls},
        _table.column(column_index),
        row_index,
        hash_combiner);
    };

    // Hash each element and combine all the hash values together
    return thrust::transform_reduce(
      thrust::seq,
      // note that this starts at 1 and not 0 now since we already hashed the first column
      thrust::make_counting_iterator(1),
      thrust::make_counting_iterator(_table.num_columns()),
      hasher,
      initial_hash,
      hash_combiner);
  }

 private:
  table_device_view _table;
  Nullate _has_nulls;
  uint32_t _seed{DEFAULT_HASH_SEED};
};

}  // namespace experimental
}  // namespace cudf
