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
                          null_equality nulls_are_equal = true)
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

}  // namespace experimental
}  // namespace cudf
