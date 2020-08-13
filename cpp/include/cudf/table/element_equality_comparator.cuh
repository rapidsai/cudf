/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/column/column_device_view.cuh>
#include <cudf/table/table_device_view.cuh>

namespace cudf
{
/**
 * @brief A specialization for floating-point `Element` type to check if
 * `lhs` is equivalent to `rhs`. `nan == nan`.
 *
 * @param[in] lhs first element
 * @param[in] rhs second element
 * @return bool `true` if `lhs` == `rhs` else `false`.
 */
template <typename Element, std::enable_if_t<std::is_floating_point<Element>::value>* = nullptr>
__device__ bool equality_compare(Element lhs, Element rhs)
{
  if (isnan(lhs) and isnan(rhs)) { return true; }
  return lhs == rhs;
}

/**
 * @brief A specialization for non-floating-point `Element` type to check if
 * `lhs` is equivalent to `rhs`.
 *
 * @param[in] lhs first element
 * @param[in] rhs second element
 * @return bool `true` if `lhs` == `rhs` else `false`.
 */
template <typename Element, std::enable_if_t<not std::is_floating_point<Element>::value>* = nullptr>
__device__ bool equality_compare(Element const lhs, Element const rhs)
{
  return lhs == rhs;
}

/**
 * @brief Performs an equality comparison between two elements in two columns.
 *
 * @tparam has_nulls Indicates the potential for null values in either column.
 **/
template <bool has_nulls = true>
class element_equality_comparator {
 public:
  /**
   * @brief Construct type-dispatched function object for comparing equality
   * between two elements.
   *
   * @note `lhs` and `rhs` may be the same.
   *
   * @param lhs The column containing the first element
   * @param rhs The column containing the second element (may be the same as lhs)
   * @param nulls_are_equal Indicates if two null elements are treated as equivalent
   **/
  __host__ __device__ element_equality_comparator(column_device_view lhs,
                                                  column_device_view rhs,
                                                  bool nulls_are_equal = true)
    : lhs{lhs}, rhs{rhs}, nulls_are_equal{nulls_are_equal}
  {
  }

  /**
   * @brief Compares the specified elements for equality.
   *
   * @param lhs_element_index The index of the first element
   * @param rhs_element_index The index of the second element
   *
   */
  template <typename Element,
            std::enable_if_t<cudf::is_equality_comparable<Element, Element>()>* = nullptr>
  __device__ bool operator()(size_type lhs_element_index, size_type rhs_element_index) const
    noexcept
  {
    if (has_nulls) {
      bool const lhs_is_null{lhs.nullable() and lhs.is_null(lhs_element_index)};
      bool const rhs_is_null{rhs.nullable() and rhs.is_null(rhs_element_index)};
      if (lhs_is_null and rhs_is_null) {
        return nulls_are_equal;
      } else if (lhs_is_null != rhs_is_null) {
        return false;
      }
    }

    return equality_compare(lhs.element<Element>(lhs_element_index),
                            rhs.element<Element>(rhs_element_index));
  }

  // Implementation moved out of line, for it requires classes not visible here.
  template <typename Element,
            std::enable_if_t<std::is_same<Element, cudf::list_view>::value>* = nullptr>
  __device__ bool operator()(size_type lhs_element_index, size_type rhs_element_index);
  
  template <typename Element,
            std::enable_if_t<not cudf::is_equality_comparable<Element, Element>()>* = nullptr>
  __device__ bool operator()(size_type lhs_element_index, size_type rhs_element_index)
  {
    release_assert(false && "Attempted to compare elements of uncomparable types.");
    return false;
  }

 private:
  column_device_view lhs;
  column_device_view rhs;
  bool nulls_are_equal;
};

template <bool has_nulls = true>
class row_equality_comparator {
 public:
  row_equality_comparator(table_device_view lhs, table_device_view rhs, bool nulls_are_equal = true)
    : lhs{lhs}, rhs{rhs}, nulls_are_equal{nulls_are_equal}
  {
    CUDF_EXPECTS(lhs.num_columns() == rhs.num_columns(), "Mismatched number of columns.");
  }

  __device__ bool operator()(size_type lhs_row_index, size_type rhs_row_index) const noexcept
  {
    auto equal_elements = [=](column_device_view l, column_device_view r) {
      return cudf::type_dispatcher(l.type(),
                                   element_equality_comparator<has_nulls>{l, r, nulls_are_equal},
                                   lhs_row_index,
                                   rhs_row_index);
    };

    return thrust::equal(thrust::seq, lhs.begin(), lhs.end(), rhs.begin(), equal_elements);
  }

 private:
  table_device_view lhs;
  table_device_view rhs;
  bool nulls_are_equal;
};

}