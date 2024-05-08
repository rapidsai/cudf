/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.
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
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/algorithm.cuh>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/hashing/detail/default_hash.cuh>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/lists/detail/dremel.hpp>
#include <cudf/lists/list_device_view.cuh>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/sorting.hpp>
#include <cudf/structs/structs_column_device_view.cuh>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/std/tuple>
#include <cuda/std/utility>
#include <thrust/detail/use_default.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/iterator_categories.h>
#include <thrust/iterator/iterator_facade.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/logical.h>
#include <thrust/swap.h>
#include <thrust/transform_reduce.h>

#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

namespace cudf {

namespace experimental {

template <typename T>
using type_identity = T;

template <typename T, template <typename U> typename... c_rest>
struct nested_conditional_t;

template <typename T>
struct nested_conditional_t<T> {
  using type = type_identity<T>;
};

template <typename T,
          template <typename U>
          typename c_first,
          template <typename V>
          typename... c_rest>
struct nested_condtional_t {
  using type = c_first<typename nested_conditional_t<T, c_rest...>::type>;
};

template <bool B, typename T>
using dispatch_void_conditional_t = std::conditional_t<B, void, T>;

template <typename... Types>
struct dispatch_void_conditional_generator {
  template <typename T>
  using type = dispatch_void_conditional_t<std::disjunction<std::is_same<T, Types>...>::value, T>;
};

/**
 * @brief A map from cudf::type_id to cudf type that excludes LIST and STRUCT types.
 *
 * To be used with type_dispatcher in place of the default map, when it is required that STRUCT
 * and LIST map to void. This is useful when we want to avoid recursion in a functor. For example,
 * in element_comparator, we have a specialization for STRUCT but the type_dispatcher in it is
 * only used to dispatch to the same functor for non-nested types. Even when we're guaranteed to
 * not have non-nested types at that point, the compiler doesn't know this and would try to create
 * recursive code which is very slow.
 *
 * Usage:
 * @code
 * type_dispatcher<dispatch_nested_to_void>(data_type(), functor{});
 * @endcode
 */
template <typename T>
using dispatch_void_if_nested_t =
  dispatch_void_conditional_t<std::is_same_v<cudf::struct_view, T> or
                                std::is_same_v<cudf::list_view, T>,
                              T>;

template <cudf::type_id t>
struct dispatch_void_if_nested {
  using type = dispatch_void_if_nested_t<id_to_type<t>>;
};

namespace row {
enum class lhs_index_type : size_type {};
enum class rhs_index_type : size_type {};

/**
 * @brief A counting iterator that uses strongly typed indices bound to tables.
 *
 * Performing lexicographic or equality comparisons between values in two
 * tables requires the use of strongly typed indices. The strong index types
 * `lhs_index_type` and `rhs_index_type` ensure that index values are bound to
 * the correct table, regardless of the order in which these indices are
 * provided to the call operator. This struct and its type aliases
 * `lhs_iterator` and `rhs_iterator` provide an interface similar to a counting
 * iterator, with strongly typed values to represent the table indices.
 *
 * @tparam Index The strong index type
 */
template <typename Index, typename Underlying = std::underlying_type_t<Index>>
struct strong_index_iterator : public thrust::iterator_facade<strong_index_iterator<Index>,
                                                              Index,
                                                              thrust::use_default,
                                                              thrust::random_access_traversal_tag,
                                                              Index,
                                                              Underlying> {
  using super_t =
    thrust::iterator_adaptor<strong_index_iterator<Index>, Index>;  ///< The base class

  /**
   * @brief Constructs a strong index iterator
   *
   * @param n The beginning index
   */
  explicit constexpr strong_index_iterator(Underlying n) : begin{n} {}

  friend class thrust::iterator_core_access;  ///< Allow access to the base class

 private:
  __device__ constexpr void increment() { ++begin; }
  __device__ constexpr void decrement() { --begin; }

  __device__ constexpr void advance(Underlying n) { begin += n; }

  __device__ constexpr bool equal(strong_index_iterator<Index> const& other) const noexcept
  {
    return begin == other.begin;
  }

  __device__ constexpr Index dereference() const noexcept { return static_cast<Index>(begin); }

  __device__ constexpr Underlying distance_to(
    strong_index_iterator<Index> const& other) const noexcept
  {
    return other.begin - begin;
  }

  Underlying begin{};
};

/**
 * @brief Iterator representing indices into a left-side table.
 */
using lhs_iterator = strong_index_iterator<lhs_index_type>;

/**
 * @brief Iterator representing indices into a right-side table.
 */
using rhs_iterator = strong_index_iterator<rhs_index_type>;

namespace lexicographic {

/**
 * @brief Computes a weak ordering of two values with special sorting behavior.
 *
 * This relational comparator functor compares physical values rather than logical
 * elements like lists, strings, or structs. It evaluates `NaN` as not less than, equal to, or
 * greater than other values and is IEEE-754 compliant.
 */
struct physical_element_comparator {
  /**
   * @brief Operator for relational comparisons.
   *
   * @param lhs First element
   * @param rhs Second element
   * @return Relation between elements
   */
  template <typename Element>
  __device__ constexpr weak_ordering operator()(Element const lhs, Element const rhs) const noexcept
  {
    return detail::compare_elements(lhs, rhs);
  }
};

/**
 * @brief Relational comparator functor that compares physical values rather than logical
 * elements like lists, strings, or structs. It evaluates `NaN` as equivalent to other `NaN`s and
 * greater than all other values.
 */
struct sorting_physical_element_comparator {
  /**
   * @brief Operator for relational comparison of non-floating point values.
   *
   * @param lhs First element
   * @param rhs Second element
   * @return Relation between elements
   */
  template <typename Element, CUDF_ENABLE_IF(not std::is_floating_point_v<Element>)>
  __device__ constexpr weak_ordering operator()(Element const lhs, Element const rhs) const noexcept
  {
    return detail::compare_elements(lhs, rhs);
  }

  /**
   * @brief Operator for relational comparison of floating point values.
   *
   * @param lhs First element
   * @param rhs Second element
   * @return Relation between elements
   */
  template <typename Element, CUDF_ENABLE_IF(std::is_floating_point_v<Element>)>
  __device__ constexpr weak_ordering operator()(Element const lhs, Element const rhs) const noexcept
  {
    if (isnan(lhs)) {
      return isnan(rhs) ? weak_ordering::EQUIVALENT : weak_ordering::GREATER;
    } else if (isnan(rhs)) {
      return weak_ordering::LESS;
    }

    return detail::compare_elements(lhs, rhs);
  }
};

using optional_dremel_view = thrust::optional<detail::dremel_device_view const>;

// The has_nested_columns template parameter of the device_row_comparator is
// necessary to help the compiler optimize our code. Without it, the list and
// struct view specializations are present in the code paths used for primitive
// types, and the compiler fails to inline this nearly as well resulting in a
// significant performance drop.  As a result, there is some minor tension in
// the current design between the presence of this parameter and the way that
// the Dremel data is passed around, first as a
// std::optional<device_span<dremel_device_view>> in the
// preprocessed_table/device_row_comparator (which is always valid when
// has_nested_columns and is otherwise invalid) that is then unpacked to a
// thrust::optional<dremel_device_view> at the element_comparator level (which
// is always valid for a list column and otherwise invalid).  We cannot use an
// additional template parameter for the element_comparator on a per-column
// basis because we cannot conditionally define dremel_device_view member
// variables without jumping through extra hoops with inheritance, so the
// thrust::optional<dremel_device_view> member must be an optional rather than
// a raw dremel_device_view.
/**
 * @brief Computes the lexicographic comparison between 2 rows.
 *
 * Lexicographic ordering is determined by:
 * - Two rows are compared element by element.
 * - The first mismatching element defines which row is lexicographically less
 * or greater than the other.
 * - If the rows are compared without mismatched elements, the rows are equivalent
 *
 *
 * Lexicographic ordering is exactly equivalent to doing an alphabetical sort of
 * two words, for example, `aac` would be *less* than (or precede) `abb`. The
 * second letter in both words is the first non-equal letter, and `a < b`, thus
 * `aac < abb`.
 *
 * @note The operator overloads in sub-class `element_comparator` are templated via the
 *        `type_dispatcher` to help select an overload instance for each column in a table.
 *        So, `cudf::is_nested<Element>` will return `true` if the table has nested-type columns,
 *        but it will be a runtime error if template parameter `has_nested_columns != true`.
 *
 * @tparam has_nested_columns compile-time optimization for primitive types.
 *         This template parameter is to be used by the developer by querying
 *         `cudf::detail::has_nested_columns(input)`. `true` compiles operator
 *         overloads for nested types, while `false` only compiles operator
 *         overloads for primitive types.
 * @tparam Nullate A cudf::nullate type describing whether to check for nulls.
 * @tparam PhysicalElementComparator A relational comparator functor that compares individual
 * values rather than logical elements, defaults to `NaN` aware relational comparator that
 * evaluates `NaN` as greater than all other values.
 */
template <bool has_nested_columns,
          typename Nullate,
          typename PhysicalElementComparator = sorting_physical_element_comparator>
class device_row_comparator {
 public:
  friend class self_comparator;       ///< Allow self_comparator to access private members
  friend class two_table_comparator;  ///< Allow two_table_comparator to access private members

  /**
   * @brief Construct a function object for performing a lexicographic
   * comparison between the rows of two tables.
   *
   * @param check_nulls Indicates if any input column contains nulls.
   * @param lhs The first table
   * @param rhs The second table (may be the same table as `lhs`)
   * @param l_dremel_device_views lhs table dremel device view for list type
   * @param r_dremel_device_views rhs table dremel device view for list type
   * @param depth Optional, device array the same length as a row that contains starting depths of
   * columns if they're nested, and 0 otherwise.
   * @param column_order Optional, device array the same length as a row that indicates the
   * desired ascending/descending order of each column in a row. If `nullopt`, it is assumed all
   * columns are sorted in ascending order.
   * @param null_precedence Optional, device array the same length as a row and indicates how null
   * values compare to all other for every column. If `nullopt`, then null precedence would be
   * `null_order::BEFORE` for all columns.
   * @param comparator Physical element relational comparison functor.
   */
  device_row_comparator(Nullate check_nulls,
                        table_device_view lhs,
                        table_device_view rhs,
                        device_span<detail::dremel_device_view const> l_dremel_device_views,
                        device_span<detail::dremel_device_view const> r_dremel_device_views,
                        std::optional<device_span<int const>> depth                  = std::nullopt,
                        std::optional<device_span<order const>> column_order         = std::nullopt,
                        std::optional<device_span<null_order const>> null_precedence = std::nullopt,
                        PhysicalElementComparator comparator                         = {}) noexcept
    : _lhs{lhs},
      _rhs{rhs},
      _l_dremel(l_dremel_device_views),
      _r_dremel(r_dremel_device_views),
      _check_nulls{check_nulls},
      _depth{depth},
      _column_order{column_order},
      _null_precedence{null_precedence},
      _comparator{comparator}
  {
  }

  /**
   * @brief Construct a function object for performing a lexicographic
   * comparison between the rows of two tables.
   * This is a special overload to allow device-side construction of the
   * comparator for cases where no preprocessing is needed, i.e. tables with
   * non-nested type columns.
   *
   * @param check_nulls Indicates if any input column contains nulls.
   * @param lhs The first table
   * @param rhs The second table (may be the same table as `lhs`)
   * @param column_order Optional, device array the same length as a row that indicates the
   * desired ascending/descending order of each column in a row. If `nullopt`, it is assumed all
   * columns are sorted in ascending order.
   * @param null_precedence Optional, device array the same length as a row and indicates how null
   * values compare to all other for every column. If `nullopt`, then null precedence would be
   * `null_order::BEFORE` for all columns.
   * @param comparator Physical element relational comparison functor.
   */
  template <bool nested_disable = not has_nested_columns, CUDF_ENABLE_IF(nested_disable)>
  __device__ device_row_comparator(
    Nullate check_nulls,
    table_device_view lhs,
    table_device_view rhs,
    std::optional<device_span<order const>> column_order         = std::nullopt,
    std::optional<device_span<null_order const>> null_precedence = std::nullopt,
    PhysicalElementComparator comparator                         = {}) noexcept
    : _lhs{lhs},
      _rhs{rhs},
      _l_dremel{},
      _r_dremel{},
      _check_nulls{check_nulls},
      _depth{},
      _column_order{column_order},
      _null_precedence{null_precedence},
      _comparator{comparator}
  {
  }

  /**
   * @brief Performs a relational comparison between two elements in two columns.
   */
  class element_comparator {
   public:
    /**
     * @brief Construct type-dispatched function object for performing a
     * relational comparison between two elements.
     *
     * @note `lhs` and `rhs` may be the same.
     *
     * @param check_nulls Indicates if either input column contains nulls.
     * @param lhs The column containing the first element
     * @param rhs The column containing the second element (may be the same as lhs)
     * @param null_precedence Indicates how null values are ordered with other values
     * @param depth The depth of the column if part of a nested column @see
     * preprocessed_table::depths
     * @param comparator Physical element relational comparison functor.
     * @param l_dremel_device_view <>
     * @param r_dremel_device_view <>
     */
    __device__ element_comparator(Nullate check_nulls,
                                  column_device_view lhs,
                                  column_device_view rhs,
                                  null_order null_precedence                = null_order::BEFORE,
                                  int depth                                 = 0,
                                  PhysicalElementComparator comparator      = {},
                                  optional_dremel_view l_dremel_device_view = {},
                                  optional_dremel_view r_dremel_device_view = {})
      : _lhs{lhs},
        _rhs{rhs},
        _check_nulls{check_nulls},
        _null_precedence{null_precedence},
        _depth{depth},
        _l_dremel_device_view{l_dremel_device_view},
        _r_dremel_device_view{r_dremel_device_view},
        _comparator{comparator}
    {
    }

    /**
     * @brief Performs a relational comparison between the specified elements
     *
     * @param lhs_element_index The index of the first element
     * @param rhs_element_index The index of the second element
     * @return Indicates the relationship between the elements in the `lhs` and `rhs` columns,
     * along with the depth at which a null value was encountered.
     */
    template <typename Element,
              CUDF_ENABLE_IF(cudf::is_relationally_comparable<Element, Element>())>
    __device__ cuda::std::pair<weak_ordering, int> operator()(
      size_type const lhs_element_index, size_type const rhs_element_index) const noexcept
    {
      if (_check_nulls) {
        bool const lhs_is_null{_lhs.is_null(lhs_element_index)};
        bool const rhs_is_null{_rhs.is_null(rhs_element_index)};

        if (lhs_is_null or rhs_is_null) {  // at least one is null
          return cuda::std::pair(null_compare(lhs_is_null, rhs_is_null, _null_precedence), _depth);
        }
      }

      return cuda::std::pair(_comparator(_lhs.element<Element>(lhs_element_index),
                                         _rhs.element<Element>(rhs_element_index)),
                             std::numeric_limits<int>::max());
    }

    /**
     * @brief Throws run-time error when columns types cannot be compared
     *        or if this class is instantiated with `has_nested_columns = false` but
     *        passed tables with nested columns
     *
     * @return Ordering
     */
    template <typename Element,
              CUDF_ENABLE_IF(not cudf::is_relationally_comparable<Element, Element>() and
                             (not has_nested_columns or not cudf::is_nested<Element>()))>
    __device__ cuda::std::pair<weak_ordering, int> operator()(size_type const,
                                                              size_type const) const noexcept
    {
      CUDF_UNREACHABLE("Attempted to compare elements of uncomparable types.");
    }

    /**
     * @brief Compares two struct-type columns
     *
     * @param lhs_element_index The index of the first element
     * @param rhs_element_index The index of the second element
     * @return Indicates the relationship between the elements in the `lhs` and `rhs` columns,
     * along with the depth at which a null value was encountered.
     */
    template <typename Element,
              CUDF_ENABLE_IF(has_nested_columns and std::is_same_v<Element, cudf::struct_view>)>
    __device__ cuda::std::pair<weak_ordering, int> operator()(
      size_type const lhs_element_index, size_type const rhs_element_index) const noexcept
    {
      column_device_view lcol = _lhs;
      column_device_view rcol = _rhs;
      int depth               = _depth;
      while (lcol.type().id() == type_id::STRUCT) {
        bool const lhs_is_null{lcol.is_null(lhs_element_index)};
        bool const rhs_is_null{rcol.is_null(rhs_element_index)};

        if (lhs_is_null or rhs_is_null) {  // at least one is null
          weak_ordering state = null_compare(lhs_is_null, rhs_is_null, _null_precedence);
          return cuda::std::pair(state, depth);
        }

        if (lcol.num_child_columns() == 0) {
          return cuda::std::pair(weak_ordering::EQUIVALENT, std::numeric_limits<int>::max());
        }

        // Non-empty structs have been modified to only have 1 child when using this.
        lcol = detail::structs_column_device_view(lcol).get_sliced_child(0);
        rcol = detail::structs_column_device_view(rcol).get_sliced_child(0);
        ++depth;
      }

      return cudf::type_dispatcher<dispatch_void_if_nested>(
        lcol.type(),
        element_comparator{_check_nulls, lcol, rcol, _null_precedence, depth, _comparator},
        lhs_element_index,
        rhs_element_index);
    }

    /**
     * @brief Compares two list-type columns
     *
     * @param lhs_element_index The index of the first element
     * @param rhs_element_index The index of the second element
     * @return Indicates the relationship between the elements in the `lhs` and `rhs` columns,
     * along with the depth at which a null value was encountered.
     */
    template <typename Element,
              CUDF_ENABLE_IF(has_nested_columns and std::is_same_v<Element, cudf::list_view>)>
    __device__ cuda::std::pair<weak_ordering, int> operator()(size_type lhs_element_index,
                                                              size_type rhs_element_index)
    {
      // only order top-NULLs according to null_order
      auto const is_l_row_null = _lhs.is_null(lhs_element_index);
      auto const is_r_row_null = _rhs.is_null(rhs_element_index);
      if (is_l_row_null || is_r_row_null) {
        return cuda::std::pair(null_compare(is_l_row_null, is_r_row_null, _null_precedence),
                               _depth);
      }

      // These are all the values from the Dremel encoding.
      auto const l_max_def_level = _l_dremel_device_view->max_def_level;
      auto const r_max_def_level = _r_dremel_device_view->max_def_level;
      auto const l_def_levels    = _l_dremel_device_view->def_levels;
      auto const r_def_levels    = _r_dremel_device_view->def_levels;
      auto const l_rep_levels    = _l_dremel_device_view->rep_levels;
      auto const r_rep_levels    = _r_dremel_device_view->rep_levels;

      // Traverse the nested list hierarchy to get a column device view
      // pointing to the underlying child data.
      column_device_view lcol = _lhs.slice(lhs_element_index, 1);
      column_device_view rcol = _rhs.slice(rhs_element_index, 1);

      while (lcol.type().id() == type_id::LIST) {
        lcol = detail::lists_column_device_view(lcol).get_sliced_child();
        rcol = detail::lists_column_device_view(rcol).get_sliced_child();
      }

      // These start and end values indicate the start and end points of all
      // the elements of the lists in the current list element
      // (`[lhs|rhs]_element_index`) that we are comparing.
      auto const l_offsets = _l_dremel_device_view->offsets;
      auto const r_offsets = _r_dremel_device_view->offsets;
      auto l_start         = l_offsets[lhs_element_index];
      auto l_end           = l_offsets[lhs_element_index + 1];
      auto r_start         = r_offsets[rhs_element_index];
      auto r_end           = r_offsets[rhs_element_index + 1];

      // This comparator will be used to compare leaf (non-nested) data types.
      auto comparator =
        element_comparator{_check_nulls, lcol, rcol, _null_precedence, _depth, _comparator};

      // Loop over each element in the encoding. Note that this includes nulls
      // and empty lists, so not every index corresponds to an actual element
      // in the child column. The element_index is used to keep track of the current
      // child element that we're actually comparing.
      for (int l_dremel_index = l_start, r_dremel_index = r_start, element_index = 0;
           l_dremel_index < l_end and r_dremel_index < r_end;
           ++l_dremel_index, ++r_dremel_index) {
        auto const l_rep_level = l_rep_levels[l_dremel_index];
        auto const r_rep_level = r_rep_levels[r_dremel_index];

        // early exit for smaller sub-list
        if (l_rep_level != r_rep_level) {
          // the lower repetition level is a smaller sub-list
          return l_rep_level < r_rep_level ? cuda::std::pair(weak_ordering::LESS, _depth)
                                           : cuda::std::pair(weak_ordering::GREATER, _depth);
        }

        // only compare if left and right are at same nesting level
        auto const l_def_level = l_def_levels[l_dremel_index];
        auto const r_def_level = r_def_levels[r_dremel_index];

        // either left or right are empty or NULLs of arbitrary nesting
        if (l_def_level < l_max_def_level || r_def_level < r_max_def_level) {
          // in the fully unraveled version of the list column, only the
          // most nested NULLs and leafs are present
          // In this rare condition that we get to the most nested NULL, we increment
          // element_index because either both rows have a deeply nested NULL at the
          // same position, and we'll "continue" in our iteration, or we will early
          // exit if only one of the rows has a deeply nested NULL
          if ((lcol.nullable() and l_def_levels[l_dremel_index] == l_max_def_level - 1) or
              (rcol.nullable() and r_def_levels[r_dremel_index] == r_max_def_level - 1)) {
            ++element_index;
          }
          if (l_def_level == r_def_level) { continue; }
          // We require [] < [NULL] < [leaf] for nested nulls.
          // The null_precedence only affects top level nulls.
          return l_def_level < r_def_level ? cuda::std::pair(weak_ordering::LESS, _depth)
                                           : cuda::std::pair(weak_ordering::GREATER, _depth);
        }

        // finally, compare leaf to leaf
        weak_ordering state{weak_ordering::EQUIVALENT};
        int last_null_depth                    = _depth;
        cuda::std::tie(state, last_null_depth) = cudf::type_dispatcher<dispatch_void_if_nested>(
          lcol.type(), comparator, element_index, element_index);
        if (state != weak_ordering::EQUIVALENT) { return cuda::std::pair(state, _depth); }
        ++element_index;
      }

      // If we have reached this stage, we know that definition levels,
      // repetition levels, and actual elements are identical in both list
      // columns up to the `min(l_end - l_start, r_end - r_start)` element of
      // the Dremel encoding. However, two lists can only compare equivalent if
      // they are of the same length. Otherwise, the shorter of the two is less
      // than the longer. This final check determines the appropriate resulting
      // ordering by checking how many total elements each list is composed of.
      return cuda::std::pair(detail::compare_elements(l_end - l_start, r_end - r_start), _depth);
    }

   private:
    column_device_view const _lhs;
    column_device_view const _rhs;
    Nullate const _check_nulls;
    null_order const _null_precedence;
    int const _depth;
    optional_dremel_view _l_dremel_device_view;
    optional_dremel_view _r_dremel_device_view;
    PhysicalElementComparator const _comparator;
  };

 public:
  /**
   * @brief Checks whether the row at `lhs_index` in the `lhs` table compares
   * lexicographically less, greater, or equivalent to the row at `rhs_index` in the `rhs` table.
   *
   * @param lhs_index The index of the row in the `lhs` table to examine
   * @param rhs_index The index of the row in the `rhs` table to examine
   * @return weak ordering comparison of the row in the `lhs` table relative to the row in the
   * `rhs` table
   */
  __device__ constexpr weak_ordering operator()(size_type const lhs_index,
                                                size_type const rhs_index) const noexcept
  {
    int last_null_depth = std::numeric_limits<int>::max();
    size_type list_column_index{-1};
    for (size_type i = 0; i < _lhs.num_columns(); ++i) {
      if (_lhs.column(i).type().id() == type_id::LIST) { ++list_column_index; }

      int const depth = _depth.has_value() ? (*_depth)[i] : 0;
      if (depth > last_null_depth) { continue; }

      bool const ascending =
        _column_order.has_value() ? (*_column_order)[i] == order::ASCENDING : true;

      null_order const null_precedence =
        _null_precedence.has_value() ? (*_null_precedence)[i] : null_order::BEFORE;

      // TODO: At what point do we verify that the columns of lhs and rhs are
      // all of the same types? I assume that it's already happened before
      // here, otherwise the current code would be failing.
      auto const [l_dremel_i, r_dremel_i] =
        _lhs.column(i).type().id() == type_id::LIST
          ? std::make_tuple(optional_dremel_view(_l_dremel[list_column_index]),
                            optional_dremel_view(_r_dremel[list_column_index]))
          : std::make_tuple(optional_dremel_view{}, optional_dremel_view{});

      auto element_comp = element_comparator{_check_nulls,
                                             _lhs.column(i),
                                             _rhs.column(i),
                                             null_precedence,
                                             depth,
                                             _comparator,
                                             l_dremel_i,
                                             r_dremel_i};

      weak_ordering state;
      cuda::std::tie(state, last_null_depth) =
        cudf::type_dispatcher(_lhs.column(i).type(), element_comp, lhs_index, rhs_index);

      if (state == weak_ordering::EQUIVALENT) { continue; }

      return ascending
               ? state
               : (state == weak_ordering::GREATER ? weak_ordering::LESS : weak_ordering::GREATER);
    }
    return weak_ordering::EQUIVALENT;
  }

 private:
  table_device_view const _lhs;
  table_device_view const _rhs;
  device_span<detail::dremel_device_view const> const _l_dremel;
  device_span<detail::dremel_device_view const> const _r_dremel;
  Nullate const _check_nulls;
  std::optional<device_span<int const>> const _depth;
  std::optional<device_span<order const>> const _column_order;
  std::optional<device_span<null_order const>> const _null_precedence;
  PhysicalElementComparator const _comparator;
};  // class device_row_comparator

/**
 * @brief Wraps and interprets the result of templated Comparator that returns a weak_ordering.
 * Returns true if the weak_ordering matches any of the templated values.
 *
 * Note that this should never be used with only `weak_ordering::EQUIVALENT`.
 * An equality comparator should be used instead for optimal performance.
 *
 * @tparam Comparator generic comparator that returns a weak_ordering.
 * @tparam values weak_ordering parameter pack of orderings to interpret as true
 */
template <typename Comparator, weak_ordering... values>
struct weak_ordering_comparator_impl {
  static_assert(not((weak_ordering::EQUIVALENT == values) && ...),
                "weak_ordering_comparator should not be used for pure equality comparisons. The "
                "`row_equality_comparator` should be used instead");

  template <typename LhsType, typename RhsType>
  __device__ constexpr bool operator()(LhsType const lhs_index,
                                       RhsType const rhs_index) const noexcept
  {
    weak_ordering const result = comparator(lhs_index, rhs_index);
    return ((result == values) || ...);
  }
  Comparator const comparator;
};

/**
 * @brief Wraps and interprets the result of device_row_comparator, true if the result is
 * weak_ordering::LESS meaning one row is lexicographically *less* than another row.
 *
 * @tparam Comparator generic comparator that returns a weak_ordering
 */
template <typename Comparator>
struct less_comparator : weak_ordering_comparator_impl<Comparator, weak_ordering::LESS> {
  /**
   * @brief Constructs a less_comparator
   *
   * @param comparator The comparator to wrap
   */
  less_comparator(Comparator const& comparator)
    : weak_ordering_comparator_impl<Comparator, weak_ordering::LESS>{comparator}
  {
  }
};

/**
 * @brief Wraps and interprets the result of device_row_comparator, true if the result is
 * weak_ordering::LESS or weak_ordering::EQUIVALENT meaning one row is lexicographically *less*
 * than or *equivalent* to another row.
 *
 * @tparam Comparator generic comparator that returns a weak_ordering
 */
template <typename Comparator>
struct less_equivalent_comparator
  : weak_ordering_comparator_impl<Comparator, weak_ordering::LESS, weak_ordering::EQUIVALENT> {
  /**
   * @brief Constructs a less_equivalent_comparator
   *
   * @param comparator The comparator to wrap
   */
  less_equivalent_comparator(Comparator const& comparator)
    : weak_ordering_comparator_impl<Comparator, weak_ordering::LESS, weak_ordering::EQUIVALENT>{
        comparator}
  {
  }
};

/**
 * @brief Preprocessed table for use with lexicographical comparison
 *
 */
struct preprocessed_table {
  /// Type of table device view owner for the preprocessed table.
  using table_device_view_owner =
    std::invoke_result_t<decltype(table_device_view::create), table_view, rmm::cuda_stream_view>;

  /**
   * @brief Preprocess table for use with lexicographical comparison
   *
   * Sets up the table for use with lexicographical comparison. The resulting preprocessed table
   * can be passed to the constructor of `lexicographic::self_comparator` or
   * `lexicographic::two_table_comparator` to avoid preprocessing again.
   *
   * Note that the output of this factory function should not be used in `two_table_comparator` if
   * the input table contains lists-of-structs. In such cases, please use the overload
   * `preprocessed_table::create(table_view const&, table_view const&,...)` to preprocess both
   * input tables at the same time.
   *
   * @param table The table to preprocess
   * @param column_order Optional, host array the same length as a row that indicates the desired
   *        ascending/descending order of each column in a row. If empty, it is assumed all
   * columns are sorted in ascending order.
   * @param null_precedence Optional, an array having the same length as the number of columns in
   *        the input tables that indicates how null values compare to all other. If it is empty,
   *        the order `null_order::BEFORE` will be used for all columns.
   * @param stream The stream to launch kernels and h->d copies on while preprocessing
   * @return A shared pointer to a preprocessed table
   */
  static std::shared_ptr<preprocessed_table> create(table_view const& table,
                                                    host_span<order const> column_order,
                                                    host_span<null_order const> null_precedence,
                                                    rmm::cuda_stream_view stream);

  /**
   * @brief Preprocess tables for use with lexicographical comparison
   *
   * Sets up the tables for use with lexicographical comparison. The resulting preprocessed tables
   * can be passed to the constructor of `lexicographic::self_comparator` or
   * `lexicographic::two_table_comparator` to avoid preprocessing again.
   *
   * This factory function performs some extra operations to guarantee that its output can be used
   * in `two_table_comparator` for all cases.
   *
   * @param lhs The lhs table to preprocess
   * @param rhs The rhs table to preprocess
   * @param column_order Optional, host array the same length as a row that indicates the desired
   *        ascending/descending order of each column in a row. If empty, it is assumed all
   * columns are sorted in ascending order.
   * @param null_precedence Optional, an array having the same length as the number of columns in
   *        the input tables that indicates how null values compare to all other. If it is empty,
   *        the order `null_order::BEFORE` will be used for all columns.
   * @param stream The stream to launch kernels and h->d copies on while preprocessing
   * @return A pair of shared pointers to the preprocessed tables
   */
  static std::pair<std::shared_ptr<preprocessed_table>, std::shared_ptr<preprocessed_table>> create(
    table_view const& lhs,
    table_view const& rhs,
    host_span<order const> column_order,
    host_span<null_order const> null_precedence,
    rmm::cuda_stream_view stream);

 private:
  friend class self_comparator;       ///< Allow self_comparator to access private members
  friend class two_table_comparator;  ///< Allow two_table_comparator to access private members

  /**
   * @brief Create the output preprocessed table from intermediate preprocessing results
   *
   * @param preprocessed_input The table resulted from preprocessing
   * @param verticalized_col_depths The depths of each column resulting from decomposing struct
   *        columns in the original input table
   * @param transformed_columns Store the intermediate columns generated from transforming
   *        nested children columns into integers columns using `cudf::rank()`
   * @param column_order Optional, host array the same length as a row that indicates the desired
   *        ascending/descending order of each column in a row. If empty, it is assumed all
   * columns are sorted in ascending order.
   * @param null_precedence Optional, an array having the same length as the number of columns in
   *        the input tables that indicates how null values compare to all other. If it is empty,
   *        the order `null_order::BEFORE` will be used for all columns.
   * @param has_ranked_children Flag indicating if the input table was preprocessed to transform
   *        any nested child column into an integer column using `cudf::rank`
   * @param stream The stream to launch kernels and h->d copies on while preprocessing
   * @return A shared pointer to a preprocessed table
   */
  static std::shared_ptr<preprocessed_table> create(
    table_view const& preprocessed_input,
    std::vector<int>&& verticalized_col_depths,
    std::vector<std::unique_ptr<column>>&& transformed_columns,
    host_span<order const> column_order,
    host_span<null_order const> null_precedence,
    bool has_ranked_children,
    rmm::cuda_stream_view stream);

  /**
   * @brief Construct a preprocessed table for use with lexicographical comparison
   *
   * Sets up the table for use with lexicographical comparison. The resulting preprocessed table
   * can be passed to the constructor of `lexicographic::self_comparator` to avoid preprocessing
   * again.
   *
   * @param table The table to preprocess
   * @param column_order Optional, device array the same length as a row that indicates the
   * desired ascending/descending order of each column in a row. If empty, it is assumed all
   * columns are sorted in ascending order.
   * @param null_precedence Optional, device array the same length as a row and indicates how null
   *        values compare to all other for every column. If it is nullptr, then null precedence
   *        would be `null_order::BEFORE` for all columns.
   * @param depths The depths of each column resulting from decomposing struct columns.
   * @param dremel_data The dremel data for each list column. The length of this object is the
   *        number of list columns in the table.
   * @param dremel_device_views Device views into the dremel_data structs contained in the
   *        `dremel_data` parameter. For columns that are not list columns, this uvector will
   * should contain an empty `dremel_device_view`. As such, this uvector has as many elements as
   *        there are columns in the table (unlike the `dremel_data` parameter, which is only as
   *        long as the number of list columns).
   * @param transformed_columns Store the intermediate columns generated from transforming
   *        nested children columns into integers columns using `cudf::rank()`
   * @param has_ranked_children Flag indicating if the input table was preprocessed to transform
   *        any lists-of-structs column having floating-point children using `cudf::rank`
   */
  preprocessed_table(table_device_view_owner&& table,
                     rmm::device_uvector<order>&& column_order,
                     rmm::device_uvector<null_order>&& null_precedence,
                     rmm::device_uvector<size_type>&& depths,
                     std::vector<detail::dremel_data>&& dremel_data,
                     rmm::device_uvector<detail::dremel_device_view>&& dremel_device_views,
                     std::vector<std::unique_ptr<column>>&& transformed_columns,
                     bool has_ranked_children);

  preprocessed_table(table_device_view_owner&& table,
                     rmm::device_uvector<order>&& column_order,
                     rmm::device_uvector<null_order>&& null_precedence,
                     rmm::device_uvector<size_type>&& depths,
                     std::vector<std::unique_ptr<column>>&& transformed_columns,
                     bool has_ranked_children);

  /**
   * @brief Implicit conversion operator to a `table_device_view` of the preprocessed table.
   *
   * @return table_device_view
   */
  operator table_device_view() { return *_t; }

  /**
   * @brief Get a device array containing the desired order of each column in the preprocessed
   * table
   *
   * @return Device array containing respective column orders. If no explicit column orders were
   * specified during the creation of this object then this will be `nullopt`.
   */
  [[nodiscard]] std::optional<device_span<order const>> column_order() const
  {
    return _column_order.size() ? std::optional<device_span<order const>>(_column_order)
                                : std::nullopt;
  }

  /**
   * @brief Get a device array containing the desired null precedence of each column in the
   * preprocessed table
   *
   * @return Device array containing respective column null precedence. If no explicit column null
   * precedences were specified during the creation of this object then this will be `nullopt`.
   */
  [[nodiscard]] std::optional<device_span<null_order const>> null_precedence() const
  {
    return _null_precedence.size() ? std::optional<device_span<null_order const>>(_null_precedence)
                                   : std::nullopt;
  }

  /**
   * @brief Get a device array containing the depth of each column in the preprocessed table
   *
   * @see struct_linearize()
   *
   * @return std::optional<device_span<int const>> Device array containing respective column
   * depths. If there are no nested columns in the table then this will be `nullopt`.
   */
  [[nodiscard]] std::optional<device_span<int const>> depths() const
  {
    return _depths.size() ? std::optional<device_span<int const>>(_depths) : std::nullopt;
  }

  [[nodiscard]] device_span<detail::dremel_device_view const> dremel_device_views() const
  {
    if (_dremel_device_views.has_value()) {
      return device_span<detail::dremel_device_view const>(*_dremel_device_views);
    } else {
      return {};
    }
  }

  template <typename PhysicalElementComparator>
  void check_physical_element_comparator()
  {
    if constexpr (!std::is_same_v<PhysicalElementComparator, sorting_physical_element_comparator>) {
      CUDF_EXPECTS(!_has_ranked_children,
                   "The input table has nested type children and they were transformed using a "
                   "different type of physical element comparator.");
    }
  }

 private:
  table_device_view_owner const _t;
  rmm::device_uvector<order> const _column_order;
  rmm::device_uvector<null_order> const _null_precedence;
  rmm::device_uvector<size_type> const _depths;

  // Dremel encoding of list columns used for the comparison algorithm
  std::optional<std::vector<detail::dremel_data>> _dremel_data;
  std::optional<rmm::device_uvector<detail::dremel_device_view>> _dremel_device_views;

  // Intermediate columns generated from transforming nested children columns into
  // integers columns using `cudf::rank()`, need to be kept alive.
  std::vector<std::unique_ptr<column>> _transformed_columns;

  // Flag to record if the input table was preprocessed to transform any nested children column(s)
  // into integer column(s) using `cudf::rank`.
  bool const _has_ranked_children;
};

/**
 * @brief An owning object that can be used to lexicographically compare two rows of the same
 * table
 *
 * This class can take a table_view and preprocess certain columns to allow for lexicographical
 * comparison. The preprocessed table and temporary data required for the comparison are created
 * and owned by this class.
 *
 * Alternatively, `self_comparator` can be constructed from an existing
 * `shared_ptr<preprocessed_table>` when sharing the same table among multiple comparators.
 *
 * This class can then provide a functor object that can used on the device.
 * The object of this class must outlive the usage of the device functor.
 */
class self_comparator {
 public:
  /**
   * @brief Construct an owning object for performing a lexicographic comparison between two rows
   * of the same table.
   *
   * @param t The table to compare
   * @param column_order Optional, host array the same length as a row that indicates the desired
   *        ascending/descending order of each column in a row. If empty, it is assumed all
   * columns are sorted in ascending order.
   * @param null_precedence Optional, device array the same length as a row and indicates how null
   *        values compare to all other for every column. If empty, then null precedence would be
   *        `null_order::BEFORE` for all columns.
   * @param stream The stream to construct this object on. Not the stream that will be used for
   *        comparisons using this object.
   */
  self_comparator(table_view const& t,
                  host_span<order const> column_order         = {},
                  host_span<null_order const> null_precedence = {},
                  rmm::cuda_stream_view stream                = cudf::get_default_stream())
    : d_t{preprocessed_table::create(t, column_order, null_precedence, stream)}
  {
  }

  /**
   * @brief Construct an owning object for performing a lexicographic comparison between two rows
   * of the same preprocessed table.
   *
   * This constructor allows independently constructing a `preprocessed_table` and sharing it
   * among multiple comparators.
   *
   * @param t A table preprocessed for lexicographic comparison
   */
  self_comparator(std::shared_ptr<preprocessed_table> t) : d_t{std::move(t)} {}

  /**
   * @brief Return the binary operator for comparing rows in the table.
   *
   * Returns a binary callable, `F`, with signature `bool F(size_type, size_type)`.
   *
   * `F(i,j)` returns true if and only if row `i` compares lexicographically less than row `j`.
   *
   * @note The operator overloads in sub-class `element_comparator` are templated via the
   *       `type_dispatcher` to help select an overload instance for each column in a table.
   *       So, `cudf::is_nested<Element>` will return `true` if the table has nested-type columns,
   *       but it will be a runtime error if template parameter `has_nested_columns != true`.
   *
   * @tparam has_nested_columns compile-time optimization for primitive types.
   *         This template parameter is to be used by the developer by querying
   *         `cudf::detail::has_nested_columns(input)`. `true` compiles operator
   *         overloads for nested types, while `false` only compiles operator
   *         overloads for primitive types.
   * @tparam Nullate A cudf::nullate type describing whether to check for nulls.
   * @tparam PhysicalElementComparator A relational comparator functor that compares individual
   *         values rather than logical elements, defaults to `NaN` aware relational comparator
   *         that evaluates `NaN` as greater than all other values.
   * @throw cudf::logic_error if the input table was preprocessed to transform any nested children
   *        columns into integer columns but `PhysicalElementComparator` is not
   *        `sorting_physical_element_comparator`.
   * @param nullate Indicates if any input column contains nulls.
   * @param comparator Physical element relational comparison functor.
   * @return A binary callable object.
   */
  template <bool has_nested_columns,
            typename Nullate,
            typename PhysicalElementComparator = sorting_physical_element_comparator>
  auto less(Nullate nullate = {}, PhysicalElementComparator comparator = {}) const
  {
    d_t->check_physical_element_comparator<PhysicalElementComparator>();

    return less_comparator{
      device_row_comparator<has_nested_columns, Nullate, PhysicalElementComparator>{
        nullate,
        *d_t,
        *d_t,
        d_t->dremel_device_views(),
        d_t->dremel_device_views(),
        d_t->depths(),
        d_t->column_order(),
        d_t->null_precedence(),
        comparator}};
  }

  /// @copydoc less()
  template <bool has_nested_columns,
            typename Nullate,
            typename PhysicalElementComparator = sorting_physical_element_comparator>
  auto less_equivalent(Nullate nullate = {}, PhysicalElementComparator comparator = {}) const
  {
    d_t->check_physical_element_comparator<PhysicalElementComparator>();

    return less_equivalent_comparator{
      device_row_comparator<has_nested_columns, Nullate, PhysicalElementComparator>{
        nullate,
        *d_t,
        *d_t,
        d_t->dremel_device_views(),
        d_t->dremel_device_views(),
        d_t->depths(),
        d_t->column_order(),
        d_t->null_precedence(),
        comparator}};
  }

 private:
  std::shared_ptr<preprocessed_table> d_t;
};

// @cond
template <typename Comparator>
struct strong_index_comparator_adapter {
  strong_index_comparator_adapter(Comparator const& comparator) : comparator{comparator} {}

  __device__ constexpr weak_ordering operator()(lhs_index_type const lhs_index,
                                                rhs_index_type const rhs_index) const noexcept
  {
    return comparator(static_cast<cudf::size_type>(lhs_index),
                      static_cast<cudf::size_type>(rhs_index));
  }

  __device__ constexpr weak_ordering operator()(rhs_index_type const rhs_index,
                                                lhs_index_type const lhs_index) const noexcept
  {
    auto const left_right_ordering =
      comparator(static_cast<cudf::size_type>(lhs_index), static_cast<cudf::size_type>(rhs_index));

    // Invert less/greater values to reflect right to left ordering
    if (left_right_ordering == weak_ordering::LESS) {
      return weak_ordering::GREATER;
    } else if (left_right_ordering == weak_ordering::GREATER) {
      return weak_ordering::LESS;
    }
    return weak_ordering::EQUIVALENT;
  }

  Comparator const comparator;
};
// @endcond

/**
 * @brief An owning object that can be used to lexicographically compare rows of two different
 * tables
 *
 * This class takes two table_views and preprocesses certain columns to allow for lexicographical
 * comparison. The preprocessed table and temporary data required for the comparison are created
 * and owned by this class.
 *
 * Alternatively, `two_table_comparator` can be constructed from two existing
 * `shared_ptr<preprocessed_table>`s when sharing the same tables among multiple comparators.
 *
 * This class can then provide a functor object that can used on the device.
 * The object of this class must outlive the usage of the device functor.
 */
class two_table_comparator {
 public:
  /**
   * @brief Construct an owning object for performing a lexicographic comparison between rows of
   * two different tables.
   *
   * The left and right table are expected to have the same number of columns
   * and data types for each column.
   *
   * @param left The left table to compare
   * @param right The right table to compare
   * @param column_order Optional, host array the same length as a row that indicates the desired
   *        ascending/descending order of each column in a row. If empty, it is assumed all
   * columns are sorted in ascending order.
   * @param null_precedence Optional, device array the same length as a row and indicates how null
   *        values compare to all other for every column. If empty, then null precedence would be
   *        `null_order::BEFORE` for all columns.
   * @param stream The stream to construct this object on. Not the stream that will be used for
   *        comparisons using this object.
   */
  two_table_comparator(table_view const& left,
                       table_view const& right,
                       host_span<order const> column_order         = {},
                       host_span<null_order const> null_precedence = {},
                       rmm::cuda_stream_view stream                = cudf::get_default_stream());

  /**
   * @brief Construct an owning object for performing a lexicographic comparison between two rows
   * of the same preprocessed table.
   *
   * This constructor allows independently constructing a `preprocessed_table` and sharing it
   * among multiple comparators.
   *
   * The preprocessed_table(s) should have been pre-generated together using the factory function
   * `preprocessed_table::create(table_view const&, table_view const&)`. Otherwise, the comparison
   * results between two tables may be incorrect.
   *
   * @param left A table preprocessed for lexicographic comparison
   * @param right A table preprocessed for lexicographic comparison
   */
  two_table_comparator(std::shared_ptr<preprocessed_table> left,
                       std::shared_ptr<preprocessed_table> right)
    : d_left_table{std::move(left)}, d_right_table{std::move(right)}
  {
  }

  /**
   * @brief Return the binary operator for comparing rows in the table.
   *
   * Returns a binary callable, `F`, with signatures
   * `bool F(lhs_index_type, rhs_index_type)` and
   * `bool F(rhs_index_type, lhs_index_type)`.
   *
   * `F(lhs_index_type i, rhs_index_type j)` returns true if and only if row
   * `i` of the left table compares lexicographically less than row `j` of the
   * right table.
   *
   * Similarly, `F(rhs_index_type i, lhs_index_type j)` returns true if and
   * only if row `i` of the right table compares lexicographically less than row
   * `j` of the left table.
   *
   * @note The operator overloads in sub-class `element_comparator` are templated via the
   *       `type_dispatcher` to help select an overload instance for each column in a table.
   *       So, `cudf::is_nested<Element>` will return `true` if the table has nested-type columns,
   *       but it will be a runtime error if template parameter `has_nested_columns != true`.
   *
   * @tparam has_nested_columns compile-time optimization for primitive types.
   *         This template parameter is to be used by the developer by querying
   *         `cudf::detail::has_nested_columns(input)`. `true` compiles operator
   *         overloads for nested types, while `false` only compiles operator
   *         overloads for primitive types.
   * @tparam Nullate A cudf::nullate type describing whether to check for nulls.
   * @tparam PhysicalElementComparator A relational comparator functor that compares individual
   *         values rather than logical elements, defaults to `NaN` aware relational comparator
   *         that evaluates `NaN` as greater than all other values.
   * @throw cudf::logic_error if the input tables were preprocessed to transform any nested
   * children columns into integer columns but `PhysicalElementComparator` is not
   *        `sorting_physical_element_comparator`.
   * @param nullate Indicates if any input column contains nulls.
   * @param comparator Physical element relational comparison functor.
   * @return A binary callable object.
   */
  template <bool has_nested_columns,
            typename Nullate,
            typename PhysicalElementComparator = sorting_physical_element_comparator>
  auto less(Nullate nullate = {}, PhysicalElementComparator comparator = {}) const
  {
    d_left_table->check_physical_element_comparator<PhysicalElementComparator>();
    d_right_table->check_physical_element_comparator<PhysicalElementComparator>();

    return less_comparator{strong_index_comparator_adapter{
      device_row_comparator<has_nested_columns, Nullate, PhysicalElementComparator>{
        nullate,
        *d_left_table,
        *d_right_table,
        d_left_table->dremel_device_views(),
        d_right_table->dremel_device_views(),
        d_left_table->depths(),
        d_left_table->column_order(),
        d_left_table->null_precedence(),
        comparator}}};
  }

  /// @copydoc less()
  template <bool has_nested_columns,
            typename Nullate,
            typename PhysicalElementComparator = sorting_physical_element_comparator>
  auto less_equivalent(Nullate nullate = {}, PhysicalElementComparator comparator = {}) const
  {
    d_left_table->check_physical_element_comparator<PhysicalElementComparator>();
    d_right_table->check_physical_element_comparator<PhysicalElementComparator>();

    return less_equivalent_comparator{strong_index_comparator_adapter{
      device_row_comparator<has_nested_columns, Nullate, PhysicalElementComparator>{
        nullate,
        *d_left_table,
        *d_right_table,
        d_left_table->dremel_device_views(),
        d_right_table->dremel_device_views(),
        d_left_table->depths(),
        d_left_table->column_order(),
        d_left_table->null_precedence(),
        comparator}}};
  }

 private:
  std::shared_ptr<preprocessed_table> d_left_table;
  std::shared_ptr<preprocessed_table> d_right_table;
};

}  // namespace lexicographic

namespace hash {
class row_hasher;
}  // namespace hash

namespace equality {

/**
 * @brief Equality comparator functor that compares physical values rather than logical
 * elements like lists, strings, or structs. It evaluates `NaN` not equal to all other values for
 * IEEE-754 compliance.
 */
struct physical_equality_comparator {
  /**
   * @brief Operator for equality comparisons.
   *
   * Note that `NaN != NaN`, following IEEE-754.
   *
   * @param lhs First element
   * @param rhs Second element
   * @return `true` if `lhs == rhs` else `false`
   */
  template <typename Element>
  __device__ constexpr bool operator()(Element const lhs, Element const rhs) const noexcept
  {
    return lhs == rhs;
  }
};

/**
 * @brief Equality comparator functor that compares physical values rather than logical
 * elements like lists, strings, or structs. It evaluates `NaN` as equal to other `NaN`s.
 */
struct nan_equal_physical_equality_comparator {
  /**
   * @brief Operator for equality comparison of non-floating point values.
   *
   * @param lhs First element
   * @param rhs Second element
   * @return `true` if `lhs == rhs` else `false`
   */
  template <typename Element, CUDF_ENABLE_IF(not std::is_floating_point_v<Element>)>
  __device__ constexpr bool operator()(Element const lhs, Element const rhs) const noexcept
  {
    return lhs == rhs;
  }

  /**
   * @brief Operator for equality comparison of floating point values.
   *
   * Note that `NaN == NaN`.
   *
   * @param lhs First element
   * @param rhs Second element
   * @return `true` if `lhs` == `rhs` else `false`
   */
  template <typename Element, CUDF_ENABLE_IF(std::is_floating_point_v<Element>)>
  __device__ constexpr bool operator()(Element const lhs, Element const rhs) const noexcept
  {
    return isnan(lhs) and isnan(rhs) ? true : lhs == rhs;
  }
};

/**
 * @brief Computes the equality comparison between 2 rows.
 *
 * Equality is determined by comparing rows element by element. The first mismatching element
 * returns false, representing unequal rows. If the rows are compared without mismatched elements,
 * the rows are equal.
 *
 * @note The operator overloads in sub-class `element_comparator` are templated via the
 *        `type_dispatcher` to help select an overload instance for each column in a table.
 *        So, `cudf::is_nested<Element>` will return `true` if the table has nested-type columns,
 *        but it will be a runtime error if template parameter `has_nested_columns != true`.
 *
 * @tparam has_nested_columns compile-time optimization for primitive types.
 *         This template parameter is to be used by the developer by querying
 *         `cudf::detail::has_nested_columns(input)`. `true` compiles operator
 *         overloads for nested types, while `false` only compiles operator
 *         overloads for primitive types.
 * @tparam Nullate A cudf::nullate type describing whether to check for nulls.
 * @tparam PhysicalEqualityComparator A equality comparator functor that compares individual
 * values rather than logical elements, defaults to a comparator for which `NaN == NaN`.
 */
template <bool has_nested_columns,
          typename Nullate,
          typename PhysicalEqualityComparator = nan_equal_physical_equality_comparator>
class device_row_comparator {
  friend class self_comparator;       ///< Allow self_comparator to access private members
  friend class two_table_comparator;  ///< Allow two_table_comparator to access private members

 public:
  /**
   * @brief Checks whether the row at `lhs_index` in the `lhs` table is equal to the row at
   * `rhs_index` in the `rhs` table.
   *
   * @param lhs_index The index of the row in the `lhs` table to examine
   * @param rhs_index The index of the row in the `rhs` table to examine
   * @return `true` if row from the `lhs` table is equal to the row in the `rhs` table
   */
  __device__ constexpr bool operator()(size_type const lhs_index,
                                       size_type const rhs_index) const noexcept
  {
    auto equal_elements = [=](column_device_view l, column_device_view r) {
      return cudf::type_dispatcher(
        l.type(),
        element_comparator{check_nulls, l, r, nulls_are_equal, comparator},
        lhs_index,
        rhs_index);
    };

    return thrust::equal(thrust::seq, lhs.begin(), lhs.end(), rhs.begin(), equal_elements);
  }

 private:
  /**
   * @brief Construct a function object for performing equality comparison between the rows of two
   * tables.
   *
   * @param check_nulls Indicates if any input column contains nulls.
   * @param lhs The first table
   * @param rhs The second table (may be the same table as `lhs`)
   * @param nulls_are_equal Indicates if two null elements are treated as equivalent
   * @param comparator Physical element equality comparison functor.
   */
  device_row_comparator(Nullate check_nulls,
                        table_device_view lhs,
                        table_device_view rhs,
                        null_equality nulls_are_equal         = null_equality::EQUAL,
                        PhysicalEqualityComparator comparator = {}) noexcept
    : lhs{lhs},
      rhs{rhs},
      check_nulls{check_nulls},
      nulls_are_equal{nulls_are_equal},
      comparator{comparator}
  {
  }

  /**
   * @brief Performs an equality comparison between two elements in two columns.
   */
  class element_comparator {
   public:
    /**
     * @brief Construct type-dispatched function object for comparing equality
     * between two elements.
     *
     * @note `lhs` and `rhs` may be the same.
     *
     * @param check_nulls Indicates if either input column contains nulls.
     * @param lhs The column containing the first element
     * @param rhs The column containing the second element (may be the same as lhs)
     * @param nulls_are_equal Indicates if two null elements are treated as equivalent
     * @param comparator Physical element equality comparison functor.
     */
    __device__ element_comparator(Nullate check_nulls,
                                  column_device_view lhs,
                                  column_device_view rhs,
                                  null_equality nulls_are_equal         = null_equality::EQUAL,
                                  PhysicalEqualityComparator comparator = {}) noexcept
      : lhs{lhs},
        rhs{rhs},
        check_nulls{check_nulls},
        nulls_are_equal{nulls_are_equal},
        comparator{comparator}
    {
    }

    /**
     * @brief Compares the specified elements for equality.
     *
     * @param lhs_element_index The index of the first element
     * @param rhs_element_index The index of the second element
     * @return True if lhs and rhs are equal or if both lhs and rhs are null and nulls are
     * considered equal (`nulls_are_equal` == `null_equality::EQUAL`)
     */
    template <typename Element, CUDF_ENABLE_IF(cudf::is_equality_comparable<Element, Element>())>
    __device__ bool operator()(size_type const lhs_element_index,
                               size_type const rhs_element_index) const noexcept
    {
      if (check_nulls) {
        bool const lhs_is_null{lhs.is_null(lhs_element_index)};
        bool const rhs_is_null{rhs.is_null(rhs_element_index)};
        if (lhs_is_null and rhs_is_null) {
          return nulls_are_equal == null_equality::EQUAL;
        } else if (lhs_is_null != rhs_is_null) {
          return false;
        }
      }

      return comparator(lhs.element<Element>(lhs_element_index),
                        rhs.element<Element>(rhs_element_index));
    }

    template <typename Element,
              CUDF_ENABLE_IF(not cudf::is_equality_comparable<Element, Element>() and
                             (not has_nested_columns or not cudf::is_nested<Element>())),
              typename... Args>
    __device__ bool operator()(Args...)
    {
      CUDF_UNREACHABLE("Attempted to compare elements of uncomparable types.");
    }

    template <typename Element, CUDF_ENABLE_IF(has_nested_columns and cudf::is_nested<Element>())>
    __device__ bool operator()(size_type const lhs_element_index,
                               size_type const rhs_element_index) const noexcept
    {
      column_device_view lcol = lhs.slice(lhs_element_index, 1);
      column_device_view rcol = rhs.slice(rhs_element_index, 1);
      while (lcol.type().id() == type_id::STRUCT || lcol.type().id() == type_id::LIST) {
        if (check_nulls) {
          auto lvalid = detail::make_validity_iterator<true>(lcol);
          auto rvalid = detail::make_validity_iterator<true>(rcol);
          if (nulls_are_equal == null_equality::UNEQUAL) {
            if (thrust::any_of(
                  thrust::seq, lvalid, lvalid + lcol.size(), thrust::logical_not<bool>()) or
                thrust::any_of(
                  thrust::seq, rvalid, rvalid + rcol.size(), thrust::logical_not<bool>())) {
              return false;
            }
          } else {
            if (not thrust::equal(thrust::seq, lvalid, lvalid + lcol.size(), rvalid)) {
              return false;
            }
          }
        }
        if (lcol.type().id() == type_id::STRUCT) {
          if (lcol.num_child_columns() == 0) { return true; }
          // Non-empty structs are assumed to be decomposed and contain only one child
          lcol = detail::structs_column_device_view(lcol).get_sliced_child(0);
          rcol = detail::structs_column_device_view(rcol).get_sliced_child(0);
        } else if (lcol.type().id() == type_id::LIST) {
          auto l_list_col = detail::lists_column_device_view(lcol);
          auto r_list_col = detail::lists_column_device_view(rcol);

          auto lsizes = make_list_size_iterator(l_list_col);
          auto rsizes = make_list_size_iterator(r_list_col);
          if (not thrust::equal(thrust::seq, lsizes, lsizes + lcol.size(), rsizes)) {
            return false;
          }

          lcol = l_list_col.get_sliced_child();
          rcol = r_list_col.get_sliced_child();
          if (lcol.size() != rcol.size()) { return false; }
        }
      }

      auto comp = column_comparator{
        element_comparator{check_nulls, lcol, rcol, nulls_are_equal, comparator}, lcol.size()};
      return type_dispatcher<dispatch_void_if_nested>(lcol.type(), comp);
    }

   private:
    /**
     * @brief Serially compare two columns for equality.
     *
     * When we want to get the equivalence of two columns by serially comparing all elements in
     * one column with the corresponding elements in the other column, this saves us from type
     * dispatching for each individual element in the range
     */
    struct column_comparator {
      element_comparator const comp;
      size_type const size;

      /**
       * @brief Serially compare two columns for equality.
       *
       * @return True if ALL elements compare equal, false otherwise
       */
      template <typename Element, CUDF_ENABLE_IF(cudf::is_equality_comparable<Element, Element>())>
      __device__ bool operator()() const noexcept
      {
        return thrust::all_of(thrust::seq,
                              thrust::make_counting_iterator(0),
                              thrust::make_counting_iterator(0) + size,
                              [=](auto i) { return comp.template operator()<Element>(i, i); });
      }

      template <typename Element,
                CUDF_ENABLE_IF(not cudf::is_equality_comparable<Element, Element>()),
                typename... Args>
      __device__ bool operator()(Args...) const noexcept
      {
        CUDF_UNREACHABLE("Attempted to compare elements of uncomparable types.");
      }
    };

    column_device_view const lhs;
    column_device_view const rhs;
    Nullate const check_nulls;
    null_equality const nulls_are_equal;
    PhysicalEqualityComparator const comparator;
  };

  table_device_view const lhs;
  table_device_view const rhs;
  Nullate const check_nulls;
  null_equality const nulls_are_equal;
  PhysicalEqualityComparator const comparator;
};

/**
 * @brief Preprocessed table for use with row equality comparison or row hashing
 *
 */
struct preprocessed_table {
  /**
   * @brief Factory to construct preprocessed_table for use with
   * row equality comparison or row hashing
   *
   * Sets up the table for use with row equality comparison or row hashing. The resulting
   * preprocessed table can be passed to the constructor of `equality::self_comparator` to
   * avoid preprocessing again.
   *
   * @param table The table to preprocess
   * @param stream The cuda stream to use while preprocessing.
   * @return A preprocessed table as shared pointer
   */
  static std::shared_ptr<preprocessed_table> create(table_view const& table,
                                                    rmm::cuda_stream_view stream);

 private:
  friend class self_comparator;       ///< Allow self_comparator to access private members
  friend class two_table_comparator;  ///< Allow two_table_comparator to access private members
  friend class hash::row_hasher;      ///< Allow row_hasher to access private members
  using table_device_view_owner =
    std::invoke_result_t<decltype(table_device_view::create), table_view, rmm::cuda_stream_view>;

  preprocessed_table(table_device_view_owner&& table,
                     std::vector<rmm::device_buffer>&& null_buffers,
                     std::vector<std::unique_ptr<column>>&& tmp_columns)
    : _t(std::move(table)),
      _null_buffers(std::move(null_buffers)),
      _tmp_columns(std::move(tmp_columns))
  {
  }

  /**
   * @brief Implicit conversion operator to a `table_device_view` of the preprocessed table.
   *
   * @return table_device_view
   */
  operator table_device_view() { return *_t; }

  table_device_view_owner _t;
  std::vector<rmm::device_buffer> _null_buffers;
  std::vector<std::unique_ptr<column>> _tmp_columns;
};

/**
 * @brief Comparator for performing equality comparisons between two rows of the same table.
 *
 */
class self_comparator {
 public:
  /**
   * @brief Construct an owning object for performing equality comparisons between two rows of the
   * same table.
   *
   * @param t The table to compare
   * @param stream The stream to construct this object on. Not the stream that will be used for
   * comparisons using this object.
   */
  self_comparator(table_view const& t, rmm::cuda_stream_view stream)
    : d_t(preprocessed_table::create(t, stream))
  {
  }

  /**
   * @brief Construct an owning object for performing equality comparisons between two rows of the
   * same table.
   *
   * This constructor allows independently constructing a `preprocessed_table` and sharing it
   * among multiple comparators.
   *
   * @param t A table preprocessed for equality comparison
   */
  self_comparator(std::shared_ptr<preprocessed_table> t) : d_t{std::move(t)} {}

  /**
   * @brief Get the comparison operator to use on the device
   *
   * Returns a binary callable, `F`, with signature `bool F(size_type, size_type)`.
   *
   * `F(i,j)` returns true if and only if row `i` compares equal to row `j`.
   *
   * @note The operator overloads in sub-class `element_comparator` are templated via the
   *        `type_dispatcher` to help select an overload instance for each column in a table.
   *        So, `cudf::is_nested<Element>` will return `true` if the table has nested-type
   * columns, but it will be a runtime error if template parameter `has_nested_columns != true`.
   *
   * @tparam has_nested_columns compile-time optimization for primitive types.
   *         This template parameter is to be used by the developer by querying
   *         `cudf::detail::has_nested_columns(input)`. `true` compiles operator
   *         overloads for nested types, while `false` only compiles operator
   *         overloads for primitive types.
   * @tparam Nullate A cudf::nullate type describing whether to check for nulls.
   * @tparam PhysicalEqualityComparator A equality comparator functor that compares individual
   * values rather than logical elements, defaults to a comparator for which `NaN == NaN`.
   * @param nullate Indicates if any input column contains nulls.
   * @param nulls_are_equal Indicates if nulls are equal.
   * @param comparator Physical element equality comparison functor.
   * @return A binary callable object
   */
  template <bool has_nested_columns,
            typename Nullate,
            typename PhysicalEqualityComparator = nan_equal_physical_equality_comparator>
  auto equal_to(Nullate nullate                       = {},
                null_equality nulls_are_equal         = null_equality::EQUAL,
                PhysicalEqualityComparator comparator = {}) const noexcept
  {
    return device_row_comparator<has_nested_columns, Nullate, PhysicalEqualityComparator>{
      nullate, *d_t, *d_t, nulls_are_equal, comparator};
  }

 private:
  std::shared_ptr<preprocessed_table> d_t;
};

// @cond
template <typename Comparator>
struct strong_index_comparator_adapter {
  strong_index_comparator_adapter(Comparator const& comparator) : comparator{comparator} {}

  __device__ constexpr bool operator()(lhs_index_type const lhs_index,
                                       rhs_index_type const rhs_index) const noexcept
  {
    return comparator(static_cast<cudf::size_type>(lhs_index),
                      static_cast<cudf::size_type>(rhs_index));
  }

  __device__ constexpr bool operator()(rhs_index_type const rhs_index,
                                       lhs_index_type const lhs_index) const noexcept
  {
    return this->operator()(lhs_index, rhs_index);
  }

  Comparator const comparator;
};
// @endcond

/**
 * @brief An owning object that can be used to equality compare rows of two different tables.
 *
 * This class takes two table_views and preprocesses certain columns to allow for equality
 * comparison. The preprocessed table and temporary data required for the comparison are created
 * and owned by this class.
 *
 * Alternatively, `two_table_comparator` can be constructed from two existing
 * `shared_ptr<preprocessed_table>`s when sharing the same tables among multiple comparators.
 *
 * This class can then provide a functor object that can used on the device.
 * The object of this class must outlive the usage of the device functor.
 */
class two_table_comparator {
 public:
  /**
   * @brief Construct an owning object for performing equality comparisons between two rows from
   * two tables.
   *
   * The left and right table are expected to have the same number of columns and data types for
   * each column.
   *
   * @param left The left table to compare.
   * @param right The right table to compare.
   * @param stream The stream to construct this object on. Not the stream that will be used for
   * comparisons using this object.
   */
  two_table_comparator(table_view const& left,
                       table_view const& right,
                       rmm::cuda_stream_view stream);

  /**
   * @brief Construct an owning object for performing equality comparisons between two rows from
   * two tables.
   *
   * This constructor allows independently constructing a `preprocessed_table` and sharing it
   * among multiple comparators.
   *
   * @param left The left table preprocessed for equality comparison.
   * @param right The right table preprocessed for equality comparison.
   */
  two_table_comparator(std::shared_ptr<preprocessed_table> left,
                       std::shared_ptr<preprocessed_table> right)
    : d_left_table{std::move(left)}, d_right_table{std::move(right)}
  {
  }

  /**
   * @brief Return the binary operator for comparing rows in the table.
   *
   * Returns a binary callable, `F`, with signatures `bool F(lhs_index_type, rhs_index_type)` and
   * `bool F(rhs_index_type, lhs_index_type)`.
   *
   * `F(lhs_index_type i, rhs_index_type j)` returns true if and only if row `i` of the left table
   * compares equal to row `j` of the right table.
   *
   * Similarly, `F(rhs_index_type i, lhs_index_type j)` returns true if and only if row `i` of the
   * right table compares equal to row `j` of the left table.
   *
   * @note The operator overloads in sub-class `element_comparator` are templated via the
   *        `type_dispatcher` to help select an overload instance for each column in a table.
   *        So, `cudf::is_nested<Element>` will return `true` if the table has nested-type
   * columns, but it will be a runtime error if template parameter `has_nested_columns != true`.
   *
   * @tparam has_nested_columns compile-time optimization for primitive types.
   *         This template parameter is to be used by the developer by querying
   *         `cudf::detail::has_nested_columns(input)`. `true` compiles operator
   *         overloads for nested types, while `false` only compiles operator
   *         overloads for primitive types.
   * @tparam Nullate A cudf::nullate type describing whether to check for nulls.
   * @tparam PhysicalEqualityComparator A equality comparator functor that compares individual
   * values rather than logical elements, defaults to a `NaN == NaN` equality comparator.
   * @param nullate Indicates if any input column contains nulls.
   * @param nulls_are_equal Indicates if nulls are equal.
   * @param comparator Physical element equality comparison functor.
   * @return A binary callable object
   */
  template <bool has_nested_columns,
            typename Nullate,
            typename PhysicalEqualityComparator = nan_equal_physical_equality_comparator>
  auto equal_to(Nullate nullate                       = {},
                null_equality nulls_are_equal         = null_equality::EQUAL,
                PhysicalEqualityComparator comparator = {}) const noexcept
  {
    return strong_index_comparator_adapter{
      device_row_comparator<has_nested_columns, Nullate, PhysicalEqualityComparator>(
        nullate, *d_left_table, *d_right_table, nulls_are_equal, comparator)};
  }

 private:
  std::shared_ptr<preprocessed_table> d_left_table;
  std::shared_ptr<preprocessed_table> d_right_table;
};

}  // namespace equality

namespace hash {

/**
 * @brief Computes the hash value of an element in the given column.
 *
 * @tparam hash_function Hash functor to use for hashing elements.
 * @tparam Nullate A cudf::nullate type describing whether to check for nulls.
 */
template <template <typename> class hash_function, typename Nullate>
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
    uint32_t seed             = DEFAULT_HASH_SEED,
    hash_value_type null_hash = std::numeric_limits<hash_value_type>::max()) noexcept
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
  template <typename T, CUDF_ENABLE_IF(column_device_view::has_element_accessor<T>())>
  __device__ hash_value_type operator()(column_device_view const& col,
                                        size_type row_index) const noexcept
  {
    if (_check_nulls && col.is_null(row_index)) { return _null_hash; }
    return hash_function<T>{_seed}(col.element<T>(row_index));
  }

  /**
   * @brief Returns the hash value of the given element.
   *
   * @tparam T The type of the element to hash
   * @param col The column to hash
   * @param row_index The index of the row to hash
   * @return The hash value of the given element
   */
  template <typename T, CUDF_ENABLE_IF(not column_device_view::has_element_accessor<T>())>
  __device__ hash_value_type operator()(column_device_view const& col,
                                        size_type row_index) const noexcept
  {
    CUDF_UNREACHABLE("Unsupported type in hash.");
  }

  Nullate _check_nulls;        ///< Whether to check for nulls
  uint32_t _seed;              ///< The seed to use for hashing
  hash_value_type _null_hash;  ///< Hash value to use for null elements
};

/**
 * @brief Computes the hash value of a row in the given table.
 *
 * @tparam hash_function Hash functor to use for hashing elements.
 * @tparam Nullate A cudf::nullate type describing whether to check for nulls.
 */
template <template <typename> class hash_function,
          typename Nullate,
          template <typename>
          typename dispatch_conditional_t>
class device_row_hasher {
  friend class row_hasher;  ///< Allow row_hasher to access private members.
  template <cudf::type_id t>
  struct dispatch_storage_type {
    using type = nested_conditional_t<id_to_type<t>, device_storage_type_t, dispatch_conditional_t>;
  };

  template <cudf::type_id t>
  struct dispatch_void_if_nested {
    using type =
      nested_conditional_t<id_to_type<t>, dispatch_void_if_nested_t, dispatch_conditional_t>;
  };

 public:
  /**
   * @brief Return the hash value of a row in the given table.
   *
   * @param row_index The row index to compute the hash value of
   * @return The hash value of the row
   */
  __device__ auto operator()(size_type row_index) const noexcept
  {
    auto it = thrust::make_transform_iterator(_table.begin(), [=](auto const& column) {
      return cudf::type_dispatcher<dispatch_storage_type>(
        column.type(),
        element_hasher_adapter<hash_function>{_check_nulls, _seed},
        column,
        row_index);
    });

    // Hash each element and combine all the hash values together
    return detail::accumulate(it, it + _table.num_columns(), _seed, [](auto hash, auto h) {
      return cudf::hashing::detail::hash_combine(hash, h);
    });
  }

 private:
  /**
   * @brief Computes the hash value of an element in the given column.
   *
   * When the column is non-nested, this is a simple wrapper around the element_hasher.
   * When the column is nested, this uses the element_hasher to hash the shape and values of the
   * column.
   */
  template <template <typename> class hash_fn>
  class element_hasher_adapter {
    static constexpr hash_value_type NULL_HASH     = std::numeric_limits<hash_value_type>::max();
    static constexpr hash_value_type NON_NULL_HASH = 0;

   public:
    __device__ element_hasher_adapter(Nullate check_nulls, uint32_t seed) noexcept
      : _element_hasher(check_nulls, seed), _check_nulls(check_nulls)
    {
    }

    template <typename T, CUDF_ENABLE_IF(not cudf::is_nested<T>())>
    __device__ hash_value_type operator()(column_device_view const& col,
                                          size_type row_index) const noexcept
    {
      return _element_hasher.template operator()<T>(col, row_index);
    }

    template <typename T, CUDF_ENABLE_IF(cudf::is_nested<T>())>
    __device__ hash_value_type operator()(column_device_view const& col,
                                          size_type row_index) const noexcept
    {
      auto hash                   = hash_value_type{0};
      column_device_view curr_col = col.slice(row_index, 1);
      while (curr_col.type().id() == type_id::STRUCT || curr_col.type().id() == type_id::LIST) {
        if (_check_nulls) {
          auto validity_it = detail::make_validity_iterator<true>(curr_col);
          hash             = detail::accumulate(
            validity_it, validity_it + curr_col.size(), hash, [](auto hash, auto is_valid) {
              return cudf::hashing::detail::hash_combine(hash,
                                                         is_valid ? NON_NULL_HASH : NULL_HASH);
            });
        }
        if (curr_col.type().id() == type_id::STRUCT) {
          if (curr_col.num_child_columns() == 0) { return hash; }
          // Non-empty structs are assumed to be decomposed and contain only one child
          curr_col = detail::structs_column_device_view(curr_col).get_sliced_child(0);
        } else if (curr_col.type().id() == type_id::LIST) {
          auto list_col   = detail::lists_column_device_view(curr_col);
          auto list_sizes = make_list_size_iterator(list_col);
          hash            = detail::accumulate(
            list_sizes, list_sizes + list_col.size(), hash, [](auto hash, auto size) {
              return cudf::hashing::detail::hash_combine(hash, hash_fn<size_type>{}(size));
            });
          curr_col = list_col.get_sliced_child();
        }
      }
      for (int i = 0; i < curr_col.size(); ++i) {
        hash = cudf::hashing::detail::hash_combine(
          hash,
          type_dispatcher<dispatch_void_if_nested>(curr_col.type(), _element_hasher, curr_col, i));
      }
      return hash;
    }

    element_hasher<hash_fn, Nullate> const _element_hasher;
    Nullate const _check_nulls;
  };

  CUDF_HOST_DEVICE device_row_hasher(Nullate check_nulls,
                                     table_device_view t,
                                     uint32_t seed = DEFAULT_HASH_SEED) noexcept
    : _check_nulls{check_nulls}, _table{t}, _seed(seed)
  {
  }

  Nullate const _check_nulls;
  table_device_view const _table;
  uint32_t const _seed;
};

// Inject row::equality::preprocessed_table into the row::hash namespace
// As a result, row::equality::preprocessed_table and row::hash::preprocessed table are the same
// type and are interchangeable.
using preprocessed_table = row::equality::preprocessed_table;

/**
 * @brief Computes the hash value of a row in the given table.
 *
 */
class row_hasher {
 public:
  /**
   * @brief Construct an owning object for hashing the rows of a table
   *
   * @param t The table containing rows to hash
   * @param stream The stream to construct this object on. Not the stream that will be used for
   * comparisons using this object.
   */
  row_hasher(table_view const& t, rmm::cuda_stream_view stream)
    : d_t(preprocessed_table::create(t, stream))
  {
  }

  /**
   * @brief Construct an owning object for hashing the rows of a table from an existing
   * preprocessed_table
   *
   * This constructor allows independently constructing a `preprocessed_table` and sharing it
   * among multiple `row_hasher` and `equality::self_comparator` objects.
   *
   * @param t A table preprocessed for hashing or equality.
   */
  row_hasher(std::shared_ptr<preprocessed_table> t) : d_t{std::move(t)} {}

  /**
   * @brief Get the hash operator to use on the device
   *
   * Returns a unary callable, `F`, with signature `hash_function::hash_value_type F(size_type)`.
   *
   * `F(i)` returns the hash of row i.
   *
   * @tparam Nullate A cudf::nullate type describing whether to check for nulls
   * @param nullate Indicates if any input column contains nulls
   * @param seed The seed to use for the hash function
   * @return A hash operator to use on the device
   */
  template <template <typename> typename dispatch_cond = type_identity,
            template <typename> class hash_function    = cudf::hashing::detail::default_hash,
            template <template <typename> class, typename, template <typename> typename>
            class DeviceRowHasher = device_row_hasher,
            typename Nullate>
  DeviceRowHasher<hash_function, Nullate, dispatch_cond> device_hasher(
    Nullate nullate = {}, uint32_t seed = DEFAULT_HASH_SEED) const
  {
    return DeviceRowHasher<hash_function, Nullate, dispatch_cond>(nullate, *d_t, seed);
  }

 private:
  std::shared_ptr<preprocessed_table> d_t;
};

}  // namespace hash

}  // namespace row

}  // namespace experimental
}  // namespace cudf
