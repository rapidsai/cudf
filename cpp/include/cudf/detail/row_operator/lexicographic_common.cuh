/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/detail/row_operator/common_utils.cuh>
#include <cudf/lists/detail/dremel.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/std/optional>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cmath>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace CUDF_EXPORT cudf {
namespace detail::row::primitive {
class row_lexicographic_comparator;
}  // namespace detail::row::primitive

namespace detail::row::lexicographic {

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
  __device__ constexpr cudf::detail::weak_ordering operator()(Element const lhs,
                                                              Element const rhs) const noexcept
  {
    return cudf::detail::compare_elements(lhs, rhs);
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
  template <typename Element>
  __device__ constexpr cudf::detail::weak_ordering operator()(Element const lhs,
                                                              Element const rhs) const noexcept
    requires(not cuda::std::is_floating_point_v<Element>)
  {
    return cudf::detail::compare_elements(lhs, rhs);
  }

  /**
   * @brief Operator for relational comparison of floating point values.
   *
   * @param lhs First element
   * @param rhs Second element
   * @return Relation between elements
   */
  template <typename Element>
  __device__ constexpr cudf::detail::weak_ordering operator()(Element const lhs,
                                                              Element const rhs) const noexcept
    requires(cuda::std::is_floating_point_v<Element>)
  {
    if (isnan(lhs)) {
      return isnan(rhs) ? cudf::detail::weak_ordering::EQUIVALENT
                        : cudf::detail::weak_ordering::GREATER;
    } else if (isnan(rhs)) {
      return cudf::detail::weak_ordering::LESS;
    }

    return cudf::detail::compare_elements(lhs, rhs);
  }
};

/**
 * @brief Wraps and interprets the result of templated Comparator that returns a
 * cudf::detail::weak_ordering. Returns true if the cudf::detail::weak_ordering matches any of the
 * templated values.
 *
 * Note that this should never be used with only `cudf::detail::weak_ordering::EQUIVALENT`.
 * An equality comparator should be used instead for optimal performance.
 *
 * @tparam Comparator generic comparator that returns a cudf::detail::weak_ordering.
 * @tparam values cudf::detail::weak_ordering parameter pack of orderings to interpret as true
 */
template <typename Comparator, cudf::detail::weak_ordering... values>
struct weak_ordering_comparator_impl {
  static_assert(
    not((cudf::detail::weak_ordering::EQUIVALENT == values) && ...),
    "cudf::detail::weak_ordering_comparator should not be used for pure equality comparisons. The "
    "`row_equality_comparator` should be used instead");

  template <typename LhsType, typename RhsType>
  __device__ constexpr bool operator()(LhsType const lhs_index,
                                       RhsType const rhs_index) const noexcept
  {
    cudf::detail::weak_ordering const result = comparator(lhs_index, rhs_index);
    return ((result == values) || ...);
  }
  Comparator comparator;
};

/**
 * @brief Wraps and interprets the result of device_row_comparator, true if the result is
 * cudf::detail::weak_ordering::LESS meaning one row is lexicographically *less* than another row.
 *
 * @tparam Comparator generic comparator that returns a cudf::detail::weak_ordering
 */
template <typename Comparator>
struct less_comparator
  : weak_ordering_comparator_impl<Comparator, cudf::detail::weak_ordering::LESS> {
  /**
   * @brief Constructs a less_comparator
   *
   * @param comparator The comparator to wrap
   */
  less_comparator(Comparator const& comparator)
    : weak_ordering_comparator_impl<Comparator, cudf::detail::weak_ordering::LESS>{comparator}
  {
  }
};

/**
 * @brief Wraps and interprets the result of device_row_comparator, true if the result is
 * cudf::detail::weak_ordering::LESS or cudf::detail::weak_ordering::EQUIVALENT meaning one row is
 * lexicographically *less* than or *equivalent* to another row.
 *
 * @tparam Comparator generic comparator that returns a cudf::detail::weak_ordering
 */
template <typename Comparator>
struct less_equivalent_comparator
  : weak_ordering_comparator_impl<Comparator,
                                  cudf::detail::weak_ordering::LESS,
                                  cudf::detail::weak_ordering::EQUIVALENT> {
  /**
   * @brief Constructs a less_equivalent_comparator
   *
   * @param comparator The comparator to wrap
   */
  less_equivalent_comparator(Comparator const& comparator)
    : weak_ordering_comparator_impl<Comparator,
                                    cudf::detail::weak_ordering::LESS,
                                    cudf::detail::weak_ordering::EQUIVALENT>{comparator}
  {
  }
};

/**
 * @brief Preprocessed table for use with lexicographical comparison
 *
 */
struct preprocessed_table {
  using table_device_view_owner = std::invoke_result_t<decltype(table_device_view::create),
                                                       table_view,
                                                       cuda::stream_ref,
                                                       rmm::device_async_resource_ref>;

  /**
   * @brief Preprocess table for use with lexicographical comparison
   *
   * Sets up the table for use with lexicographical comparison. The resulting preprocessed table can
   * be passed to the constructor of `lexicographic::self_comparator` or
   * `lexicographic::two_table_comparator` to avoid preprocessing again.
   *
   * Note that the output of this factory function should not be used in `two_table_comparator` if
   * the input table contains lists-of-structs. In such cases, please use the overload
   * `preprocessed_table::create(table_view const&, table_view const&,...)` to preprocess both input
   * tables at the same time.
   *
   * @param table The table to preprocess
   * @param column_order Optional, host array the same length as a row that indicates the desired
   *        ascending/descending order of each column in a row. If empty, it is assumed all columns
   *        are sorted in ascending order.
   * @param null_precedence Optional, an array having the same length as the number of columns in
   *        the input tables that indicates how null values compare to all other. If it is empty,
   *        the order `null_order::BEFORE` will be used for all columns.
   * @param stream The stream to launch kernels and h->d copies on while preprocessing
   * @return A shared pointer to a preprocessed table
   */
  static std::shared_ptr<preprocessed_table> create(table_view const& table,
                                                    host_span<order const> column_order,
                                                    host_span<null_order const> null_precedence,
                                                    cuda::stream_ref stream);

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
   *        ascending/descending order of each column in a row. If empty, it is assumed all columns
   *        are sorted in ascending order.
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
    cuda::stream_ref stream);

 private:
  friend class self_comparator;
  friend class two_table_comparator;
  friend class ::cudf::detail::row::primitive::row_lexicographic_comparator;

  /**
   * @brief Create the output preprocessed table from intermediate preprocessing results
   *
   * @param preprocessed_input The table resulted from preprocessing
   * @param verticalized_col_depths The depths of each column resulting from decomposing struct
   *        columns in the original input table
   * @param transformed_columns Store the intermediate columns generated from transforming
   *        nested children columns into integers columns using `cudf::rank()`
   * @param column_order Optional, host array the same length as a row that indicates the desired
   *        ascending/descending order of each column in a row. If empty, it is assumed all columns
   *        are sorted in ascending order.
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
    cuda::stream_ref stream);

  /**
   * @brief Construct a preprocessed table for use with lexicographical comparison
   *
   * Sets up the table for use with lexicographical comparison. The resulting preprocessed table can
   * be passed to the constructor of `lexicographic::self_comparator` to avoid preprocessing again.
   *
   * @param table The table to preprocess
   * @param column_order Optional, device array the same length as a row that indicates the desired
   *        ascending/descending order of each column in a row. If empty, it is assumed all columns
   *        are sorted in ascending order.
   * @param null_precedence Optional, device array the same length as a row and indicates how null
   *        values compare to all other for every column. If it is nullptr, then null precedence
   *        would be `null_order::BEFORE` for all columns.
   * @param depths The depths of each column resulting from decomposing struct columns.
   * @param dremel_data The dremel data for each list column. The length of this object is the
   *        number of list columns in the table.
   * @param dremel_device_views Device views into the dremel_data structs contained in the
   *        `dremel_data` parameter. For columns that are not list columns, this uvector will should
   *        contain an empty `dremel_device_view`. As such, this uvector has as many elements as
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
   * @brief Get a device array containing the desired order of each column in the preprocessed table
   *
   * @return Device array containing respective column orders. If no explicit column orders were
   * specified during the creation of this object then this will be `nullopt`.
   */
  [[nodiscard]] cuda::std::optional<device_span<order const>> column_order() const
  {
    return _column_order.size() ? cuda::std::optional<device_span<order const>>(_column_order)
                                : cuda::std::nullopt;
  }

  /**
   * @brief Get a device array containing the desired null precedence of each column in the
   * preprocessed table
   *
   * @return Device array containing respective column null precedence. If no explicit column null
   * precedences were specified during the creation of this object then this will be `nullopt`.
   */
  [[nodiscard]] cuda::std::optional<device_span<null_order const>> null_precedence() const
  {
    return _null_precedence.size()
             ? cuda::std::optional<device_span<null_order const>>(_null_precedence)
             : cuda::std::nullopt;
  }

  /**
   * @brief Get a device array containing the depth of each column in the preprocessed table
   *
   * @see struct_linearize()
   *
   * @return std::optional<device_span<int const>> Device array containing respective column depths.
   * If there are no nested columns in the table then this will be `nullopt`.
   */
  [[nodiscard]] cuda::std::optional<device_span<int const>> depths() const
  {
    return _depths.size() ? cuda::std::optional<device_span<int const>>(_depths)
                          : cuda::std::nullopt;
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
    if constexpr (!cuda::std::is_same_v<PhysicalElementComparator,
                                        sorting_physical_element_comparator>) {
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

  cuda::std::optional<std::vector<detail::dremel_data>> _dremel_data;
  cuda::std::optional<rmm::device_uvector<detail::dremel_device_view>> _dremel_device_views;

  std::vector<std::unique_ptr<column>> _transformed_columns;

  bool const _has_ranked_children;
};

}  // namespace detail::row::lexicographic
}  // namespace CUDF_EXPORT cudf
