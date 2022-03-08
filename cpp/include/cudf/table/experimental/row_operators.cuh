/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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
#include <cudf/sorting.hpp>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <thrust/equal.h>
#include <thrust/swap.h>
#include <thrust/transform_reduce.h>

#include <cuda/std/tuple>
#include <cuda/std/utility>

#include <limits>
#include <memory>
#include <utility>

namespace cudf {
namespace experimental {

template <cudf::type_id t>
struct non_nested_id_to_type {
  using type = std::conditional_t<cudf::is_nested(data_type(t)), void, id_to_type<t>>;
};

namespace lexicographic_comparison {

/**
 * @brief Performs a relational comparison between two elements in two columns.
 *
 * @tparam Nullate A cudf::nullate type describing how to check for nulls.
 */
template <typename Nullate>
class element_relational_comparator {
 public:
  /**
   * @brief Construct type-dispatched function object for performing a
   * relational comparison between two elements.
   *
   * @note `lhs` and `rhs` may be the same.
   *
   * @param lhs The column containing the first element
   * @param rhs The column containing the second element (may be the same as lhs)
   * @param has_nulls Indicates if either input column contains nulls.
   * @param null_precedence Indicates how null values are ordered with other values
   */
  __host__ __device__ element_relational_comparator(Nullate has_nulls,
                                                    column_device_view lhs,
                                                    column_device_view rhs,
                                                    null_order null_precedence,
                                                    int depth = 0)
    : lhs{lhs}, rhs{rhs}, nulls{has_nulls}, null_precedence{null_precedence}, depth{depth}
  {
  }

  __host__ __device__ element_relational_comparator(Nullate has_nulls,
                                                    column_device_view lhs,
                                                    column_device_view rhs)
    : lhs{lhs}, rhs{rhs}, nulls{has_nulls}
  {
  }

  /**
   * @brief Performs a relational comparison between the specified elements
   *
   * @param lhs_element_index The index of the first element
   * @param rhs_element_index The index of the second element
   * @return Indicates the relationship between the elements in
   * the `lhs` and `rhs` columns.
   */
  template <typename Element,
            std::enable_if_t<cudf::is_relationally_comparable<Element, Element>()>* = nullptr>
  __device__ cuda::std::pair<weak_ordering, int> operator()(
    size_type lhs_element_index, size_type rhs_element_index) const noexcept
  {
    if (nulls) {
      bool const lhs_is_null{lhs.is_null(lhs_element_index)};
      bool const rhs_is_null{rhs.is_null(rhs_element_index)};

      if (lhs_is_null or rhs_is_null) {  // at least one is null
        return cuda::std::make_pair(null_compare(lhs_is_null, rhs_is_null, null_precedence), depth);
      }
    }

    return cuda::std::make_pair(relational_compare(lhs.element<Element>(lhs_element_index),
                                                   rhs.element<Element>(rhs_element_index)),
                                std::numeric_limits<int>::max());
  }

  template <typename Element,
            CUDF_ENABLE_IF(not cudf::is_relationally_comparable<Element, Element>() and
                           not std::is_same_v<Element, cudf::struct_view>)>
  __device__ cuda::std::pair<weak_ordering, int> operator()(size_type lhs_element_index,
                                                            size_type rhs_element_index)
  {
    cudf_assert(false && "Attempted to compare elements of uncomparable types.");
    return cuda::std::make_pair(weak_ordering::LESS, std::numeric_limits<int>::max());
  }

  template <typename Element,
            CUDF_ENABLE_IF(not cudf::is_relationally_comparable<Element, Element>() and
                           std::is_same_v<Element, cudf::struct_view>)>
  __device__ cuda::std::pair<weak_ordering, int> operator()(size_type lhs_element_index,
                                                            size_type rhs_element_index)
  {
    weak_ordering state{weak_ordering::EQUIVALENT};

    column_device_view lcol = lhs;
    column_device_view rcol = rhs;
    while (lcol.type().id() == type_id::STRUCT) {
      bool const lhs_is_null{lcol.is_null(lhs_element_index)};
      bool const rhs_is_null{rcol.is_null(rhs_element_index)};

      if (lhs_is_null or rhs_is_null) {  // atleast one is null
        state = null_compare(lhs_is_null, rhs_is_null, null_precedence);
        return cuda::std::make_pair(state, depth);
      }

      lcol = lcol.children()[0];
      rcol = rcol.children()[0];
      ++depth;
    }

    auto comparator = element_relational_comparator{nulls, lcol, rcol, null_precedence, depth};
    return cudf::type_dispatcher<non_nested_id_to_type>(
      lcol.type(), comparator, lhs_element_index, rhs_element_index);
  }

 private:
  column_device_view lhs;
  column_device_view rhs;
  Nullate nulls;
  null_order null_precedence{};
  int depth{};
};

/**
 * @brief Computes whether one row is lexicographically *less* than another row.
 *
 * Lexicographic ordering is determined by:
 * - Two rows are compared element by element.
 * - The first mismatching element defines which row is lexicographically less
 * or greater than the other.
 *
 * Lexicographic ordering is exactly equivalent to doing an alphabetical sort of
 * two words, for example, `aac` would be *less* than (or precede) `abb`. The
 * second letter in both words is the first non-equal letter, and `a < b`, thus
 * `aac < abb`.
 *
 * @tparam Nullate A cudf::nullate type describing how to check for nulls.
 */
template <typename Nullate>
class device_row_comparator {
  friend class self_comparator;

  /**
   * @brief Construct a function object for performing a lexicographic
   * comparison between the rows of two tables.
   *
   * @throws cudf::logic_error if `lhs.num_columns() != rhs.num_columns()`
   * @throws cudf::logic_error if column types of `lhs` and `rhs` are not comparable.
   *
   * @param lhs The first table
   * @param rhs The second table (may be the same table as `lhs`)
   * @param has_nulls Indicates if either input table contains columns with nulls.
   * @param column_order Optional, device array the same length as a row that
   * indicates the desired ascending/descending order of each column in a row.
   * If `nullptr`, it is assumed all columns are sorted in ascending order.
   * @param null_precedence Optional, device array the same length as a row
   * and indicates how null values compare to all other for every column. If
   * it is nullptr, then null precedence would be `null_order::BEFORE` for all
   * columns.
   */
  device_row_comparator(Nullate has_nulls,
                        table_device_view lhs,
                        table_device_view rhs,
                        std::optional<device_span<int const>> depth                  = std::nullopt,
                        std::optional<device_span<order const>> column_order         = std::nullopt,
                        std::optional<device_span<null_order const>> null_precedence = std::nullopt)
    : _lhs{lhs},
      _rhs{rhs},
      _nulls{has_nulls},
      _depth{depth},
      _column_order{column_order},
      _null_precedence{null_precedence}
  {
  }

 public:
  /**
   * @brief Checks whether the row at `lhs_index` in the `lhs` table compares
   * lexicographically less than the row at `rhs_index` in the `rhs` table.
   *
   * @param lhs_index The index of row in the `lhs` table to examine
   * @param rhs_index The index of the row in the `rhs` table to examine
   * @return `true` if row from the `lhs` table compares less than row in the
   * `rhs` table
   */
  __device__ bool operator()(size_type lhs_index, size_type rhs_index) const noexcept
  {
    int last_null_depth = std::numeric_limits<int>::max();
    for (size_type i = 0; i < _lhs.num_columns(); ++i) {
      int depth = _depth.has_value() ? (*_depth)[i] : 0;
      if (depth > last_null_depth) { continue; }

      bool ascending = _column_order.has_value() ? (*_column_order)[i] == order::ASCENDING : true;

      null_order null_precedence =
        _null_precedence.has_value() ? (*_null_precedence)[i] : null_order::BEFORE;

      auto comparator = element_relational_comparator{
        _nulls, _lhs.column(i), _rhs.column(i), null_precedence, depth};

      weak_ordering state;
      cuda::std::tie(state, last_null_depth) =
        cudf::type_dispatcher(_lhs.column(i).type(), comparator, lhs_index, rhs_index);

      if (state == weak_ordering::EQUIVALENT) { continue; }

      return state == (ascending ? weak_ordering::LESS : weak_ordering::GREATER);
    }
    return false;
  }

 private:
  table_device_view _lhs;
  table_device_view _rhs;
  Nullate _nulls{};
  std::optional<device_span<int const>> _depth;
  std::optional<device_span<order const>> _column_order;
  std::optional<device_span<null_order const>> _null_precedence;
};  // class row_lexicographic_comparator

struct preprocessed_table {
  preprocessed_table(table_view const& table,
                     host_span<order const> column_order,
                     host_span<null_order const> null_precedence,
                     rmm::cuda_stream_view stream);

  operator table_device_view() { return **d_t; }

  [[nodiscard]] std::optional<device_span<order const>> column_order() const
  {
    return d_column_order.size() ? std::optional<device_span<order const>>(d_column_order)
                                 : std::nullopt;
  }

  [[nodiscard]] std::optional<device_span<null_order const>> null_precedence() const
  {
    return d_null_precedence.size()
             ? std::optional<device_span<null_order const>>(d_null_precedence)
             : std::nullopt;
  }

  [[nodiscard]] std::optional<device_span<int const>> depths() const
  {
    return d_depths.size() ? std::optional<device_span<int const>>(d_depths) : std::nullopt;
  }

  [[nodiscard]] bool has_nulls() const { return _has_nulls; }

 private:
  using table_device_view_owner =
    std::invoke_result_t<decltype(table_device_view::create), table_view, rmm::cuda_stream_view>;

  std::unique_ptr<table_device_view_owner> d_t;
  rmm::device_uvector<order> d_column_order;
  rmm::device_uvector<null_order> d_null_precedence;
  rmm::device_uvector<size_type> d_depths;
  bool _has_nulls;
};

class self_comparator {
 public:
  self_comparator(table_view const& t,
                  host_span<order const> column_order,
                  host_span<null_order const> null_precedence,
                  rmm::cuda_stream_view stream)
    : d_t{std::make_shared<preprocessed_table>(t, column_order, null_precedence, stream)}
  {
  }

  self_comparator(std::shared_ptr<preprocessed_table> t) : d_t{std::move(t)} {}

  template <typename Nullate = nullate::DYNAMIC>
  device_row_comparator<Nullate> device_comparator()
  {
    if constexpr (std::is_same_v<Nullate, nullate::DYNAMIC>) {
      return device_row_comparator(Nullate{d_t->has_nulls()},
                                   *d_t,
                                   *d_t,
                                   d_t->depths(),
                                   d_t->column_order(),
                                   d_t->null_precedence());
    } else {
      return device_row_comparator<Nullate>(
        Nullate{}, *d_t, *d_t, d_t->depths(), d_t->column_order(), d_t->null_precedence());
    }
  }

 private:
  std::shared_ptr<preprocessed_table> d_t;
};

}  // namespace lexicographic_comparison
}  // namespace experimental
}  // namespace cudf
