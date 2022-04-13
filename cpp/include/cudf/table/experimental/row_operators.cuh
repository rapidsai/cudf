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
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/detail/utilities/hash_functions.cuh>
#include <cudf/lists/list_device_view.cuh>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/sorting.hpp>
#include <cudf/structs/structs_column_device_view.cuh>
#include <cudf/table/row_operators.cuh>
#include <cudf/table/table_device_view.cuh>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <thrust/equal.h>
#include <thrust/logical.h>
#include <thrust/swap.h>
#include <thrust/transform_reduce.h>

#include <cuda/std/tuple>
#include <cuda/std/utility>

#include <limits>
#include <memory>
#include <optional>
#include <utility>

namespace cudf {
namespace experimental {

/**
 * @brief A map from cudf::type_id to cudf type that excludes LIST and STRUCT types.
 *
 * To be used with type_dispatcher in place of the default map, when it is required that STRUCT and
 * LIST map to void. This is useful when we want to avoid recursion in a functor. For example, in
 * element_comparator, we have a specialization for STRUCT but the type_dispatcher in it is only
 * used to dispatch to the same functor for non-nested types. Even when we're guaranteed to not have
 * non-nested types at that point, the compiler doesn't know this and would try to create recursive
 * code which is very slow.
 *
 * Usage:
 * @code
 * type_dispatcher<dispatch_nested_to_void>(data_type(), functor{});
 * @endcode
 */
template <cudf::type_id t>
struct dispatch_void_if_nested {
  using type = std::conditional_t<cudf::is_nested(data_type(t)), void, id_to_type<t>>;
};

namespace row {

namespace lexicographic {

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
   * @param has_nulls Indicates if either input table contains columns with nulls.
   * @param lhs The first table
   * @param rhs The second table (may be the same table as `lhs`)
   * @param depth Optional, device array the same length as a row that contains starting depths of
   * columns if they're nested, and 0 otherwise.
   * @param column_order Optional, device array the same length as a row that indicates the desired
   * ascending/descending order of each column in a row. If `nullopt`, it is assumed all columns are
   * sorted in ascending order.
   * @param null_precedence Optional, device array the same length as a row and indicates how null
   * values compare to all other for every column. If `nullopt`, then null precedence would be
   * `null_order::BEFORE` for all columns.
   */
  device_row_comparator(
    Nullate has_nulls,
    table_device_view lhs,
    table_device_view rhs,
    std::optional<device_span<int const>> depth                  = std::nullopt,
    std::optional<device_span<order const>> column_order         = std::nullopt,
    std::optional<device_span<null_order const>> null_precedence = std::nullopt) noexcept
    : _lhs{lhs},
      _rhs{rhs},
      _nulls{has_nulls},
      _depth{depth},
      _column_order{column_order},
      _null_precedence{null_precedence}
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
     * @param has_nulls Indicates if either input column contains nulls.
     * @param lhs The column containing the first element
     * @param rhs The column containing the second element (may be the same as lhs)
     * @param null_precedence Indicates how null values are ordered with other values
     * @param depth The depth of the column if part of a nested column @see
     * preprocessed_table::depths
     */
    __device__ element_comparator(Nullate has_nulls,
                                  column_device_view lhs,
                                  column_device_view rhs,
                                  null_order null_precedence = null_order::BEFORE,
                                  int depth                  = 0)
      : _lhs{lhs}, _rhs{rhs}, _nulls{has_nulls}, _null_precedence{null_precedence}, _depth{depth}
    {
    }

    /**
     * @brief Performs a relational comparison between the specified elements
     *
     * @param lhs_element_index The index of the first element
     * @param rhs_element_index The index of the second element
     * @return Indicates the relationship between the elements in the `lhs` and `rhs` columns, along
     * with the depth at which a null value was encountered.
     */
    template <typename Element,
              CUDF_ENABLE_IF(cudf::is_relationally_comparable<Element, Element>())>
    __device__ cuda::std::pair<weak_ordering, int> operator()(
      size_type const lhs_element_index, size_type const rhs_element_index) const noexcept
    {
      if (_nulls) {
        bool const lhs_is_null{_lhs.is_null(lhs_element_index)};
        bool const rhs_is_null{_rhs.is_null(rhs_element_index)};

        if (lhs_is_null or rhs_is_null) {  // at least one is null
          return cuda::std::make_pair(null_compare(lhs_is_null, rhs_is_null, _null_precedence),
                                      _depth);
        }
      }

      return cuda::std::make_pair(relational_compare(_lhs.element<Element>(lhs_element_index),
                                                     _rhs.element<Element>(rhs_element_index)),
                                  std::numeric_limits<int>::max());
    }

    template <typename Element,
              CUDF_ENABLE_IF(not cudf::is_relationally_comparable<Element, Element>() and
                             not std::is_same_v<Element, cudf::struct_view>),
              typename... Args>
    __device__ cuda::std::pair<weak_ordering, int> operator()(Args...)
    {
      CUDF_UNREACHABLE("Attempted to compare elements of uncomparable types.");
    }

    template <typename Element, CUDF_ENABLE_IF(std::is_same_v<Element, cudf::struct_view>)>
    __device__ cuda::std::pair<weak_ordering, int> operator()(size_type const lhs_element_index,
                                                              size_type const rhs_element_index)
    {
      column_device_view lcol = _lhs;
      column_device_view rcol = _rhs;
      int depth               = _depth;
      while (lcol.type().id() == type_id::STRUCT) {
        bool const lhs_is_null{lcol.is_null(lhs_element_index)};
        bool const rhs_is_null{rcol.is_null(rhs_element_index)};

        if (lhs_is_null or rhs_is_null) {  // at least one is null
          weak_ordering state = null_compare(lhs_is_null, rhs_is_null, _null_precedence);
          return cuda::std::make_pair(state, depth);
        }

        if (lcol.num_child_columns() == 0) {
          return cuda::std::make_pair(weak_ordering::EQUIVALENT, depth);
        }

        // Non-empty structs have been modified to only have 1 child when using this.
        lcol = lcol.children()[0];
        rcol = rcol.children()[0];
        ++depth;
      }

      auto const comparator = element_comparator{_nulls, lcol, rcol, _null_precedence, depth};
      return cudf::type_dispatcher<dispatch_void_if_nested>(
        lcol.type(), comparator, lhs_element_index, rhs_element_index);
    }

   private:
    column_device_view const _lhs;
    column_device_view const _rhs;
    Nullate const _nulls;
    null_order const _null_precedence;
    int const _depth;
  };

 public:
  /**
   * @brief Checks whether the row at `lhs_index` in the `lhs` table compares
   * lexicographically less than the row at `rhs_index` in the `rhs` table.
   *
   * @param lhs_index The index of row in the `lhs` table to examine
   * @param rhs_index The index of the row in the `rhs` table to examine
   * @return `true` if row from the `lhs` table compares less than row in the `rhs` table
   */
  __device__ bool operator()(size_type const lhs_index, size_type const rhs_index) const noexcept
  {
    int last_null_depth = std::numeric_limits<int>::max();
    for (size_type i = 0; i < _lhs.num_columns(); ++i) {
      int const depth = _depth.has_value() ? (*_depth)[i] : 0;
      if (depth > last_null_depth) { continue; }

      bool const ascending =
        _column_order.has_value() ? (*_column_order)[i] == order::ASCENDING : true;

      null_order const null_precedence =
        _null_precedence.has_value() ? (*_null_precedence)[i] : null_order::BEFORE;

      auto const comparator =
        element_comparator{_nulls, _lhs.column(i), _rhs.column(i), null_precedence, depth};

      weak_ordering state;
      cuda::std::tie(state, last_null_depth) =
        cudf::type_dispatcher(_lhs.column(i).type(), comparator, lhs_index, rhs_index);

      if (state == weak_ordering::EQUIVALENT) { continue; }

      return state == (ascending ? weak_ordering::LESS : weak_ordering::GREATER);
    }
    return false;
  }

 private:
  table_device_view const _lhs;
  table_device_view const _rhs;
  Nullate const _nulls{};
  std::optional<device_span<int const>> const _depth;
  std::optional<device_span<order const>> const _column_order;
  std::optional<device_span<null_order const>> const _null_precedence;
};  // class device_row_comparator

struct preprocessed_table {
  using table_device_view_owner =
    std::invoke_result_t<decltype(table_device_view::create), table_view, rmm::cuda_stream_view>;

  /**
   * @brief Preprocess table for use with lexicographical comparison
   *
   * Sets up the table for use with lexicographical comparison. The resulting preprocessed table can
   * be passed to the constructor of `lex::self_comparator` to avoid preprocessing again.
   *
   * @param table The table to preprocess
   * @param column_order Optional, host array the same length as a row that indicates the desired
   * ascending/descending order of each column in a row. If empty, it is assumed all columns are
   * sorted in ascending order.
   * @param null_precedence Optional, device array the same length as a row and indicates how null
   * values compare to all other for every column. If it is nullptr, then null precedence would be
   * `null_order::BEFORE` for all columns.
   * @param stream The stream to launch kernels and h->d copies on while preprocessing.
   */
  static std::shared_ptr<preprocessed_table> create(table_view const& table,
                                                    host_span<order const> column_order,
                                                    host_span<null_order const> null_precedence,
                                                    rmm::cuda_stream_view stream);

 private:
  friend class self_comparator;

  preprocessed_table(table_device_view_owner&& table,
                     rmm::device_uvector<order>&& column_order,
                     rmm::device_uvector<null_order>&& null_precedence,
                     rmm::device_uvector<size_type>&& depths)
    : _t(std::move(table)),
      _column_order(std::move(column_order)),
      _null_precedence(std::move(null_precedence)),
      _depths(std::move(depths)){};

  /**
   * @brief Implicit conversion operator to a `table_device_view` of the preprocessed table.
   *
   * @return table_device_view
   */
  operator table_device_view() { return *_t; }

  /**
   * @brief Get a device array containing the desired order of each column in the preprocessed table
   *
   * @return std::optional<device_span<order const>> Device array containing respective column
   * orders. If no explicit column orders were specified during the creation of this object then
   * this will be `nullopt`.
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
   * @return std::optional<device_span<null_order const>> Device array containing respective column
   * null precedence. If no explicit column null precedences were specified during the creation of
   * this object then this will be `nullopt`.
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
   * @return std::optional<device_span<int const>> Device array containing respective column depths.
   * If there are no nested columns in the table then this will be `nullopt`.
   */
  [[nodiscard]] std::optional<device_span<int const>> depths() const
  {
    return _depths.size() ? std::optional<device_span<int const>>(_depths) : std::nullopt;
  }

 private:
  table_device_view_owner _t;
  rmm::device_uvector<order> _column_order;
  rmm::device_uvector<null_order> _null_precedence;
  rmm::device_uvector<size_type> _depths;
};

/**
 * @brief An owning object that can be used to lexicographically compare two rows of the same table
 *
 * This class can take a table_view and preprocess certain columns to allow for lexicographical
 * comparison. The preprocessed table and temporary data required for the comparison are created and
 * owned by this class.
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
   * @brief Construct an owning object for performing a lexicographic comparison between two rows of
   * the same table.
   *
   * @param table The table to compare
   * @param column_order Optional, host array the same length as a row that indicates the desired
   * ascending/descending order of each column in a row. If empty, it is assumed all columns are
   * sorted in ascending order.
   * @param null_precedence Optional, device array the same length as a row and indicates how null
   * values compare to all other for every column. If empty, then null precedence would be
   * `null_order::BEFORE` for all columns.
   * @param stream The stream to construct this object on. Not the stream that will be used for
   * comparisons using this object.
   */
  self_comparator(table_view const& t,
                  host_span<order const> column_order         = {},
                  host_span<null_order const> null_precedence = {},
                  rmm::cuda_stream_view stream                = rmm::cuda_stream_default)
    : d_t{preprocessed_table::create(t, column_order, null_precedence, stream)}
  {
  }

  /**
   * @brief Construct an owning object for performing a lexicographic comparison between two rows of
   * the same preprocessed table.
   *
   * This constructor allows independently constructing a `preprocessed_table` and sharing it among
   * multiple comparators.
   *
   * @param t A table preprocessed for lexicographic comparison
   */
  self_comparator(std::shared_ptr<preprocessed_table> t) : d_t{std::move(t)} {}

  /**
   * @brief Return the binary operator for comparing rows in the table.
   *
   * Returns a binary callable, `F`, with signature `bool F(size_t, size_t)`.
   *
   * `F(i,j)` returns true if and only if row `i` compares lexicographically less than row `j`.
   *
   * @tparam Nullate Optional, A cudf::nullate type describing how to check for nulls.
   */
  template <typename Nullate>
  device_row_comparator<Nullate> device_comparator(Nullate nullate = {}) const
  {
    return device_row_comparator(
      nullate, *d_t, *d_t, d_t->depths(), d_t->column_order(), d_t->null_precedence());
  }

 private:
  std::shared_ptr<preprocessed_table> d_t;
};

}  // namespace lexicographic

namespace equality {

template <typename Nullate>
class device_row_comparator {
  friend class self_comparator;

 public:
  /**
   * @brief Checks whether the row at `lhs_index` in the `lhs` table is equal to the row at
   * `rhs_index` in the `rhs` table.
   *
   * @param lhs_index The index of row in the `lhs` table to examine
   * @param rhs_index The index of the row in the `rhs` table to examine
   * @return `true` if row from the `lhs` table is equal to the row in the `rhs` table
   */
  __device__ bool operator()(size_type const lhs_index, size_type const rhs_index) const noexcept
  {
    auto equal_elements = [=](column_device_view l, column_device_view r) {
      return cudf::type_dispatcher(
        l.type(), element_comparator{nulls, l, r, nulls_are_equal}, lhs_index, rhs_index);
    };

    return thrust::equal(thrust::seq, lhs.begin(), lhs.end(), rhs.begin(), equal_elements);
  }

 private:
  /**
   * @brief Construct a function object for performing equality comparison between the rows of two
   * tables.
   *
   * @param has_nulls Indicates if either input table contains columns with nulls.
   * @param lhs The first table
   * @param rhs The second table (may be the same table as `lhs`)
   * @param nulls_are_equal Indicates if two null elements are treated as equivalent
   */
  device_row_comparator(Nullate has_nulls,
                        table_device_view lhs,
                        table_device_view rhs,
                        null_equality nulls_are_equal = null_equality::EQUAL) noexcept
    : lhs{lhs}, rhs{rhs}, nulls{has_nulls}, nulls_are_equal{nulls_are_equal}
  {
  }

  /**
   * @brief Performs an equality comparison between two elements in two columns.
   *
   * @tparam Nullate A cudf::nullate type describing how to check for nulls.
   */
  class element_comparator {
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
    __device__ element_comparator(Nullate has_nulls,
                                  column_device_view lhs,
                                  column_device_view rhs,
                                  null_equality nulls_are_equal = null_equality::EQUAL) noexcept
      : lhs{lhs}, rhs{rhs}, nulls{has_nulls}, nulls_are_equal{nulls_are_equal}
    {
    }

    /**
     * @brief Compares the specified elements for equality.
     *
     * @param lhs_element_index The index of the first element
     * @param rhs_element_index The index of the second element
     * @return True if lhs and rhs are equal or if both lhs and rhs are null and nulls are
     * configured to be considered equal (`nulls_are_equal` == `null_equality::EQUAL`)
     */
    template <typename Element, CUDF_ENABLE_IF(cudf::is_equality_comparable<Element, Element>())>
    __device__ bool operator()(size_type const lhs_element_index,
                               size_type const rhs_element_index) const noexcept
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
              CUDF_ENABLE_IF(not cudf::is_equality_comparable<Element, Element>() and
                             not cudf::is_nested<Element>()),
              typename... Args>
    __device__ bool operator()(Args...)
    {
      CUDF_UNREACHABLE("Attempted to compare elements of uncomparable types.");
    }

    template <typename Element, CUDF_ENABLE_IF(cudf::is_nested<Element>())>
    __device__ bool operator()(size_type const lhs_element_index,
                               size_type const rhs_element_index) const noexcept
    {
      column_device_view lcol = lhs.slice(lhs_element_index, 1);
      column_device_view rcol = rhs.slice(rhs_element_index, 1);
      while (is_nested(lcol.type())) {
        if (nulls) {
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
          lcol = detail::structs_column_device_view(lcol).sliced_child(0);
          rcol = detail::structs_column_device_view(rcol).sliced_child(0);
        } else if (lcol.type().id() == type_id::LIST) {
          auto l_list_col = detail::lists_column_device_view(lcol);
          auto r_list_col = detail::lists_column_device_view(rcol);

          auto lsizes = make_list_size_iterator(l_list_col);
          auto rsizes = make_list_size_iterator(r_list_col);
          if (not thrust::equal(thrust::seq, lsizes, lsizes + lcol.size(), rsizes)) {
            return false;
          }

          lcol = l_list_col.sliced_child();
          rcol = r_list_col.sliced_child();
          if (lcol.size() != rcol.size()) { return false; }
        }
      }

      auto comp =
        column_comparator{element_comparator{nulls, lcol, rcol, nulls_are_equal}, lcol.size()};
      return type_dispatcher<dispatch_void_if_nested>(lcol.type(), comp);
    }

   private:
    /**
     * @brief Serially compare two columns for equality.
     *
     * When we want to get the equivalence of two columns by serially comparing all elements in a
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
    Nullate const nulls;
    null_equality const nulls_are_equal;
  };

  table_device_view const lhs;
  table_device_view const rhs;
  Nullate const nulls;
  null_equality const nulls_are_equal;
};

struct preprocessed_table {
  /**
   * @brief Preprocess table for use with row equality comparison or row hashing
   *
   * Sets up the table for use with row equality comparison or row hashing. The resulting
   * preprocessed table can be passed to the constructor of `equality::self_comparator` to
   * avoid preprocessing again.
   *
   * @param table The table to preprocess
   * @param stream The cuda stream to use while preprocessing.
   */
  static std::shared_ptr<preprocessed_table> create(table_view const& table,
                                                    rmm::cuda_stream_view stream);

 private:
  friend class self_comparator;

  using table_device_view_owner =
    std::invoke_result_t<decltype(table_device_view::create), table_view, rmm::cuda_stream_view>;

  preprocessed_table(table_device_view_owner&& table,
                     std::vector<rmm::device_buffer>&& null_buffers)
    : _t(std::move(table)), _null_buffers(std::move(null_buffers))
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
};

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
   * This constructor allows independently constructing a `preprocessed_table` and sharing it among
   * multiple comparators.
   *
   * @param t A table preprocessed for equality comparison
   */
  self_comparator(std::shared_ptr<preprocessed_table> t) : d_t{std::move(t)} {}

  /**
   * @brief Get the comparison operator to use on the device
   *
   * Returns a binary callable, `F`, with signature `bool F(size_t, size_t)`.
   *
   * `F(i,j)` returns true if and only if row `i` compares equal to row `j`.
   *
   * @tparam Nullate Optional, A cudf::nullate type describing how to check for nulls.
   */
  template <typename Nullate>
  device_row_comparator<Nullate> device_comparator(Nullate nullate = {}) const
  {
    return device_row_comparator(nullate, *d_t, *d_t);
  }

 private:
  std::shared_ptr<preprocessed_table> d_t;
};

}  // namespace equality

}  // namespace row
}  // namespace experimental
}  // namespace cudf
