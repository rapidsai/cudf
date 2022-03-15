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
#include <cudf/detail/hashing.hpp>
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

template <cudf::type_id t>
struct non_nested_id_to_type {
  using type = std::conditional_t<cudf::is_nested(data_type(t)), void, id_to_type<t>>;
};

namespace equality_hashing {

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
   * @return True if lhs and rhs are equal or if both lhs and rhs are null and nulls are configured
   * to be considered equal (`nulls_are_equal` == `null_equality::EQUAL`)
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
                             not cudf::is_nested<Element>()>* = nullptr>
  __device__ bool operator()(size_type lhs_element_index, size_type rhs_element_index)
  {
    cudf_assert(false && "Attempted to compare elements of uncomparable types.");
    return false;
  }

  template <typename Element,
            std::enable_if_t<not cudf::is_equality_comparable<Element, Element>() and
                             cudf::is_nested<Element>()>* = nullptr>
  __device__ bool operator()(size_type lhs_element_index, size_type rhs_element_index)
  {
    column_device_view lcol = lhs;
    column_device_view rcol = rhs;
    int l_start_off         = lhs_element_index;
    int r_start_off         = rhs_element_index;
    int l_end_off           = lhs_element_index + 1;
    int r_end_off           = rhs_element_index + 1;
    auto l_size             = 1;
    auto r_size             = 1;
    while (is_nested(lcol.type())) {
      if (nulls) {
        for (int i = 0; i < l_size; ++i) {
          bool const lhs_is_null{lcol.is_null(l_start_off + i)};
          bool const rhs_is_null{rcol.is_null(r_start_off + i)};

          if (lhs_is_null and rhs_is_null) {
            if (nulls_are_equal == null_equality::UNEQUAL) { return false; }
          } else if (lhs_is_null != rhs_is_null) {
            return false;
          }
        }
      }
      if (lcol.type().id() == type_id::STRUCT) {
        lcol = lcol.child(0);
        rcol = rcol.child(0);
      } else if (lcol.type().id() == type_id::LIST) {
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
        l_size      = l_end_off - l_start_off;
        r_size      = r_end_off - r_start_off;
        if (l_size != r_size) { return false; }
      }
    }

    for (int i = 0; i < l_size; ++i) {
      bool equal = type_dispatcher<non_nested_id_to_type>(
        lcol.type(),
        element_equality_comparator{nulls, lcol, rcol, nulls_are_equal},
        l_start_off + i,
        r_start_off + i);
      if (not equal) { return false; }
    }
    return true;
  }

 private:
  column_device_view lhs;
  column_device_view rhs;
  Nullate nulls;
  null_equality nulls_are_equal;
};

template <typename Nullate>
class row_equality_comparator {
  friend class self_eq_comparator;

 public:
  /**
   * @brief Construct a function object for performing equality comparison between the rows of two
   * tables.
   *
   * @param has_nulls Indicates if either input table contains columns with nulls.
   * @param lhs The first table
   * @param rhs The second table (may be the same table as `lhs`)
   * @param nulls_are_equal Indicates if two null elements are treated as equivalent
   */
  row_equality_comparator(Nullate has_nulls,
                          table_device_view lhs,
                          table_device_view rhs,
                          null_equality nulls_are_equal = null_equality::EQUAL)
    : lhs{lhs}, rhs{rhs}, nulls{has_nulls}, nulls_are_equal{nulls_are_equal}
  {
    CUDF_EXPECTS(lhs.num_columns() == rhs.num_columns(), "Mismatched number of columns.");
  }

 public:
  /**
   * @brief Checks whether the row at `lhs_index` in the `lhs` table is equal to the row at
   * `rhs_index` in the `rhs` table.
   *
   * @param lhs_index The index of row in the `lhs` table to examine
   * @param rhs_index The index of the row in the `rhs` table to examine
   * @return `true` if row from the `lhs` table is equal to the row in the `rhs` table
   */
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
      hash = hash_combine(
        hash,
        type_dispatcher<non_nested_id_to_type>(curr_col.type(),
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
    // Hash the first column w/ the seed
    auto const initial_hash = cudf::detail::hash_combine(
      hash_value_type{0},
      type_dispatcher<dispatch_storage_type>(_table.column(0).type(),
                                             // TODO: revert back to using seed
                                             element_hasher<hash_function, Nullate>{_has_nulls},
                                             _table.column(0),
                                             row_index,
                                             [](hash_value_type lhs, hash_value_type rhs) {
                                               return cudf::detail::hash_combine(lhs, rhs);
                                             }));

    // Hashes an element in a column
    auto hasher = [=](size_type column_index) {
      return cudf::type_dispatcher<dispatch_storage_type>(
        _table.column(column_index).type(),
        element_hasher<hash_function, Nullate>{_has_nulls},
        _table.column(column_index),
        row_index,
        [](hash_value_type lhs, hash_value_type rhs) {
          return cudf::detail::hash_combine(lhs, rhs);
        });
    };

    // Hash each element and combine all the hash values together
    return thrust::transform_reduce(
      thrust::seq,
      // note that this starts at 1 and not 0 now since we already hashed the first column
      thrust::make_counting_iterator(1),
      thrust::make_counting_iterator(_table.num_columns()),
      hasher,
      initial_hash,
      [](hash_value_type lhs, hash_value_type rhs) {
        return cudf::detail::hash_combine(lhs, rhs);
      });
  }

 private:
  table_device_view _table;
  Nullate _has_nulls;
  uint32_t _seed{DEFAULT_HASH_SEED};
};

struct preprocessed_table {
  /**
   * @brief Preprocess table for use with row equality comparison or row hashing
   *
   * Sets up the table for use with row equality comparison or row hashing. The resulting
   * preprocessed table can be passed to the constructor of `equality_hashing::self_comparator` to
   * avoid preprocessing again.
   *
   * @param table The table to preprocess
   * @param stream The cuda stream to use while preprocessing.
   */
  preprocessed_table(table_view const& table, rmm::cuda_stream_view stream);

  /**
   * @brief Implicit conversion operator to a `table_device_view` of the preprocessed table.
   *
   * @return table_device_view
   */
  operator table_device_view() { return **d_t; }

  /**
   * @brief Whether the table has any nullable column
   *
   */
  [[nodiscard]] bool has_nulls() const { return _has_nulls; }

 private:
  using table_device_view_owner =
    std::invoke_result_t<decltype(table_device_view::create), table_view, rmm::cuda_stream_view>;

  std::unique_ptr<table_device_view_owner> d_t;
  bool _has_nulls;
};

class self_eq_comparator {
 public:
  /**
   * @brief Construct an owning object for performing equality comparisons between two rows of the
   * same table.
   *
   * @param t The table to compare
   * @param stream The stream to construct this object on. Not the stream that will be used for
   * comparisons using this object.
   */
  self_eq_comparator(table_view const& t, rmm::cuda_stream_view stream)
    : d_t(std::make_shared<preprocessed_table>(t, stream))
  {
  }

  /**
   * @brief Construct an owning object for performing equality comparisons between two rows of the
   * same table.
   *
   * @param t A table preprocessed for equality comparison
   */
  self_eq_comparator(std::shared_ptr<preprocessed_table> t) : d_t{std::move(t)} {}

  /**
   * @brief Get the comparison operator to use on the device
   *
   * @tparam Nullate Optional, A cudf::nullate type describing how to check for nulls.
   */
  template <typename Nullate = nullate::DYNAMIC>
  row_equality_comparator<Nullate> device_comparator()
  {
    if constexpr (std::is_same_v<Nullate, nullate::DYNAMIC>) {
      return row_equality_comparator(Nullate{d_t->has_nulls()}, *d_t, *d_t);
    } else {
      return row_equality_comparator(Nullate{}, *d_t, *d_t);
    }
  }

 private:
  using table_device_view_owner =
    std::invoke_result_t<decltype(table_device_view::create), table_view, rmm::cuda_stream_view>;

  std::shared_ptr<preprocessed_table> d_t;
};

}  // namespace equality_hashing
}  // namespace experimental
}  // namespace cudf
