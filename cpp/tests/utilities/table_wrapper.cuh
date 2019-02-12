/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#ifndef TABLE_WRAPPER_H
#define TABLE_WRAPPER_H

#include <utility>
#include "column_wrapper.cuh"
#include "cudf.h"
#include "rmm/rmm.h"
#include "tuple_vectors.h"
#include "utilities/bit_util.cuh"
#include "utilities/type_dispatcher.hpp"

namespace cudf {
namespace test {
namespace detail {
template <typename Tuple, typename F, std::size_t... Indices>
void for_each_impl(Tuple&& tuple, F&& f, std::index_sequence<Indices...>) {
  using swallow = int[];
  (void)swallow{
      1, (f(std::get<Indices>(std::forward<Tuple>(tuple))), void(), int{})...};
}

/**---------------------------------------------------------------------------*
 * @brief A `for_each` over the elements of a tuple.
 *
 * For every element in a tuple, invokes a unary callable and passes the tuple
 *element into the callable.
 *
 * @tparam Tuple The type of the tuple
 * @tparam F The type of the callable
 * @param tuple The tuple to iterate over
 * @param f The unary callable
 *---------------------------------------------------------------------------**/
template <typename Tuple, typename F>
void for_each(Tuple&& tuple, F&& f) {
  constexpr std::size_t N =
      std::tuple_size<std::remove_reference_t<Tuple>>::value;
  for_each_impl(std::forward<Tuple>(tuple), std::forward<F>(f),
                std::make_index_sequence<N>{});
}
}  // namespace detail

template <typename>
struct table_wrapper;

/**---------------------------------------------------------------------------*
 * @brief A wrapper for constructing a cudf::table with columns of arbitrary
 * types.
 *
 * Accepts a std::tuple of types as a template parameter. The type of element `i`
 * in the tuple determines the type of column `i` in the cudf::table.
 * 
 * For example,
 * 
 * `table_wrapper< std::tuple< int, float, double > >` will construct a cudf::table
 * with 3 columns of types GDF_INT32, GDF_FLOAT32, and GDF_FLOAT64, respectively.
 *
 * @tparam Ts The pack of types to use to create the set of columns
 *---------------------------------------------------------------------------**/
template <typename... Ts>
struct table_wrapper<std::tuple<Ts...>> {
  /**---------------------------------------------------------------------------*
   * @brief Construct a new table wrapper object
   *
   * Constructor that zero-intiailizes each column and it's bitmasks.
   *
   * @param[in] num_rows The number of rows in each column
   *---------------------------------------------------------------------------**/
  table_wrapper(gdf_size_type num_rows)
      : column_wrappers{std::unique_ptr<column_wrapper<Ts>>(
            new column_wrapper<Ts>(num_rows))...} {
    initialize_structures();
  }

  /**---------------------------------------------------------------------------*
   * @brief Construct a new table wrapper object
   *
   * Uses a unary callable to initialize each row's value and validity bitmask.
   *
   * Each column's value at row `i` will be equal to `v(i)` and bit will be
   *equal to `b(i)`.
   *
   * @tparam ValueIntializerType The type of the callable to initialize column
   *values
   * @tparam BitInitializeType  The type of the callable to initialize column
   *bitmasks
   * @param num_rows The number of rows in each column
   * @param v The value intializer callable
   * @param b The bitmask initializer callable
   *---------------------------------------------------------------------------**/
  template <typename ValueIntializerType, typename BitInitializeType>
  table_wrapper(gdf_size_type num_rows, ValueIntializerType v,
                BitInitializeType b)
      : column_wrappers{std::unique_ptr<column_wrapper<Ts>>(
            new column_wrapper<Ts>(num_rows, v, b))...} {
    initialize_structures();
  }

  /**---------------------------------------------------------------------------*
   * @brief Return pointer to array of underlying `gdf_column`s
   *
   *---------------------------------------------------------------------------**/
  gdf_column** get_columns() { return gdf_columns.data(); }
  gdf_column const* const* get_columns() const { return gdf_columns.data(); }

  /**---------------------------------------------------------------------------*
   * @brief Returns pointer to underlying cudf::table
   *
   *---------------------------------------------------------------------------**/
  cudf::table* get() { return the_table.get(); }
  cudf::table const* get() const { return the_table.get(); }

 private:
  /**---------------------------------------------------------------------------*
   * @brief Initializes the underlying gdf_column's and cudf::table
   * with the corresponding data from the tuple of column_wrappers.
   *
   *---------------------------------------------------------------------------**/
  void initialize_structures() {
    // Iterate the tuple of column_wrappers and store each underlying
    // `gdf_column` in a vector
    detail::for_each(column_wrappers, [this](auto const& col) {
      gdf_columns.push_back(col->get());
    });
    the_table.reset(new cudf::table(gdf_columns.data(), gdf_columns.size()));
  }

  gdf_size_type const num_columns{
      sizeof...(Ts)};  ///< number of columns in the table
  std::tuple<std::unique_ptr<column_wrapper<Ts>>...>
      column_wrappers;  ///< Collection of column_wrapper s of different types
  std::vector<gdf_column*>
      gdf_columns;  ///< Pointers to each column_wrapper's underlying gdf_column
  std::unique_ptr<cudf::table> the_table;  ///< The wrapped cudf::table
};
}  // namespace test
}  // namespace cudf
#endif
