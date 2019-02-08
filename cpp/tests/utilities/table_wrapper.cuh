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
/**---------------------------------------------------------------------------*
 * @brief Invokes a callable with a specified index sequence.
 *
 * This is a helper function for `index_apply` that will invoke the callable `f`
 * with the expansion of the specified `index_sequence` as
 * `std::integral_constant`s.
 *
 * @tparam F The callable's type
 * @tparam Is The index_sequence
 * @param f The callable
 * @return constexpr auto Returns whatever is returned by the callable
 *---------------------------------------------------------------------------**/
template <class F, size_t... Is>
constexpr auto index_apply_impl(F f, std::index_sequence<Is...>) {
  return f(std::integral_constant<size_t, Is>{}...);
}

/**---------------------------------------------------------------------------*
 * @brief Invokes a callable with arguments consisting of an index sequence of a
 * specified sized.
 *
 * Given a callable `f`, and a size `N`, this function will invoke
 * `f(0, 1, ..., N-1)` where the index_sequence are integral_constant values.
 *
 * @tparam N The size of the index sequence
 * @tparam F The callable's type
 * @param f The callable
 *---------------------------------------------------------------------------**/
template <size_t N, class F>
constexpr auto index_apply(F f) {
  return index_apply_impl(f, std::make_index_sequence<N>{});
}

template <class Tuple, class F>
constexpr auto apply(Tuple const &t, F f) {
  return index_apply<std::tuple_size<Tuple>::value>(
      [&](auto... Is) { return f(std::get<Is>(t)...); });
}

}  // namespace detail

template <typename>
struct table_wrapper;

template <typename... Ts>
struct table_wrapper<std::tuple<Ts...>> {
  table_wrapper(gdf_size_type num_rows)
      : column_wrappers{std::unique_ptr<column_wrapper<Ts>>(
            new column_wrapper<Ts>(num_rows))...}

  {
    initialize_columns();
  }

  gdf_column **get() { return gdf_columns.data(); }

 private:

  void init_columns(std::unique_ptr<column_wrapper<Ts>> &... cols) {
    (void)std::initializer_list<int>{
        ((void)gdf_columns.push_back(cols->get()), 0)...};
  }

  void initialize_columns() {
    return detail::index_apply<sizeof...(Ts)>([this](auto... Is) {
      return this->init_columns(std::get<Is>(column_wrappers)...);
    });
  }

  /*
    template <typename ValueGenerator, typename BitmaskGenerator,
              std::size_t... Is>
    void initialize_columns_impl(gdf_size_type num_rows, ValueGenerator v,
                                 BitmaskGenerator b, std::index_sequence<Is...>)
    {
      // This is some nasty hackery, but essentially we're just iterating
      // over the tuple of column wrappers, initializing each column_wrapper,
    and
      // pushing a pointer to the underlying `gdf_column` onto the `gdf_columns`
      // vector
      // For more details on the hackery, see:
      // https://codereview.stackexchange.com/a/67394
      using dummy = int[];
      (void)dummy{
          1, (initialize_column(std::get<Is>(column_wrappers), num_rows, v, b),
              void(), int{})...};
    }

    template <typename T, typename ValueGenerator, typename BitmaskGenerator>
    void initialize_column(std::unique_ptr<column_wrapper<T>> &column,
                           gdf_size_type num_rows, ValueGenerator v,
                           BitmaskGenerator b) {
      column.reset(new column_wrapper<T>(num_rows, v, b));
      gdf_columns.push_back(column->get());
    }
    */

  gdf_size_type const num_columns{
      sizeof...(Ts)};  ///< number of columns in the table
  std::tuple<std::unique_ptr<column_wrapper<Ts>>...>
      column_wrappers;  ///< Collection of column_wrapper s of different types
  std::vector<gdf_column *>
      gdf_columns;  ///< Pointers to each column_wrapper's underlying gdf_column
};
}  // namespace test
}  // namespace cudf
#endif
