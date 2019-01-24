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

template <typename>
struct table_wrapper;

template <typename... Ts>
struct table_wrapper<std::tuple<Ts...>> {
  table_wrapper(gdf_size_type num_rows) {}

  gdf_column **get() { return gdf_columns.data(); }

 private:
  template <typename ValueGenerator, typename BitmaskGenerator>
  void initialize_columns(gdf_size_type num_rows, ValueGenerator v,
                          BitmaskGenerator b) {
    initialize_columns_impl(num_rows, v, b, std::index_sequence_for<Ts...>{});
  }

  template <typename ValueGenerator, typename BitmaskGenerator,
            std::size_t... Is>
  void initialize_columns_impl(gdf_size_type num_rows, ValueGenerator v,
                               BitmaskGenerator b, std::index_sequence<Is...>) {
    // This is some nasty hackery, but essentially we're just iterating
    // over the tuple of column wrappers, initializing each column_wrapper, and
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

  gdf_size_type const num_columns{sizeof...(Ts)};
  std::tuple<std::unique_ptr<column_wrapper<Ts>>...> column_wrappers;
  std::vector<gdf_column *> gdf_columns;
};
}  // namespace test
}  // namespace cudf
#endif
