/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf.h>
#include <groupby.hpp>
#include <hash/concurrent_unordered_map.cuh>
#include <types.hpp>
#include <utilities/type_dispatcher.hpp>
#include "aggregation_operations.hpp"
#include "new_hash_groupby.hpp"

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/fill.h>
#include <vector>

namespace cudf {
namespace detail {

namespace {

struct identity_initializer {
  template <typename T>
  T get_identity(groupby::distributive_operators op) {
    switch (op) {
      case groupby::distributive_operators::SUM:
        return sum_op<T>::IDENTITY;
      case groupby::distributive_operators::MIN:
        return min_op<T>::IDENTITY;
      case groupby::distributive_operators::MAX:
        return max_op<T>::IDENTITY;
      case groupby::distributive_operators::COUNT:
        return sum_op<T>::IDENTITY;
      default:
        CUDF_FAIL("Invalid aggregation operation.");
    }
  }

  template <typename T>
  void operator()(gdf_column const& col, groupby::distributive_operators op,
                  cudaStream_t stream = 0) {
    T* typed_data = static_cast<T*>(col.data);
    thrust::fill(rmm::exec_policy(stream)->on(stream), typed_data,
                 typed_data + col.size, get_identity<T>(op));
  }
};

/**---------------------------------------------------------------------------*
 * @brief Initializes each column in a table with a corresponding identity value
 * of an aggregation operation.
 *
 * The `i`th column will be initialized with the identity value of the `i`th
 * aggregation operation.
 *
 * @param table The table of columns to initialize.
 * @param operators The aggregation operations whose identity values will be
 *used to initialize the columns.
 *---------------------------------------------------------------------------**/
void initialize_with_identity(
    cudf::table const& table,
    std::vector<cudf::groupby::distributive_operators> const& operators,
    cudaStream_t stream = 0) {
  // TODO: Initialize all the columns in a single kernel instead of invoking one
  // kernel per column
  for (gdf_size_type i = 0; i < table.num_columns(); ++i) {
    gdf_column const* col = table.get_column(i);
    cudf::type_dispatcher(col->dtype, identity_initializer{}, *col,
                          operators[i]);
  }
}
}  // namespace

std::tuple<cudf::table, cudf::table> hash_groupby(
    cudf::table const& keys, cudf::table const& values,
    std::vector<cudf::groupby::distributive_operators> const& operators,
    std::vector<gdf_dtype> const& output_dtypes, cudaStream_t stream) {
  // Create the output key and value tables
  // The exact output size is unknown a priori, therefore, use the input size as
  // an upper bound
  std::vector<gdf_dtype> key_dtypes(keys.num_columns());
  std::transform(keys.begin(), keys.end(), key_dtypes.begin(),
                 [](gdf_column const* col) { return col->dtype; });
  cudf::table output_keys{keys.num_rows(), key_dtypes, true, stream};
  cudf::table output_values{keys.num_rows(), output_dtypes, true, stream};
  initialize_with_identity(output_values, operators, stream);

  using map_type = concurrent_unordered_map<
      gdf_size_type, gdf_size_type, std::numeric_limits<gdf_size_type>::max(),
      default_hash<gdf_size_type>, equal_to<gdf_size_type>,
      legacy_allocator<thrust::pair<gdf_size_type, gdf_size_type> > >;

  std::unique_ptr<map_type> map =
      std::make_unique<map_type>(compute_hash_table_size(keys.num_rows()), 0);

  CHECK_STREAM(stream);

  return std::make_tuple(output_keys, output_values);
}

}  // namespace detail
}  // namespace cudf
