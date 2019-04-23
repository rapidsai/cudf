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

constexpr inline bool is_an_integer(gdf_dtype element_type) {
  return (element_type == GDF_INT8) or (element_type == GDF_INT16) or
         (element_type == GDF_INT32) or (element_type == GDF_INT64);
}

constexpr inline bool is_floating_point(gdf_dtype element_type) {
  return (element_type == GDF_FLOAT32) or (element_type == GDF_FLOAT64);
}

/**---------------------------------------------------------------------------*
 * @brief Determines the output that should be used for a given input type and
 * operator.
 *
 * @param input_type The type of the input aggregation column
 * @param op The aggregation operation
 * @return gdf_dtype Type to use for output aggregation column
 *---------------------------------------------------------------------------**/
constexpr gdf_dtype output_type(gdf_dtype input_type,
                                cudf::groupby::distributive_operators op) {
  switch (op) {
    // Use same type for min/max
    case groupby::distributive_operators::MIN:
      return input_type;
    case groupby::distributive_operators::MAX:
      return input_type;

    // Always use int64_t for count
    case groupby::distributive_operators::COUNT:
      return GDF_INT64;

    case groupby::distributive_operators::SUM: {
      // Always use the largest int when computing the sum of integers
      if (is_an_integer(input_type)) {
        return GDF_INT64;
      }

      // Use same type as input when computing sum of floating point values
      if (is_floating_point(input_type)) {
        return input_type;
      }
      return GDF_invalid;
    }
    default:
      return GDF_invalid;
  }
}
}  // namespace

std::tuple<cudf::table, cudf::table> hash_groupby(
    cudf::table const& keys, cudf::table const& values,
    std::vector<cudf::groupby::distributive_operators> const& operators,
    cudaStream_t stream) {
  // The exact output size is unknown a priori, therefore, use the input size as
  // an upper bound
  gdf_size_type const output_size{keys.num_rows()};

  // Allocate output keys
  std::vector<gdf_dtype> key_dtypes(keys.num_columns());
  std::transform(keys.begin(), keys.end(), key_dtypes.begin(),
                 [](gdf_column const* col) { return col->dtype; });
  cudf::table output_keys{output_size, key_dtypes, true, stream};

  // Allocate/initialize output values
  std::vector<gdf_dtype> output_dtypes(values.num_columns());
  std::transform(
      values.begin(), values.end(), operators.begin(), output_dtypes.begin(),
      [](gdf_column const* input_col, groupby::distributive_operators op) {
        return output_type(input_col->dtype, op);
      });
  cudf::table output_values{output_size, output_dtypes, true, stream};
  initialize_with_identity(output_values, operators, stream);

  using map_type = concurrent_unordered_map<
      gdf_size_type, gdf_size_type, std::numeric_limits<gdf_size_type>::max(),
      default_hash<gdf_size_type>, equal_to<gdf_size_type>,
      legacy_allocator<thrust::pair<gdf_size_type, gdf_size_type> > >;

  std::unique_ptr<map_type> map =
      std::make_unique<map_type>(compute_hash_table_size(keys.num_rows()), 0);

  rmm::device_vector<groupby::distributive_operators> d_operators(operators);

  CHECK_STREAM(stream);

  return std::make_tuple(output_keys, output_values);
}

}  // namespace detail
}  // namespace cudf
