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
#include <type_traits>
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

/**---------------------------------------------------------------------------*
 * @brief Determines accumulator type based on input type and operation.
 *
 * @tparam InputType The type of the input to the aggregation operation
 * @tparam op The aggregation operation performed
 * @tparam dummy Dummy for SFINAE
 *---------------------------------------------------------------------------**/
template <typename InputType, groupby::distributive_operators op,
          typename dummy = void>
struct result_type {
  using type = void;
};

// Computing MIN of T, use T accumulator
template <typename T>
struct result_type<T, groupby::distributive_operators::MIN> {
  using type = T;
};

// Computing MAX of T, use T accumulator
template <typename T>
struct result_type<T, groupby::distributive_operators::MAX> {
  using type = T;
};

// Counting T, always use int64_t accumulator
template <typename T>
struct result_type<T, groupby::distributive_operators::COUNT> {
  using type = int64_t;
};

// Summing integers of any type, always used int64_t
template <typename T>
struct result_type<T, groupby::distributive_operators::SUM,
                   typename std::enable_if<std::is_integral<T>::value>::type> {
  using type = int64_t;
};

// Summing float/doubles, use same type
template <typename T>
struct result_type<
    T, groupby::distributive_operators::SUM,
    typename std::enable_if<std::is_floating_point<T>::value>::type> {
  using type = T;
};

struct type_mapper {
  template <typename InputT>
  gdf_dtype operator()(groupby::distributive_operators op) {
    switch (op) {
      case groupby::distributive_operators::MIN:
        return gdf_dtype_of<typename result_type<
            InputT, groupby::distributive_operators::MIN>::type>();
      case groupby::distributive_operators::MAX:
        return gdf_dtype_of<typename result_type<
            InputT, groupby::distributive_operators::MAX>::type>();
      case groupby::distributive_operators::COUNT:
        return gdf_dtype_of<typename result_type<
            InputT, groupby::distributive_operators::COUNT>::type>();
      case groupby::distributive_operators::SUM:
        return gdf_dtype_of<typename result_type<
            InputT, groupby::distributive_operators::SUM>::type>();
      default:
        return GDF_invalid;
    }
  }
};

/**---------------------------------------------------------------------------*
 * @brief Determines the output that should be used for a given input type and
 * operator.
 *
 * @param input_type The type of the input aggregation column
 * @param op The aggregation operation
 * @return gdf_dtype Type to use for output aggregation column
 *---------------------------------------------------------------------------**/
gdf_dtype output_dtype(gdf_dtype input_type,
                       cudf::groupby::distributive_operators op) {
  return cudf::type_dispatcher(input_type, type_mapper{}, op);
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
        gdf_dtype t = output_dtype(input_col->dtype, op);
        CUDF_EXPECTS(
            t != GDF_invalid,
            "Invalid combination of input type and aggregation operation.");
        return t;
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
