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

#ifndef _GROUPBY_COMMON_H
#define _GROUPBY_COMMON_H


#include <cudf/cudf.h>
#include <bitmask/legacy/bit_mask.cuh>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/legacy/bitmask.hpp>
#include <cudf/legacy/table.hpp>
#include <hash/concurrent_unordered_map.cuh>
#include <cudf/utilities/legacy/nvcategory_util.hpp>
#include <table/legacy/device_table.cuh>
#include <table/legacy/device_table_row_operators.cuh>
#include <utilities/column_utils.hpp>
#include <utilities/cuda_utils.hpp>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <rmm/thrust_rmm_allocator.h>
#include <thrust/fill.h>
#include <type_traits>
#include <vector>
#include "type_info.hpp"


namespace cudf {
namespace groupby {

/**---------------------------------------------------------------------------*
 * @brief Verifies the requested aggregation is valid for the type of the value
 * column.
 *
 * Given a table of values and a set of operators, verifies that `ops[i]` is
 * valid to perform on `column[i]`.
 *
 * @throw cudf::logic_error if an invalid combination of value type and operator
 * is requested.
 *
 * @param values The table of columns
 * @param ops The aggregation operators
 *---------------------------------------------------------------------------**/
static void verify_operators(table const& values, std::vector<operators> const& ops) {
  CUDF_EXPECTS(static_cast<gdf_size_type>(ops.size()) == values.num_columns(),
               "Size mismatch between ops and value columns");
  for (gdf_size_type i = 0; i < values.num_columns(); ++i) {
    // TODO Add more checks here, i.e., can't compute sum of non-arithemtic
    // types
    if ((ops[i] == SUM) and
        (values.get_column(i)->dtype == GDF_STRING_CATEGORY)) {
      CUDF_FAIL(
          "Cannot compute SUM aggregation of GDF_STRING_CATEGORY column.");
    }
  }
}

/**---------------------------------------------------------------------------*
 * @brief Determines target gdf_dtypes to use for combinations of source
 * gdf_dtypes and aggregation operations.
 *
 * Given vectors of source gdf_dtypes and corresponding aggregation operations
 * to be performed on that type, returns a vector of gdf_dtypes to use to store
 * the result of the aggregation operations.
 *
 * @param source_dtypes The source types
 * @param op The aggregation operations
 * @return Target gdf_dtypes to use for the target aggregation columns
 *---------------------------------------------------------------------------**/
inline std::vector<gdf_dtype> target_dtypes(
    std::vector<gdf_dtype> const& source_dtypes,
    std::vector<operators> const& ops) {
  std::vector<gdf_dtype> output_dtypes(source_dtypes.size());

  std::transform(
      source_dtypes.begin(), source_dtypes.end(), ops.begin(),
      output_dtypes.begin(), [](gdf_dtype source_dtype, operators op) {
        gdf_dtype t =
            cudf::type_dispatcher(source_dtype, target_type_mapper{}, op);
        CUDF_EXPECTS(
            t != GDF_invalid,
            "Invalid combination of input type and aggregation operation.");
        return t;
      });

  return output_dtypes;
}

/**---------------------------------------------------------------------------*
 * @brief Dispatched functor to initialize a column with the identity of an
 *aggregation operation.
 *---------------------------------------------------------------------------**/
struct identity_initializer {
  template <typename T>
  T get_identity(operators op) {
    switch (op) {
      case SUM:
        return corresponding_functor_t<SUM>::identity<T>();
      case MIN:
        return corresponding_functor_t<MIN>::identity<T>();
      case MAX:
        return corresponding_functor_t<MAX>::identity<T>();
      case COUNT:
        return corresponding_functor_t<COUNT>::identity<T>();
      default:
        CUDF_FAIL("Invalid aggregation operation.");
    }
  }

  template <typename T>
  void operator()(gdf_column const& col, operators op,
                  cudaStream_t stream = 0) {
    T* typed_data = static_cast<T*>(col.data);
    thrust::fill(rmm::exec_policy(stream)->on(stream), typed_data,
                 typed_data + col.size, get_identity<T>(op));

    // For COUNT operator, initialize column's bitmask to be all valid
    if ((nullptr != col.valid) and (COUNT == op)) {
      CUDA_TRY(cudaMemsetAsync(
          col.valid, 0xff,
          sizeof(gdf_valid_type) * gdf_valid_allocation_size(col.size),
          stream));
    }
  }
};

/**---------------------------------------------------------------------------*
 * @brief Initializes each column in a table with a corresponding identity value
 * of an aggregation operation.
 *
 * The `i`th column will be initialized with the identity value of the `i`th
 * aggregation operation.
 *
 * @note The validity bitmask (if not `nullptr`) for the column corresponding to
 * a COUNT operator will be initialized to all valid.
 *
 * @param table The table of columns to initialize.
 * @param operators The aggregation operations whose identity values will be
 *used to initialize the columns.
 *---------------------------------------------------------------------------**/
static void initialize_with_identity(cudf::table const& table,
                              std::vector<operators> const& ops,
                              cudaStream_t stream = 0) {
  // TODO: Initialize all the columns in a single kernel instead of invoking one
  // kernel per column
  for (gdf_size_type i = 0; i < table.num_columns(); ++i) {
    gdf_column const* col = table.get_column(i);
    cudf::type_dispatcher(col->dtype, identity_initializer{}, *col, ops[i]);
  }
}

/**---------------------------------------------------------------------------*
 * @brief Compacts any GDF_STRING_CATEGORY columns in the output keys or values.
 *
 * After the groupby operation, any GDF_STRING_CATEGORY column in either the
 * keys or values may reference only a subset of the strings in the original
 * input category. This function will create a new associated NVCategory object
 * for the output GDF_STRING_CATEGORY columns whose dictionary contains only the
 * strings referenced in the output result.
 *
 * @param[in] input_keys The set of input key columns
 * @param[in/out] output_keys The set of output key columns
 * @param[in] input_values The set of input value columns
 * @param[in/out] output_values The set of output value columns
 *---------------------------------------------------------------------------**/
static void update_nvcategories(table const& input_keys, table& output_keys,
                         table const& input_values, table& output_values) {
  gdf_error update_err = nvcategory_gather_table(input_keys, output_keys);
  CUDF_EXPECTS(update_err == GDF_SUCCESS, "nvcategory_gather_table error for keys");

  // Filter out columns from (input/output)_values where the output datatype is
  // not GDF_STRING_CATEGORY. This is possible in aggregations like `count`.
  std::vector<gdf_column *> string_input_value_cols;
  std::vector<gdf_column *> string_output_value_cols;

  for (gdf_size_type i = 0; i < input_values.num_columns(); i++) {
    if (output_values.get_column(i)->dtype == GDF_STRING_CATEGORY) {
      string_input_value_cols.push_back(
        const_cast<gdf_column*>(input_values.get_column(i)));
      string_output_value_cols.push_back(
        const_cast<gdf_column*>(output_values.get_column(i)));
    }
  }
  
  cudf::table string_input_values(string_input_value_cols);
  cudf::table string_output_values(string_output_value_cols);
  
  update_err = nvcategory_gather_table(string_input_values, string_output_values);
  CUDF_EXPECTS(update_err == GDF_SUCCESS, "nvcategory_gather_table error for values");
}


} // namespace groupby 
} // namespace cudf 

#endif
