/*
 * Copyright 2019 BlazingDB, Inc.
 *     Copyright 2019 Alexander Ocsa <alexander@blazingdb.com>
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

#include <algorithm>
#include <cassert>
#include <thrust/fill.h>
#include <tuple>

#include "cudf/copying.hpp"
#include "cudf/cudf.h"
#include "groupby/aggregation_operations.hpp"
#include "groupby/hash_groupby.cuh"
#include "groupby/sort/groupby_sort.h"
#include "string/nvcategory_util.hpp"
#include "table/device_table.cuh"
#include "cudf/types.hpp"
#include "utilities/error_utils.hpp"
#include "utilities/nvtx/nvtx_utils.h"

#include "groupby_sort.h"

#include "groupby_count_valid.h"
#include "groupby_count_wo_valid.h"
#include "groupby_valid.h"
#include "groupby_wo_valid.h"
#include <cudf/groupby.hpp>

namespace cudf {
namespace groupby {
namespace sort {

namespace {
/* --------------------------------------------------------------------------*/
/**
 * @brief Verifies that a set gdf_columns contain non-null data buffers, and are
 * all
 * of the same size.
 *
 *
 * TODO: remove when null support added.
 *
 * Also ensures that the columns do not contain any null values
 *
 * @param[in] first Pointer to first gdf_column in set
 * @param[in] last Pointer to one past the last column in set
 *
 * @returns GDF_DATASET_EMPTY if a column contains a null data buffer,
 * GDF_COLUMN_SIZE_MISMATCH if the columns are not of equal length,
 */
/* ----------------------------------------------------------------------------*/
gdf_error verify_columns(gdf_column *cols[], int num_cols) {
  GDF_REQUIRE((nullptr != cols[0]), GDF_DATASET_EMPTY);

  gdf_size_type const required_size{cols[0]->size};

  for (int i = 0; i < num_cols; ++i) {
    GDF_REQUIRE(nullptr != cols[i], GDF_DATASET_EMPTY);
    GDF_REQUIRE(nullptr != cols[i]->data, GDF_DATASET_EMPTY);
    GDF_REQUIRE(required_size == cols[i]->size, GDF_COLUMN_SIZE_MISMATCH);
  }
  return GDF_SUCCESS;
}

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis Calls the Sort Based group by compute API to compute the groupby
 * with
 * aggregation.
 *
 * @Param ncols, in_groupby_columns The input groupby table
 * @Param in_aggregation_column The input aggregation column
 * @Param out_groupby_columns The output groupby table
 * @Param out_aggregation_column The output aggregation column
 * @Param ctxt Flag to optionally sort the output
 * @tparam aggregation_type  The type of the aggregation column
 * @tparam op A binary functor that implements the aggregation operation
 *
 * @Returns On failure, returns appropriate error code. Otherwise, GDF_SUCCESS
 */
/* ----------------------------------------------------------------------------*/

gdf_error compute_sort_keys_groupby(gdf_size_type num_groupby_cols,
                               gdf_column *const *in_groupby_columns,
                               Options options,
                               rmm::device_vector<int32_t> &sorted_indices) {
  bool group_by_keys_contain_nulls = false;
  std::vector<gdf_column *> orderby_cols_vect(num_groupby_cols);
  for (int i = 0; i < num_groupby_cols; i++) {
    orderby_cols_vect[i] = (gdf_column *)in_groupby_columns[i];
    group_by_keys_contain_nulls =
        group_by_keys_contain_nulls || orderby_cols_vect[i]->null_count > 0;
  }
  int32_t nrows = in_groupby_columns[0]->size;

  sorted_indices.resize(nrows);
  gdf_column sorted_indices_col;
  gdf_error status = gdf_column_view(&sorted_indices_col,
                                     (void *)(sorted_indices.data().get()),
                                     nullptr, nrows, GDF_INT32);

  if (status != GDF_SUCCESS)
    return status;

  gdf_context ctxt;
  ctxt.flag_null_sort_behavior = GDF_NULL_AS_LARGEST;

  // run order by and get new sort indexes
  status =
      gdf_order_by(&orderby_cols_vect[0], // input columns
                   nullptr,
                   num_groupby_cols, // number of columns in the first parameter
                                     // (e.g. number of columsn to sort by)
                   &sorted_indices_col, // a gdf_column that is pre allocated
                                        // for storing sorted indices
                   &ctxt);
  return status;
}

template <template <typename> class, template <typename> class>
struct is_same_functor : std::false_type {};

template <template <typename> class T>
struct is_same_functor<T, T> : std::true_type {};

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  Helper function for gdf_groupby_sort. Deduces the type of the
 * aggregation
 * column and calls another function to perform the group by.
 *
 */
/* ----------------------------------------------------------------------------*/

/* --------------------------------------------------------------------------*/
/**
 * @Synopsis  This function provides the libgdf entry point for a Sort-based
 * group-by.
 * Performs a Group-By operation on an arbitrary number of columns with a single
 * aggregation column.
 * 
 * @Param[in] ncols The number of columns to group-by
 * @Param[in] in_groupby_columns[] The columns to group-by
 * @Param[in,out] in_aggregation_column The column to perform the aggregation on
 * @Param[in,out] out_groupby_columns[] A preallocated buffer to store the
 * resultant group-by columns
 * @Param[in,out] out_aggregation_column A preallocated buffer to store the
 * resultant aggregation column
 * @tparam[in] aggregation_operation A functor that defines the aggregation
 * operation
 *
 * @Returns gdf_error
 */
/* ----------------------------------------------------------------------------*/
gdf_error sort_keys_groupby(gdf_size_type ncols,
                                gdf_column *const *in_groupby_columns,
                                Options options,
                                rmm::device_vector<int32_t> &sorted_indices) {

  // Make sure the inputs are not null
  if (0 == ncols || nullptr == in_groupby_columns) {
    return GDF_DATASET_EMPTY;
  }

  // If there are no rows in the input, return successfully
  if (0 == in_groupby_columns[0]->size) {
    return GDF_SUCCESS;
  }

  return compute_sort_keys_groupby(ncols, in_groupby_columns, options, sorted_indices);
}

} // anonymous namespace

namespace detail {

gdf_error compute_sort_groupby(int num_key_columns,
                                 gdf_column *in_key_columns[],
                                 gdf_column *in_aggregation_column,
                                 gdf_agg_op op,
                                 gdf_column *out_key_columns[],
                                 gdf_column *out_aggregation_column,
                                 gdf_context *options,
                                 rmm::device_vector<int32_t> &sorted_indices) {
  gdf_error gdf_error_code{GDF_SUCCESS};

  bool group_by_keys_contain_nulls = false;
  for (int i = 0; i < num_key_columns; i++) {
    group_by_keys_contain_nulls =
        group_by_keys_contain_nulls || in_key_columns[i]->null_count > 0;
  }
  if (op == GDF_COUNT) {
    if (group_by_keys_contain_nulls) {
      gdf_error_code = gdf_group_by_count_with_valids(
          num_key_columns, in_key_columns, in_aggregation_column,
          out_key_columns, out_aggregation_column, op, options, sorted_indices);
    } else {
      gdf_error_code = gdf_group_by_count_wo_valids(
          num_key_columns, in_key_columns, in_aggregation_column,
          out_key_columns, out_aggregation_column, op, options, sorted_indices);
    }
  } else if (op == GDF_AVG) {
    gdf_error_code = gdf_group_by_sort_avg(
        num_key_columns, in_key_columns, in_aggregation_column, out_key_columns,
        out_aggregation_column);
  } else {
    if (group_by_keys_contain_nulls) {
      gdf_error_code = gdf_group_by_sort_with_valids(
          num_key_columns, in_key_columns, in_aggregation_column,
          out_key_columns, out_aggregation_column, op, options, sorted_indices);
    } else {
      gdf_error_code = gdf_group_by_sort_wo_valids(
          num_key_columns, in_key_columns, in_aggregation_column,
          out_key_columns, out_aggregation_column, op, options, sorted_indices);
    }
  }
  return gdf_error_code;
}


} // END: namespace detail

namespace {

void verify_operators(table const &values, std::vector<operators> const &ops) {
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


auto allocate_table_like(table const &t, cudaStream_t stream) {
  std::vector<gdf_column *> columns(t.num_columns());
  std::transform(columns.begin(), columns.end(), t.begin(), columns.begin(),
                 [stream](gdf_column *out_col, gdf_column const *in_col) {
                   out_col = new gdf_column;
                   *out_col = allocate_like(*in_col, stream);
                   return out_col;
                 });

  return table{columns.data(), static_cast<gdf_size_type>(columns.size())};
}
auto get_pointers(cudf::table const &input) {
   gdf_column** out_key_columns = new gdf_column*[input.num_columns()];
   for (gdf_size_type i = 0; i < input.num_columns(); i++) {
     out_key_columns[i] = (gdf_column*)input.get_column(i);
   }
   return out_key_columns;
}

}

std::pair<cudf::table, cudf::table> groupby(cudf::table const &keys,
                                             cudf::table const &values,
                                             std::vector<operators> const &ops,
                                             Options options) {

  CUDF_EXPECTS(keys.num_rows() == values.num_rows(),
               "Size mismatch between number of rows in keys and values.");

  auto num_key_columns = keys.num_columns();
  auto num_aggregation_columns = values.num_columns();
  auto num_key_rows = keys.num_rows();
  auto num_value_rows = values.num_rows();

  verify_operators(values, ops);

  // Ensure inputs aren't null
  if ((0 == num_key_columns) || (0 == num_aggregation_columns)) {
    CUDF_FAIL("GDF_DATASET_EMPTY");
  }

  // Return immediately if inputs are empty
  CUDF_EXPECTS(0 != num_key_rows, "num_key_rows != 0");
  CUDF_EXPECTS(0 != num_value_rows, "num_value_rows != 0");

  auto in_key_columns = get_pointers(keys);
  
  rmm::device_vector<int32_t> sorted_indices;
  auto gdf_error_code = sort_keys_groupby(num_key_columns, in_key_columns, options, sorted_indices);

  CUDF_EXPECTS(GDF_SUCCESS == gdf_error_code, "sort_keys_groupby error: " + std::to_string(gdf_error_code));

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  cudf::table output_keys{allocate_table_like(keys, stream)};
  cudf::table output_values{allocate_table_like(values, stream)};

  auto out_key_columns = get_pointers(output_keys);
  auto in_aggregation_columns = get_pointers(values);
  auto out_aggregation_columns = get_pointers(output_values);

  gdf_context context;
  context.flag_null_sort_behavior = GDF_NULL_AS_LARGEST;
  context.flag_groupby_include_nulls = !options.ignore_null_keys;

  for (size_t i = 0; i < ops.size(); i++) {
    auto in_aggregation_column = in_aggregation_columns[0];
    auto out_aggregation_column = out_aggregation_columns[0];
    switch (ops[i]) {
    case operators::SUM: {
      /*
       * gdf_error compute_sort_groupby(int num_key_columns,
                                 gdf_column *in_key_columns[],
                                 gdf_column *in_aggregation_column,
                                 gdf_agg_op op,
                                 gdf_column *out_key_columns[],
                                 gdf_column *out_aggregation_column,
                                 gdf_context *context,
                                 rmm::device_vector<int32_t> &sorted_indices) {
       * */
       gdf_error_code = detail::compute_sort_groupby(num_key_columns,
                                                     in_key_columns,
                                                     in_aggregation_column,
                                                     GDF_SUM,
                                                     out_key_columns,
                                                     out_aggregation_column,
                                                     &context,
                                                     sorted_indices);
      CUDF_EXPECTS(GDF_SUCCESS == gdf_error_code, "sort_keys_groupby error: " + std::to_string(gdf_error_code));

      break;
    }
    default: {}
    }
  }
  return std::make_pair(output_keys, output_values);
}

} // END: namespace sort
} // END: namespace groupby
} // END: namespace cudf
