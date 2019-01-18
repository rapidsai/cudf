/* Copyright 2018 NVIDIA Corporation.  All rights reserved. */

#include <cudf.h>
#include <stdio.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <memory>
#include "dataframe/cudf_table.cuh"
#include "utilities/nvtx/nvtx_utils.h"

gdf_error gdf_transpose(gdf_size_type ncols, gdf_column** in_cols,
                        gdf_column** out_cols) {
  // Make sure the inputs are not null
  GDF_REQUIRE((ncols > 0) && (nullptr != in_cols) && (nullptr != out_cols),
              GDF_DATASET_EMPTY)

  // If there are no rows in the input, return successfully
  GDF_REQUIRE(in_cols[0]->size > 0, GDF_SUCCESS)

  // Check datatype homogeneity
  gdf_dtype dtype = in_cols[0]->dtype;
  for (gdf_size_type i = 1; i < ncols; i++) {
    GDF_REQUIRE(in_cols[i]->dtype == dtype, GDF_DTYPE_MISMATCH)
  }
  gdf_size_type out_ncols = in_cols[0]->size;
  for (gdf_size_type i = 0; i < out_ncols; i++) {
    GDF_REQUIRE(out_cols[i]->dtype == dtype, GDF_DTYPE_MISMATCH)
  }

  // Check if there are nulls to be processed
  bool has_null = false;
  for (gdf_size_type i = 0; i < ncols; i++) {
    if (in_cols[i]->null_count > 0) {
      has_null = true;
      break;
    }
  }

  if (has_null) {
    for (gdf_size_type i = 0; i < out_ncols; i++) {
      GDF_REQUIRE(out_cols[i]->valid != nullptr, GDF_VALIDITY_MISSING)
    }
  }

  PUSH_RANGE("CUDF_TRANSPOSE", GDF_GREEN);
  // Wrap the input columns in a gdf_table
  using size_type = decltype(ncols);

  std::unique_ptr<const gdf_table<size_type> > input_table{
      new gdf_table<size_type>(ncols, in_cols)};

  // Copy output columns `data` and `valid` pointers to device
  std::vector<void*> out_columns_data(out_ncols);
  std::vector<gdf_valid_type*> out_columns_valid(out_ncols);
  for (gdf_size_type i = 0; i < out_ncols; ++i) {
    out_columns_data[i] = out_cols[i]->data;
    out_columns_valid[i] = out_cols[i]->valid;
  }
  rmm::device_vector<void*> d_out_columns_data(out_columns_data);
  rmm::device_vector<gdf_valid_type*> d_out_columns_valid(out_columns_valid);

  auto input_table_ptr = input_table.get();
  void** out_cols_data_ptr = d_out_columns_data.data().get();
  gdf_valid_type** out_cols_valid_ptr = d_out_columns_valid.data().get();

  auto copy_to_outcol = [input_table_ptr, out_cols_data_ptr, out_cols_valid_ptr,
                         has_null] __device__(gdf_size_type i) {

    input_table_ptr->get_packed_row_values(i, out_cols_data_ptr[i]);

    if (has_null) {
      input_table_ptr->get_row_valids(i, out_cols_valid_ptr[i]);
    }
  };

  thrust::for_each(
      rmm::exec_policy(), thrust::counting_iterator<gdf_size_type>(0),
      thrust::counting_iterator<gdf_size_type>(out_ncols), copy_to_outcol);

  POP_RANGE();
  return GDF_SUCCESS;
}