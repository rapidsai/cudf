/* Copyright 2018 NVIDIA Corporation.  All rights reserved. */

#include <cudf.h>
#include <stdio.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <cub/cub.cuh>
#include <memory>
#include "dataframe/cudf_table.cuh"
#include "utilities/nvtx/nvtx_utils.h"
#include "utilities/type_dispatcher.hpp"
#include <chrono>

constexpr int WARP_SIZE = 32;
constexpr int MAX_GRID_SIZE = (1<<16)-1;

template <typename ColumnType>
__global__
void gpu_transpose(ColumnType **in_cols, ColumnType **out_cols,
                  gdf_size_type ncols, gdf_size_type nrows)
{
  gdf_size_type x = blockIdx.x * blockDim.x + threadIdx.x;
  gdf_size_type y = blockIdx.y * blockDim.y + threadIdx.y;
    
  gdf_size_type stride_x = blockDim.x * gridDim.x;
  gdf_size_type stride_y = blockDim.y * gridDim.y;

  for(gdf_size_type i = x; i < ncols; i += stride_x)
  {
    for(gdf_size_type j = y; j < nrows; j += stride_y)
    {
      out_cols[j][i] = in_cols[i][j];
    }
  }
}

__global__
void gpu_transpose_valids(gdf_valid_type **in_cols_valid,
                          gdf_valid_type **out_cols_valid,
                          gdf_size_type ncols, gdf_size_type nrows)
{
  using MaskType = uint32_t;
  constexpr uint32_t BITS_PER_MASK{sizeof(MaskType) * 8};

  gdf_size_type x = blockIdx.x * blockDim.x + threadIdx.x;
  gdf_size_type y = blockIdx.y * blockDim.y + threadIdx.y;

  gdf_size_type stride_x = blockDim.x * gridDim.x;
  gdf_size_type stride_y = blockDim.y * gridDim.y;

  MaskType* const __restrict__ out_mask32 =
    reinterpret_cast<MaskType*>(out_cols_valid[y]);

  for(gdf_size_type i = x; i < ncols; i += stride_x)
  {
    for(gdf_size_type j = y; j < nrows; j += stride_y)
    {
      auto active_threads = __ballot_sync(0xffffffff, x < ncols && y < nrows);
      if (x < ncols && y < nrows)
      {
        bool const input_is_valid{gdf_is_valid(in_cols_valid[x], y)};

        MaskType const result_mask{__ballot_sync(active_threads, input_is_valid)};

        gdf_index_type const out_location = x / BITS_PER_MASK;

        // Only one thread writes output
        if (0 == threadIdx.x % warpSize) {
          out_mask32[out_location] = result_mask;
        }
      }
    }
  }
}

// TODO: refactor and separate `valids` kernel launch into another function.
// Should not need to pass `has_null`
struct launch_kernel{
  template <typename ColumnType>
  gdf_error operator()(
    void **in_cols_data_ptr, void **out_cols_data_ptr,
    gdf_valid_type **in_cols_valid_ptr, gdf_valid_type **out_cols_valid_ptr,
    gdf_size_type ncols, gdf_size_type nrows, bool has_null)
  {
    dim3 dimBlock(WARP_SIZE, WARP_SIZE, 1);
    dim3 dimGrid(std::min((ncols + WARP_SIZE - 1) / WARP_SIZE, MAX_GRID_SIZE),
                 std::min((nrows + WARP_SIZE - 1) / WARP_SIZE, MAX_GRID_SIZE),
                 1);

    gpu_transpose<ColumnType><<<dimGrid,dimBlock>>>(
      reinterpret_cast<ColumnType**>(in_cols_data_ptr),
      reinterpret_cast<ColumnType**>(out_cols_data_ptr),
      ncols, nrows
    );
    if (has_null){
      gpu_transpose_valids<<<dimGrid,dimBlock>>>(
        in_cols_valid_ptr,
        out_cols_valid_ptr,
        ncols, nrows
      );
    }
    cudaDeviceSynchronize();
    CUDA_CHECK_LAST();
    return GDF_SUCCESS;
  }
};

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
  gdf_size_type nrows = in_cols[0]->size;
  gdf_size_type out_ncols = nrows;
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

  void** out_cols_data_ptr = d_out_columns_data.data().get();
  gdf_valid_type** out_cols_valid_ptr = d_out_columns_valid.data().get();

  cudf::type_dispatcher(dtype,
                        launch_kernel{},
                        input_table->d_columns_data_ptr,
                        out_cols_data_ptr,
                        input_table->d_columns_valids_ptr,
                        out_cols_valid_ptr,
                        ncols, nrows, has_null);
  POP_RANGE();
  return GDF_SUCCESS;
}