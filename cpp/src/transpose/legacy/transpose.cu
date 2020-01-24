/* Copyright 2018 NVIDIA Corporation.  All rights reserved. */

#include <cudf/utilities/nvtx_utils.hpp>
#include <cudf/utilities/legacy/type_dispatcher.hpp>
#include <rmm/thrust_rmm_allocator.h>
#include <bitmask/legacy/legacy_bitmask.hpp>
#include <cudf/cudf.h>
#include <cub/cub.cuh>
#include <memory>
#include <stdio.h>
#include <algorithm>

namespace
{

constexpr int WARP_SIZE = 32;
constexpr int MAX_GRID_SIZE = (1<<16)-1;

/**
 * @brief Transposes the values from ncols x nrows input columns to
 *  nrows x ncols output columns
 * 
 * @tparam ColumnType  Datatype of values pointed to by the pointers
 * @param in_cols[in]  Pointers to input columns' data
 * @param out_cols[out]  Pointers to pre-allocated output columns' data
 * @param ncols[in]  Number of columns in input table
 * @param nrows[in]  Number of rown in input table
 */
template <typename ColumnType>
__global__
void gpu_transpose(ColumnType **in_cols, ColumnType **out_cols,
                  cudf::size_type ncols, cudf::size_type nrows)
{
  cudf::size_type x = blockIdx.x * blockDim.x + threadIdx.x;
  cudf::size_type y = blockIdx.y * blockDim.y + threadIdx.y;
    
  cudf::size_type stride_x = blockDim.x * gridDim.x;
  cudf::size_type stride_y = blockDim.y * gridDim.y;

  for(cudf::size_type i = x; i < ncols; i += stride_x)
  {
    for(cudf::size_type j = y; j < nrows; j += stride_y)
    {
      out_cols[j][i] = in_cols[i][j];
    }
  }
}

/**
 * @brief Transposes the validity mask
 * 
 * @param[in] in_cols_valid  pointers to the validity mask of the input columns
 * @param[out] out_cols_valid  pointers to the pre-allocated validity mask of
 *  the output columns
 * @param[out] out_cols_null_count  array of per output-row null counts
 * @param[in] ncols  number of columns in input table
 * @param[in] nrows  number of rows in input table
 */
__global__
void gpu_transpose_valids(cudf::valid_type **in_cols_valid,
                          cudf::valid_type **out_cols_valid,
                          cudf::size_type *out_cols_null_count,
                          cudf::size_type ncols, cudf::size_type nrows)
{
  using MaskType = uint32_t;
  constexpr uint32_t BITS_PER_MASK{sizeof(MaskType) * 8};

  cudf::size_type x = blockIdx.x * blockDim.x + threadIdx.x;
  cudf::size_type y = blockIdx.y * blockDim.y + threadIdx.y;

  cudf::size_type stride_x = blockDim.x * gridDim.x;
  cudf::size_type stride_y = blockDim.y * gridDim.y;

  cudf::size_type i = x;
  cudf::size_type j = y;
  auto active_threads = __ballot_sync(0xffffffff, i < ncols);
  while(i < ncols)
  {
    j = y;
    while(j < nrows)
    {
      bool const input_is_valid{gdf_is_valid(in_cols_valid[i], j)};
      MaskType const result_mask{__ballot_sync(active_threads, input_is_valid)};

      MaskType* const __restrict__ out_mask32 =
        reinterpret_cast<MaskType*>(out_cols_valid[j]);

      cudf::size_type const out_location = i / BITS_PER_MASK;

      // Only one thread writes output
      if (0 == threadIdx.x % warpSize) {
        out_mask32[out_location] = result_mask;
        int num_nulls = __popc(active_threads) - __popc(result_mask);
        atomicAdd(out_cols_null_count + j, num_nulls);
      }
      
      j += stride_y;
    }
    i += stride_x;
    active_threads = __ballot_sync(active_threads, i < ncols);
  }
}

// TODO: refactor and separate `valids` kernel launch into another function.
// Should not need to pass `has_null`
struct launch_kernel{
  template <typename ColumnType>
  gdf_error operator()(
    void **in_cols_data_ptr, void **out_cols_data_ptr,
    cudf::valid_type **in_cols_valid_ptr, cudf::valid_type **out_cols_valid_ptr,
    cudf::size_type *out_cols_nullct_ptr,
    cudf::size_type ncols, cudf::size_type nrows, bool has_null)
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
        out_cols_nullct_ptr,
        ncols, nrows
      );
    }
    cudaDeviceSynchronize();
    CHECK_CUDA(0);
    return GDF_SUCCESS;
  }
};

}

gdf_error gdf_transpose(cudf::size_type ncols, gdf_column** in_cols,
                        gdf_column** out_cols) {
  // Make sure the inputs are not null
  GDF_REQUIRE((ncols > 0) && (nullptr != in_cols) && (nullptr != out_cols),
              GDF_DATASET_EMPTY)

  // If there are no rows in the input, return successfully
  GDF_REQUIRE(in_cols[0]->size > 0, GDF_SUCCESS)

  // Check datatype homogeneity
  gdf_dtype dtype = in_cols[0]->dtype;
  for (cudf::size_type i = 1; i < ncols; i++) {
    GDF_REQUIRE(in_cols[i]->dtype == dtype, GDF_DTYPE_MISMATCH)
  }
  cudf::size_type nrows = in_cols[0]->size;
  cudf::size_type out_ncols = nrows;
  for (cudf::size_type i = 0; i < out_ncols; i++) {
    GDF_REQUIRE(out_cols[i]->dtype == dtype, GDF_DTYPE_MISMATCH)
  }

  // Check if there are nulls to be processed
  bool const has_null{ std::any_of(in_cols, in_cols + ncols, 
    [](gdf_column * col){ return col->null_count > 0; }) };

  if (has_null) {
    for (cudf::size_type i = 0; i < out_ncols; i++) {
      GDF_REQUIRE(out_cols[i]->valid != nullptr, GDF_VALIDITY_MISSING)
    }
  }

  cudf::nvtx::range_push("CUDF_TRANSPOSE", cudf::nvtx::color::GREEN);

  // Copy input columns `data` and `valid` pointers to device
  std::vector<void*> in_columns_data(ncols);
  std::vector<cudf::valid_type*> in_columns_valid(ncols);
  for (cudf::size_type i = 0; i < ncols; ++i) {
    in_columns_data[i] = in_cols[i]->data;
    in_columns_valid[i] = in_cols[i]->valid;
  }
  rmm::device_vector<void*> d_in_columns_data(in_columns_data);
  rmm::device_vector<cudf::valid_type*> d_in_columns_valid(in_columns_valid);

  void** in_cols_data_ptr = d_in_columns_data.data().get();
  cudf::valid_type** in_cols_valid_ptr = d_in_columns_valid.data().get();

  // Copy output columns `data` and `valid` pointers to device
  std::vector<void*> out_columns_data(out_ncols);
  std::vector<cudf::valid_type*> out_columns_valid(out_ncols);
  for (cudf::size_type i = 0; i < out_ncols; ++i) {
    out_columns_data[i] = out_cols[i]->data;
    out_columns_valid[i] = out_cols[i]->valid;
  }
  rmm::device_vector<void*> d_out_columns_data(out_columns_data);
  rmm::device_vector<cudf::valid_type*> d_out_columns_valid(out_columns_valid);
  rmm::device_vector<cudf::size_type> d_out_columns_nullct(out_ncols);

  void** out_cols_data_ptr = d_out_columns_data.data().get();
  cudf::valid_type** out_cols_valid_ptr = d_out_columns_valid.data().get();
  cudf::size_type* out_cols_nullct_ptr = d_out_columns_nullct.data().get();

  cudf::type_dispatcher(dtype,
                        launch_kernel{},
                        in_cols_data_ptr,
                        out_cols_data_ptr,
                        in_cols_valid_ptr,
                        out_cols_valid_ptr,
                        out_cols_nullct_ptr,
                        ncols, nrows, has_null);

  // Transfer null counts to gdf structs
  thrust::host_vector<cudf::size_type> out_columns_nullct(d_out_columns_nullct);
  for(cudf::size_type i = 0; i < out_ncols; i++)
  {
    out_cols[i]->null_count = out_columns_nullct[i];
  }
  
  cudf::nvtx::range_pop();
  return GDF_SUCCESS;
}
