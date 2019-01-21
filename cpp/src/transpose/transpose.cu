/* Copyright 2018 NVIDIA Corporation.  All rights reserved. */

#include <cudf.h>
#include <stdio.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <cub/cub.cuh>
#include <memory>
#include "dataframe/cudf_table.cuh"
#include "utilities/nvtx/nvtx_utils.h"
#include <chrono>

constexpr int BLOCK_SIZE = 32;
constexpr int WARP_SIZE = 32;

// __global__
// void gpu_transpose( int64_t **in_cols, int64_t **out_cols,
//                    gdf_size_type ncols, gdf_size_type nrows)
// {
//   // I'm hardcoding block to be warp size
//   int tid = threadIdx.x;
//   int blkid = blockIdx.x;
//   int blksz = blockDim.x;

//   int idx = blkid * blksz + tid;

//   int64_t thread_data[WARP_SIZE];
  
//   for(size_t i = 0; i < WARP_SIZE; i++)
//   {
//     thread_data[i] = in_cols[i][idx];
//   }
  
//   typedef cub::BlockExchange<int64_t, 32, 32> BlockExchange;
//   __shared__ typename BlockExchange::TempStorage temp_storage;
//   BlockExchange(temp_storage).StripedToBlocked(thread_data);

  
//   for(size_t i = 0; i < WARP_SIZE; i++)
//   {
//     out_cols[i + blkid * blksz][tid] = thread_data[i];
//   }
  
// }


__global__
void gpu_transpose( int64_t **in_cols, int64_t **out_cols,
                   gdf_size_type ncols, gdf_size_type nrows)
{
  // __shared__ float tile[WARP_SIZE][WARP_SIZE];
  int x = blockIdx.x * WARP_SIZE + threadIdx.x;
  int y = blockIdx.y * WARP_SIZE + threadIdx.y;

  out_cols[x][y] = in_cols[y][x];
  // tile[threadIdx.y][threadIdx.x] = in_cols[y][x];

  // __syncthreads();

  // x = blockIdx.y * WARP_SIZE + threadIdx.x;  // transpose block offset
  // y = blockIdx.x * WARP_SIZE + threadIdx.y;

  // out_cols[x][y] = tile[threadIdx.y][threadIdx.x];
}

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

  // auto copy_to_outcol = [input_table_ptr, out_cols_data_ptr, out_cols_valid_ptr,
  //                        has_null] __device__(gdf_size_type i) {

  //   input_table_ptr->get_packed_row_values(i, out_cols_data_ptr[i]);

  //   if (has_null) {
  //     input_table_ptr->get_row_valids(i, out_cols_valid_ptr[i]);
  //   }
  // };

  // auto start = std::chrono::high_resolution_clock::now();
  // thrust::for_each(
  //     rmm::exec_policy(), thrust::counting_iterator<gdf_size_type>(0),
  //     thrust::counting_iterator<gdf_size_type>(out_ncols), copy_to_outcol);
  // cudaDeviceSynchronize();
  // auto end = std::chrono::system_clock::now();
  // std::chrono::duration<double> elapsed_seconds = end-start;
  // std::cout << "Elapsed time (ms): " << elapsed_seconds.count()*1000 << std::endl;


  dim3 dimBlock(WARP_SIZE, WARP_SIZE, 1);
  dim3 dimGrid(100000, 1, 1);
  auto start = std::chrono::high_resolution_clock::now();
  gpu_transpose<<<dimGrid,dimBlock>>>(
    (int64_t **)input_table->d_columns_data,
    (int64_t **)out_cols_data_ptr,
    ncols, out_ncols
  );
  cudaDeviceSynchronize();
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "Elapsed time (ms): " << elapsed_seconds.count()*1000 << std::endl;
  POP_RANGE();
  return GDF_SUCCESS;
}