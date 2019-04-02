#pragma once

#include <cuda_runtime.h>
#include <cudf.h>

gdf_error
alloc_filtered_d_cols(const gdf_size_type sort_by_ncols,
                      void **&            out_filtered_left_d_cols_data,
                      void **&            out_filtered_right_d_cols_data,
                      gdf_valid_type **&  out_filtered_left_d_valids_data,
                      gdf_valid_type **&  out_filtered_right_d_valids_data,
                      std::int32_t *&     out_filtered_left_d_col_types,
                      std::int32_t *&     out_filtered_right_d_col_types,
                      cudaStream_t        cudaStream);
