#include "alloc_filtered_cols.cuh"

#include "rmm/rmm.h"

gdf_error
alloc_filtered_d_cols(const gdf_size_type sort_by_ncols,
                      void **&            out_filtered_left_d_cols_data,
                      void **&            out_filtered_right_d_cols_data,
                      gdf_valid_type **&  out_filtered_left_d_valids_data,
                      gdf_valid_type **&  out_filtered_right_d_valids_data,
                      std::int32_t *&     out_filtered_left_d_col_types,
                      std::int32_t *&     out_filtered_right_d_col_types,
                      cudaStream_t        cudaStream) {
    void **          filtered_left_d_cols_data    = nullptr;
    void **          filtered_right_d_cols_data   = nullptr;
    gdf_valid_type **filtered_left_d_valids_data  = nullptr;
    gdf_valid_type **filtered_right_d_valids_data = nullptr;
    std::int32_t *   filtered_left_d_col_types    = nullptr;
    std::int32_t *   filtered_right_d_col_types   = nullptr;

    rmmError_t rmmStatus;

    rmmStatus = RMM_ALLOC(reinterpret_cast<void **>(&filtered_left_d_cols_data),
                          sizeof(void *) * sort_by_ncols,
                          cudaStream);
    if (RMM_SUCCESS != rmmStatus) { return GDF_MEMORYMANAGER_ERROR; }

    rmmStatus =
        RMM_ALLOC(reinterpret_cast<void **>(&filtered_right_d_cols_data),
                  sizeof(void *) * sort_by_ncols,
                  cudaStream);
    if (RMM_SUCCESS != rmmStatus) {
        RMM_FREE(filtered_left_d_cols_data, cudaStream);
        return GDF_MEMORYMANAGER_ERROR;
    }

    rmmStatus =
        RMM_ALLOC(reinterpret_cast<void **>(&filtered_left_d_valids_data),
                  sizeof(void *) * sort_by_ncols,
                  cudaStream);
    if (RMM_SUCCESS != rmmStatus) {
        RMM_FREE(filtered_left_d_cols_data, cudaStream);
        RMM_FREE(filtered_right_d_cols_data, cudaStream);
        return GDF_MEMORYMANAGER_ERROR;
    }

    rmmStatus =
        RMM_ALLOC(reinterpret_cast<void **>(&filtered_right_d_valids_data),
                  sizeof(void *) * sort_by_ncols,
                  cudaStream);
    if (RMM_SUCCESS != rmmStatus) {
        RMM_FREE(filtered_left_d_cols_data, cudaStream);
        RMM_FREE(filtered_right_d_cols_data, cudaStream);
        RMM_FREE(filtered_left_d_valids_data, cudaStream);
        return GDF_MEMORYMANAGER_ERROR;
    }

    rmmStatus = RMM_ALLOC(reinterpret_cast<void **>(&filtered_left_d_col_types),
                          sizeof(std::int32_t) * sort_by_ncols,
                          cudaStream);
    if (RMM_SUCCESS != rmmStatus) {
        RMM_FREE(filtered_left_d_cols_data, cudaStream);
        RMM_FREE(filtered_right_d_cols_data, cudaStream);
        RMM_FREE(filtered_left_d_valids_data, cudaStream);
        RMM_FREE(filtered_right_d_valids_data, cudaStream);
        return GDF_MEMORYMANAGER_ERROR;
    }

    rmmStatus =
        RMM_ALLOC(reinterpret_cast<void **>(&filtered_right_d_col_types),
                  sizeof(std::int32_t) * sort_by_ncols,
                  cudaStream);
    if (RMM_SUCCESS != rmmStatus) {
        RMM_FREE(filtered_left_d_cols_data, cudaStream);
        RMM_FREE(filtered_right_d_cols_data, cudaStream);
        RMM_FREE(filtered_left_d_valids_data, cudaStream);
        RMM_FREE(filtered_right_d_valids_data, cudaStream);
        RMM_FREE(filtered_left_d_col_types, cudaStream);
        return GDF_MEMORYMANAGER_ERROR;
    }

    out_filtered_left_d_cols_data    = filtered_left_d_cols_data;
    out_filtered_right_d_cols_data   = filtered_right_d_cols_data;
    out_filtered_left_d_valids_data  = filtered_left_d_valids_data;
    out_filtered_right_d_valids_data = filtered_right_d_valids_data;
    out_filtered_left_d_col_types    = filtered_left_d_col_types;
    out_filtered_right_d_col_types   = filtered_right_d_col_types;

    return GDF_SUCCESS;
}

