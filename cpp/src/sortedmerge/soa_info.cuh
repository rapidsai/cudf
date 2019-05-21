#pragma once

#include <cudf.h>
#include "rmm/thrust_rmm_allocator.h"
#include "utilities/cudf_utils.h"

#define INITIALIZE_D_VALUES(PREFIX)                                            \
    rmm::device_vector<void *>           PREFIX##_d_cols(ncols);               \
    rmm::device_vector<gdf_valid_type *> PREFIX##_d_valids(ncols);             \
    rmm::device_vector<int>              PREFIX##_d_types(ncols, 0);           \
                                                                               \
    void **          PREFIX##_d_cols_data   = PREFIX##_d_cols.data().get();    \
    gdf_valid_type **PREFIX##_d_valids_data = PREFIX##_d_valids.data().get();  \
    int *            PREFIX##_d_col_types   = PREFIX##_d_types.data().get();   \
                                                                               \
    do {                                                                       \
        gdf_error gdf_status = soa_col_info(PREFIX##_cols,                     \
                                            ncols,                             \
                                            PREFIX##_d_cols_data,              \
                                            PREFIX##_d_valids_data,            \
                                            PREFIX##_d_col_types);             \
        if (GDF_SUCCESS != gdf_status) { return gdf_status; }                  \
    } while (0)
