#pragma once

#include <cuda_runtime.h>
#include <cudf.h>

gdf_error typed_sorted_merge(gdf_column **     left_cols,
                             gdf_column **     right_cols,
                             const gdf_size_type ncols,
                             gdf_column *      sort_by_cols,
                             gdf_column *      asc_desc,
                             gdf_column *      output_sides,
                             gdf_column *      output_indices,
                             cudaStream_t      cudaStream);
