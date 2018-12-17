#ifndef GDF_VALID_OPS_H
#define GDF_VALID_OPS_H

#include <cuda_runtime.h>

gdf_error count_nonzero_mask(gdf_valid_type const * masks, int num_rows, int& count, cudaStream_t stream);

#endif
