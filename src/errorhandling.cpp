#include <gdf/gdf.h>

#define GETNAME(x) case x: return #x;

const char * gdf_error_get_name(gdf_error errcode) {
    switch (errcode) {
    // There must be one entry per enum values in gdf_error.
    GETNAME(GDF_SUCCESS)
    GETNAME(GDF_CUDA_ERROR)
    GETNAME(GDF_UNSUPPORTED_DTYPE)
    GETNAME(GDF_COLUMN_SIZE_MISMATCH)
    GETNAME(GDF_COLUMN_SIZE_TOO_BIG)
    GETNAME(GDF_VALIDITY_MISSING)
    GETNAME(GDF_VALIDITY_UNSUPPORTED)
    default:
        // This means we are missing an entry above for a gdf_error value.
        return "Internal error. Unknown error code.";
    }
}
