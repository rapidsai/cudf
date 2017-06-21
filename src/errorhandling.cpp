#include <gdf/gdf.h>

#define GETNAME(x) case x: return #x;
const char * gdf_error_get_name(gdf_error errcode) {
    switch (errcode) {
    GETNAME(GDF_SUCCESS)
    GETNAME(GDF_CUDA_ERROR)
    GETNAME(GDF_UNSUPPORTED_DTYPE)
    GETNAME(GDF_COLUMN_SIZE_MISMATCH)
    GETNAME(GDF_VALIDITY_MISSING)
    }
}
