#include <gdf/gdf.h>
#include <gdf/utils.h>
#include <gdf/errorutils.h>

#include "rmm.h"

#include <cub/device/device_scan.cuh>



template <class T>
struct Scan {
    static
    gdf_error call(const T *inp, T *out, size_t size, bool inclusive) {
        using cub::DeviceScan;

        auto scan_function = (inclusive? inclusive_sum : exclusive_sum);

        // Prepare temp storage
        void *temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        scan_function(temp_storage, temp_storage_bytes, inp, out, size);
        RMM_TRY( rmmAlloc(&temp_storage, temp_storage_bytes, 0) ); // TODO: non-default stream
        // Do scan
        scan_function(temp_storage, temp_storage_bytes, inp, out, size);
        // Cleanup
        RMM_TRY( rmmFree(temp_storage, 0) ); // TODO: non-default stream

        return GDF_SUCCESS;
    }

    static
    gdf_error exclusive_sum(void *&temp_storage, size_t &temp_storage_bytes,
                            const T *inp, T *out, size_t size) {
        cub::DeviceScan::ExclusiveSum(temp_storage, temp_storage_bytes,
                                      inp, out, size);
        CUDA_CHECK_LAST();
        return GDF_SUCCESS;
    }

    static
    gdf_error inclusive_sum(void *&temp_storage, size_t &temp_storage_bytes,
                            const T *inp, T *out, size_t size) {
        cub::DeviceScan::InclusiveSum(temp_storage, temp_storage_bytes,
                                      inp, out, size);
        CUDA_CHECK_LAST();
        return GDF_SUCCESS;
    }
};

#define SCAN_IMPL(F, T)                                                       \
gdf_error gdf_prefixsum_##F(gdf_column *inp, gdf_column *out, int inclusive) {\
    GDF_REQUIRE( inp->size == out->size, GDF_COLUMN_SIZE_MISMATCH );          \
    GDF_REQUIRE( inp->dtype == out->dtype, GDF_UNSUPPORTED_DTYPE );           \
    GDF_REQUIRE( !inp->valid , GDF_VALIDITY_UNSUPPORTED );                    \
    GDF_REQUIRE( !out->valid , GDF_VALIDITY_UNSUPPORTED );                    \
    return Scan<T>::call((const T*)inp->data, (T*)out->data, inp->size,       \
                         inclusive);                                          \
}


SCAN_IMPL(i8,  int8_t)
SCAN_IMPL(i32, int32_t)
SCAN_IMPL(i64, int64_t)


gdf_error gdf_prefixsum_generic(gdf_column *inp, gdf_column *out,
                                int inclusive)
{
    switch (inp->dtype) {
    case GDF_INT8:    return gdf_prefixsum_i8(inp, out, inclusive);
    case GDF_INT32:   return gdf_prefixsum_i32(inp, out, inclusive);
    case GDF_INT64:   return gdf_prefixsum_i64(inp, out, inclusive);
    default: return GDF_SUCCESS;
    }
}
