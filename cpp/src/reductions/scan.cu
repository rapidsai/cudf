#include "cudf.h"
#include "rmm/rmm.h"
#include "utilities/cudf_utils.h"
#include "utilities/error_utils.h"
#include "utilities/type_dispatcher.hpp"

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
        RMM_TRY( RMM_ALLOC(&temp_storage, temp_storage_bytes, 0) ); // TODO: non-default stream
        // Do scan
        scan_function(temp_storage, temp_storage_bytes, inp, out, size);
        // Cleanup
        RMM_TRY( RMM_FREE(temp_storage, 0) ); // TODO: non-default stream

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

struct PrefixSumDispatcher {
    template <typename T,
        typename std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
        gdf_error operator()(gdf_column *inp, gdf_column *out,
            int inclusive) {
        GDF_REQUIRE(inp->size == out->size, GDF_COLUMN_SIZE_MISMATCH);
        GDF_REQUIRE(inp->dtype == out->dtype, GDF_UNSUPPORTED_DTYPE);
        GDF_REQUIRE(!inp->valid || !inp->null_count, GDF_VALIDITY_UNSUPPORTED);
        GDF_REQUIRE(!out->valid || !out->null_count, GDF_VALIDITY_UNSUPPORTED);
        return Scan<T>::call((const T*)inp->data, (T*)out->data, inp->size,
            inclusive);
    }

    template <typename T,
        typename std::enable_if_t<!std::is_arithmetic<T>::value, T>* = nullptr>
        gdf_error operator()(gdf_column *inp, gdf_column *out,
            int inclusive) {
        return GDF_UNSUPPORTED_DTYPE;
    }
};

gdf_error gdf_prefixsum(gdf_column *inp, gdf_column *out,
    int inclusive)
{
    return cudf::type_dispatcher(inp->dtype, PrefixSumDispatcher(),
        inp, out, inclusive);
}
