#include "cudf.h"
#include "rmm/rmm.h"
#include "utilities/cudf_utils.h"
#include "utilities/error_utils.h"
#include "bitmask/bit_mask.cuh"
#include "utilities/type_dispatcher.hpp"

#include <cub/device/device_scan.cuh>


template <class T>
__global__
void gpu_copy_mask(const T *data, const gdf_valid_type *mask,
    gdf_size_type size, T *results, T identity)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= size)return;

    const BitMask bitmask(reinterpret_cast<bit_mask_t*>(const_cast<gdf_valid_type*>(mask)), size);
    bool presence = bitmask.is_valid(id);
    results[id] = (presence) ? data[id] : identity;
}

#define COPYMASK_BLOCK_SIZE 1024

template <class T>
inline
void apply_copy_mask(const T *data, const gdf_valid_type *mask,
    gdf_size_type size, T *results, T identity)
{
    int blocksize = (size < COPYMASK_BLOCK_SIZE ?
        size : COPYMASK_BLOCK_SIZE);
    int gridsize = (size + COPYMASK_BLOCK_SIZE - 1) /
        COPYMASK_BLOCK_SIZE;

    // launch kernel
    gpu_copy_mask << <gridsize, blocksize >> > (
        data, mask, size, results, identity);
}


template <class T>
struct Scan {
    static
    gdf_error call(const gdf_column *inp, gdf_column *out, bool inclusive) {
        using cub::DeviceScan;
        cudaStream_t stream = 0;  // TODO: non-default stream
        auto scan_function = (inclusive? inclusive_sum : exclusive_sum);
        size_t size = inp->size;
        const T* input = reinterpret_cast<const T*>(inp->data);
        T* output = reinterpret_cast<T*>(out->data);

        // Prepare temp storage
        void *temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        scan_function(temp_storage, temp_storage_bytes, input, output, size);
        RMM_TRY( RMM_ALLOC(&temp_storage, temp_storage_bytes, stream) );

        if (inp->valid) {
            // copy bitmask
            size_t valid_byte_length = (size + 7) / 8;
            cudaError_t error = cudaMemcpyAsync(out->valid, inp->valid,
                valid_byte_length, cudaMemcpyDeviceToDevice, stream);
            out->null_count = inp->null_count;

            T* temp_inp;    
            RMM_TRY(RMM_ALLOC(&temp_inp, size * sizeof(T), stream));
            // copy input data and replace with 0 if mask is not valid
            apply_copy_mask(reinterpret_cast<const T*>(inp->data), inp->valid,
                size, temp_inp, static_cast<T>(0));

            // Do scan
            scan_function(temp_storage, temp_storage_bytes, temp_inp,
                output, size);

            RMM_TRY(RMM_FREE(temp_inp, stream));

            // TODO: skipna=False support
        }
        else {  // Do scan
            scan_function(temp_storage, temp_storage_bytes,
                input, output, size);
        }

        // Cleanup
        RMM_TRY( RMM_FREE(temp_storage, stream) );

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

        if (!inp->valid) {
            GDF_REQUIRE(!inp->valid || !inp->null_count, GDF_VALIDITY_UNSUPPORTED);
            GDF_REQUIRE(!out->valid || !out->null_count, GDF_VALIDITY_UNSUPPORTED);
        }
        else {
            GDF_REQUIRE(inp->valid && out->valid, GDF_VALIDITY_UNSUPPORTED);
        }
        return Scan<T>::call(inp, out, inclusive);
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
