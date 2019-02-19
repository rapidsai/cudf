#include "cudf.h"
#include "rmm/rmm.h"
#include "utilities/cudf_utils.h"
#include "utilities/error_utils.h"
#include "bitmask/bit_mask.cuh"
#include "utilities/type_dispatcher.hpp"

#include <cub/device/device_scan.cuh>

namespace { //anonymous

#define COPYMASK_BLOCK_SIZE 1024
#define ELEMENT_SIZE 32

    template <class T>
    __global__
        void gpu_copy_mask(const T *data, const gdf_valid_type *mask,
            gdf_size_type size, T *results, T identity)
    {
        int id = blockIdx.x * blockDim.x + threadIdx.x;
        if (id >= size)return;

        results[id] = (gdf_is_valid(mask, id)) ? data[id] : identity;
    }

    template <class T>
    inline
        gdf_error apply_copy_mask(const T *data, const gdf_valid_type *mask,
            gdf_size_type size, T *results, T identity, cudaStream_t stream)
    {
        int blocksize = (size < COPYMASK_BLOCK_SIZE ?
            size : COPYMASK_BLOCK_SIZE);
        int gridsize = (size + COPYMASK_BLOCK_SIZE - 1) /
            COPYMASK_BLOCK_SIZE;

        // launch kernel
        gpu_copy_mask << <gridsize, blocksize, 0, stream >> > (
            data, mask, size, results, identity);

        CUDA_CHECK_LAST();
        return GDF_SUCCESS;
    }

    template <class T>
    struct Scan {
        static
            gdf_error call(const gdf_column *inp, gdf_column *out, bool inclusive,
                cudaStream_t stream) {
            using cub::DeviceScan;
            auto scan_function = (inclusive ? inclusive_sum : exclusive_sum);
            size_t size = inp->size;
            const T* input = static_cast<const T*>(inp->data);
            T* output = static_cast<T*>(out->data);

            // Prepare temp storage
            void *temp_storage = NULL;
            size_t temp_storage_bytes = 0;
            scan_function(temp_storage, temp_storage_bytes, input, output,
                size, stream);
            RMM_TRY(RMM_ALLOC(&temp_storage, temp_storage_bytes, stream));

            if (inp->valid) {
                // allocate temporary column data
                T* temp_inp;
                RMM_TRY(RMM_ALLOC(&temp_inp, size * sizeof(T) + sizeof(gdf_size_type), stream));

                // copy bitmask
                size_t valid_byte_length = gdf_get_num_chars_bitmask(size);
                CUDA_TRY(cudaMemcpyAsync(out->valid, inp->valid,
                        valid_byte_length, cudaMemcpyDeviceToDevice, stream));
                out->null_count = inp->null_count;

                // copy input data and replace with 0 if mask is not valid
                apply_copy_mask(static_cast<const T*>(inp->data), inp->valid,
                    size, temp_inp, static_cast<T>(0), stream);

                // Do scan
                scan_function(temp_storage, temp_storage_bytes, temp_inp,
                    output, size, stream);

                RMM_TRY(RMM_FREE(temp_inp, stream));
            }
            else {  // Do scan
                scan_function(temp_storage, temp_storage_bytes,
                    input, output, size, stream);
            }

            // Cleanup
            RMM_TRY(RMM_FREE(temp_storage, stream));

            return GDF_SUCCESS;
        }

        static
            gdf_error exclusive_sum(void *&temp_storage, size_t &temp_storage_bytes,
                const T *inp, T *out, size_t size, cudaStream_t stream) {
            cub::DeviceScan::ExclusiveSum(temp_storage, temp_storage_bytes,
                inp, out, size, stream);
            CUDA_CHECK_LAST();
            return GDF_SUCCESS;
        }

        static
            gdf_error inclusive_sum(void *&temp_storage, size_t &temp_storage_bytes,
                const T *inp, T *out, size_t size, cudaStream_t stream) {
            cub::DeviceScan::InclusiveSum(temp_storage, temp_storage_bytes,
                inp, out, size, stream);
            CUDA_CHECK_LAST();
            return GDF_SUCCESS;
        }
    };

    struct PrefixSumDispatcher {
        template <typename T,
            typename std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
            gdf_error operator()(gdf_column *inp, gdf_column *out,
                bool inclusive, cudaStream_t stream = 0) {
            GDF_REQUIRE(inp->size == out->size, GDF_COLUMN_SIZE_MISMATCH);
            GDF_REQUIRE(inp->dtype == out->dtype, GDF_DTYPE_MISMATCH);

            if (!inp->valid) {
                GDF_REQUIRE(!inp->valid || !inp->null_count, GDF_VALIDITY_MISSING);
                GDF_REQUIRE(!out->valid, GDF_VALIDITY_MISSING);
            }
            else {
                GDF_REQUIRE(inp->valid && out->valid, GDF_DTYPE_MISMATCH);
            }
            return Scan<T>::call(inp, out, inclusive, stream);
        }

        template <typename T,
            typename std::enable_if_t<!std::is_arithmetic<T>::value, T>* = nullptr>
            gdf_error operator()(gdf_column *inp, gdf_column *out,
                bool inclusive, cudaStream_t stream = 0) {
            return GDF_UNSUPPORTED_DTYPE;
        }
    };

} // end anonymous namespace

gdf_error gdf_prefixsum(gdf_column *inp, gdf_column *out,
    bool inclusive)
{
    return cudf::type_dispatcher(inp->dtype, PrefixSumDispatcher(),
        inp, out, inclusive);
}
