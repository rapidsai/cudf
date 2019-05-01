#include "cudf.h"
#include "rmm/rmm.h"
#include "utilities/cudf_utils.h"
#include "utilities/error_utils.hpp"
#include "bitmask/bit_mask.cuh"
#include "utilities/type_dispatcher.hpp"
#include "bitmask/legacy_bitmask.hpp"

#include "utilities/device_atomics.cuh"
#include <cub/device/device_scan.cuh>
#include <reduction.hpp>

namespace { //anonymous

template <class T>
__global__
    void gpu_copy_and_replace_nulls(
        const T *data, const gdf_valid_type *mask,
        gdf_size_type size, T *results, T identity)
{
    gdf_size_type id = threadIdx.x + blockIdx.x * blockDim.x;

    while (id < size) {
        results[id] = (gdf_is_valid(mask, id)) ? data[id] : identity;
        id += blockDim.x * gridDim.x;
    }
}

/* --------------------------------------------------------------------------*/
/**
 * @brief Copy data stream and replace nulls by a scholar value
 *
 * @param[in] data The stream to be copied
 * @param[in] mask The bitmask stream for nulls
 * @param[in] size The element count of stream
 * @param[out] results The stream for the result
 * @param[in] identity The scholar value to be used to replace nulls
 * @param[in] stream The cuda stream to be used
 *
 * @returns  If the operation was successful, returns GDF_SUCCESS
 */
/* ----------------------------------------------------------------------------*/
template <typename T>
inline
void copy_and_replace_nulls(
        const T *data, const gdf_valid_type *mask,
        gdf_size_type size, T *results, T identity, cudaStream_t stream)
{
    int blockSize=0, minGridSize, gridSize;
    CUDA_TRY( cudaOccupancyMaxPotentialBlockSize(
        &minGridSize, &blockSize, gpu_copy_and_replace_nulls<T>, 0, 0) );
    gridSize = (size + blockSize - 1) / blockSize; 

    // launch kernel
    gpu_copy_and_replace_nulls << <gridSize, blockSize, 0, stream >> > (
        data, mask, size, results, identity);

    CUDA_CHECK_LAST();
}

template <typename T, typename Op>
struct Scan {
    static
    void call(const gdf_column *input, gdf_column *output,
        bool inclusive, cudaStream_t stream)
    {
        auto scan_function = (inclusive ? inclusive_scan : exclusive_scan);
        size_t size = input->size;
        const T* d_input = static_cast<const T*>(input->data);
        T* d_output = static_cast<T*>(output->data);

        // Prepare temp storage
        void *temp_storage = NULL;
        size_t temp_storage_bytes = 0;
        scan_function(temp_storage, temp_storage_bytes,
            d_input, d_output, size, stream);
        RMM_TRY(RMM_ALLOC(&temp_storage, temp_storage_bytes, stream));

        if( nullptr != input->valid ){
            // copy null bitmask
            CUDA_TRY(cudaMemcpyAsync(output->valid, input->valid,
                    gdf_num_bitmask_elements(input->size), cudaMemcpyDeviceToDevice, stream));
            output->null_count = input->null_count;
        }

        bool const input_has_nulls{ nullptr != input->valid &&
                                    input->null_count > 0 };
        if (input_has_nulls) {
            // allocate temporary column data
            T* temp_input;
            RMM_TRY(RMM_ALLOC(&temp_input, size * sizeof(T), stream));

            // copy d_input data and replace with identity if mask is null
            copy_and_replace_nulls(
                static_cast<const T*>(input->data), input->valid,
                size, temp_input, Op::template identity<T>(), stream);

            // Do scan
            scan_function(temp_storage, temp_storage_bytes,
                temp_input, d_output, size, stream);

            RMM_TRY(RMM_FREE(temp_input, stream));
        }
        else {  // Do scan
            scan_function(temp_storage, temp_storage_bytes,
                d_input, d_output, size, stream);
        }

        // Cleanup
        RMM_TRY(RMM_FREE(temp_storage, stream));
    }

    static
    void exclusive_scan(void *&temp_storage, size_t &temp_storage_bytes,
        const T *input, T *output, size_t size, cudaStream_t stream)
    {
        cub::DeviceScan::ExclusiveScan(temp_storage, temp_storage_bytes,
            input, output, Op{}, Op::template identity<T>(), size, stream);
        CUDA_CHECK_LAST();
    }

    static
    void inclusive_scan(void *&temp_storage, size_t &temp_storage_bytes,
        const T *input, T *output, size_t size, cudaStream_t stream)
    {
        cub::DeviceScan::InclusiveScan(temp_storage, temp_storage_bytes,
            input, output, Op{}, size, stream);
        CUDA_CHECK_LAST();
    }
};

template <typename Op>
struct PrefixSumDispatcher {
private:
    // return true if T is arithmetic type (including cudf::bool8)
    template <typename T>
    static constexpr bool is_supported()
    {
        return std::is_arithmetic<T>::value ||
            std::is_same<T, cudf::bool8>::value;
    }
public:
    template <typename T,
              typename std::enable_if_t<is_supported<T>(), T>* = nullptr>
    void operator()(const gdf_column *input, gdf_column *output,
        bool inclusive, cudaStream_t stream = 0)
    {
        CUDF_EXPECTS(input->size == output->size,
            "input and output data size must be same");
        CUDF_EXPECTS(input->dtype == output->dtype,
            "input and output data types must be same");

        CUDF_EXPECTS((nullptr != input->valid) || (0 == input->null_count),
                     "Input column has non-zero null count but no valid data");
        CUDF_EXPECTS((nullptr == output->valid) == (nullptr == input->valid),
                     "Input / output column valid data mismatch");
        
        Scan<T, Op>::call(input, output, inclusive, stream);
    }

    template <typename T,
              typename std::enable_if_t<!is_supported<T>(), T>* = nullptr>
        void operator()(const gdf_column *input, gdf_column *output,
            bool inclusive, cudaStream_t stream = 0) {
            CUDF_FAIL("Non-arithmetic types not supported for `gdf_scan`");
    }
};

} // end anonymous namespace

namespace cudf{

void scan(const gdf_column *input, gdf_column *output,
    gdf_scan_op op, bool inclusive)
{
    CUDF_EXPECTS(input  != nullptr, "Input column is null");
    CUDF_EXPECTS(output != nullptr, "Output column is null");

    switch(op){
    case GDF_SCAN_SUM:
        cudf::type_dispatcher(input->dtype,
            PrefixSumDispatcher<cudf::DeviceSum>(), input, output, inclusive);
        return;
    case GDF_SCAN_MIN:
        cudf::type_dispatcher(input->dtype,
            PrefixSumDispatcher<cudf::DeviceMin>(), input, output, inclusive);
        return;
    case GDF_SCAN_MAX:
        cudf::type_dispatcher(input->dtype,
            PrefixSumDispatcher<cudf::DeviceMax>(), input, output, inclusive);
        return;
    case GDF_SCAN_PRODUCT:
        cudf::type_dispatcher(input->dtype,
            PrefixSumDispatcher<cudf::DeviceProduct>(), input, output, inclusive);
        return;
    default:
        CUDF_FAIL("The input enum `gdf_scan_op` is out of the range");
    }
}

} // end cudf namespace

