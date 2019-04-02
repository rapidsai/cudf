#include <cuda_runtime.h>
#include <cudf.h>

#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/tuple.h>

#include "soa_info.cuh"
#include "typed_sorted_merge.cuh"

gdf_error
gdf_sorted_merge(gdf_column **       left_cols,
                 gdf_column **       right_cols,
                 const gdf_size_type ncols,
                 gdf_column *        sort_by_cols,
                 gdf_column *        asc_desc,
                 gdf_column **       output_cols) {
    const gdf_size_type left_size  = left_cols[0]->size;
    const gdf_size_type right_size = right_cols[0]->size;

    const gdf_size_type total_size = left_size + right_size;

    gdf_column sides{nullptr, nullptr, total_size, GDF_INT32, 0, {}, nullptr};
    gdf_column indices{nullptr, nullptr, total_size, GDF_INT32, 0, {}, nullptr};

    rmmError_t rmmStatus;

    rmmStatus =
        RMM_ALLOC(&sides.data, sizeof(gdf_size_type) * total_size, nullptr);
    if (RMM_SUCCESS != rmmStatus) { return GDF_MEMORYMANAGER_ERROR; }

    rmmStatus =
        RMM_ALLOC(&indices.data, sizeof(gdf_size_type) * total_size, nullptr);
    if (RMM_SUCCESS != rmmStatus) {
        RMM_FREE(sides.data, nullptr);
        return GDF_MEMORYMANAGER_ERROR;
    }

    gdf_error gdf_status = typed_sorted_merge(left_cols,
                                              right_cols,
                                              ncols,
                                              sort_by_cols,
                                              asc_desc,
                                              &sides,
                                              &indices,
                                              nullptr);
    if (GDF_SUCCESS != gdf_status) { return gdf_status; }

    const thrust::zip_iterator<thrust::tuple<gdf_size_type *, gdf_size_type *>>
        output_zip_iterator = thrust::make_zip_iterator(
            thrust::make_tuple(static_cast<gdf_size_type *>(sides.data),
                               static_cast<gdf_size_type *>(indices.data)));

    INITIALIZE_D_VALUES(left);
    INITIALIZE_D_VALUES(right);
    INITIALIZE_D_VALUES(output);

    // compute values from indices
    thrust::for_each_n(
        thrust::device,
        thrust::make_zip_iterator(thrust::make_tuple(
            thrust::make_counting_iterator(0), output_zip_iterator)),
        total_size,
        [=] __device__(
            thrust::tuple<gdf_size_type,
                          thrust::tuple<gdf_size_type, gdf_size_type>>
                group_tuple) {
            thrust::tuple<gdf_size_type, gdf_size_type> output_tuple =
                thrust::get<1>(group_tuple);

            const gdf_size_type side = thrust::get<0>(output_tuple);
            const gdf_size_type pos  = thrust::get<1>(output_tuple);

            for (gdf_size_type i = 0; i < ncols; i++) {
                const gdf_dtype output_type =
                    static_cast<gdf_dtype>(output_d_col_types[i]);

#define CASE(DTYPE, CTYPE)                                                    \
    case DTYPE: {                                                             \
        CTYPE *output = reinterpret_cast<CTYPE *>(output_d_cols_data[i]);     \
        output[thrust::get<0>(group_tuple)] =                                 \
            0 == side ? reinterpret_cast<CTYPE *>(left_d_cols_data[i])[pos]   \
                      : reinterpret_cast<CTYPE *>(right_d_cols_data[i])[pos]; \
    } break

                switch (output_type) {
                    CASE(GDF_INT8, std::int8_t);
                    CASE(GDF_INT16, std::int16_t);
                    CASE(GDF_INT32, std::int32_t);
                    CASE(GDF_INT64, std::int64_t);
                    CASE(GDF_FLOAT32, float);
                    CASE(GDF_FLOAT64, double);
                    CASE(GDF_DATE32, std::int32_t);
                    CASE(GDF_DATE64, std::int64_t);
                    default:
                        assert(false && "comparison: invalid output gdf_type");
                }
            }
        });

    // compute valids
    cudaError_t cudaStatus;
    for (std::size_t i = 0; i < static_cast<std::size_t>(ncols); i++) {
        const gdf_column *left_col  = left_cols[i];
        const gdf_column *right_col = right_cols[i];

        gdf_column *output_col = output_cols[i];

        if ((nullptr == left_col->valid) && (nullptr == right_col->valid)) {
            break;
        }

        output_col->null_count = left_col->null_count + right_col->null_count;

        const gdf_size_type total_valids = total_size - output_col->null_count;

        cudaStatus = cudaMemset(
            output_col->valid, 0, gdf_valid_allocation_size(total_size));
        if (cudaSuccess != cudaStatus) {
            RMM_FREE(sides.data, nullptr);
            RMM_FREE(indices.data, nullptr);
            return GDF_CUDA_ERROR;
        }

        const std::size_t ones_size = total_valids / GDF_VALID_BITSIZE;
        cudaStatus = cudaMemset(output_col->valid, -1, ones_size);
        if (cudaSuccess != cudaStatus) {
            RMM_FREE(sides.data, nullptr);
            RMM_FREE(indices.data, nullptr);
            return GDF_CUDA_ERROR;
        }

        const std::size_t partial_size = total_valids % GDF_VALID_BITSIZE;
        if (0 < partial_size) {
            cudaStatus = cudaMemset(output_col->valid + ones_size,
                                    static_cast<std::uint8_t>(-1) >>
                                        (sizeof(std::uint64_t) - partial_size),
                                    1);
            if (cudaSuccess != cudaStatus) {
                RMM_FREE(sides.data, nullptr);
                RMM_FREE(indices.data, nullptr);
                return GDF_CUDA_ERROR;
            }
        }
    }

    RMM_FREE(sides.data, nullptr);
    RMM_FREE(indices.data, nullptr);

    return GDF_SUCCESS;
}
