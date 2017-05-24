#include <gdf/gdf.h>
#include <gdf/utils.h>
#include <gdf/errorutils.h>


template<typename T, typename F>
__global__
void gpu_binary_op(const T *lhs_data, const gdf_valid_type *lhs_valid,
                   const T *rhs_data, const gdf_valid_type *rhs_valid,
                   gdf_size_type size, T *results, F functor) {
    int tid = threadIdx.x;
    int blkid = blockIdx.x;
    int blksz = blockDim.x;
    int gridsz = gridDim.x;

    int start = tid + blkid * blksz;
    int step = blksz * gridsz;
    if ( lhs_valid || rhs_valid ) {  // has valid mask
        for (int i=start; i<size; i+=step) {
            if (gdf_is_valid(lhs_valid, i) && gdf_is_valid(rhs_valid, i))
                results[i] = functor.apply(lhs_data[i], rhs_data[i]);
        }
    } else {                         // no valid mask
        for (int i=start; i<size; i+=step) {
            results[i] = functor.apply(lhs_data[i], rhs_data[i]);
        }
    }
}

template<typename T, typename F>
struct BinaryOp {
    static
    int launch(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
        // find optimal blocksize
        int mingridsize, blocksize;
        CUDA_TRY(
            cudaOccupancyMaxPotentialBlockSize(&mingridsize, &blocksize,
                                               gpu_binary_op<T, F>)
        );
        // find needed gridsize
        int gridsize = (lhs->size + blocksize - 1) / blocksize;

        F functor;
        gpu_binary_op<<<gridsize, blocksize>>>(
            // inputs
            (const T*)lhs->data, lhs->valid,
            (const T*)rhs->data, rhs->valid,
            lhs->size,
            // output
            (T*)output->data,
            // action
            functor
        );

        CUDA_CHECK_LAST();
        return 0;
    }
};


template<typename T>
struct DeviceAdd {
    __device__
    T apply(T lhs, T rhs) {
        return lhs + rhs;
    }
};

int gdf_add_f32(gdf_column *lhs, gdf_column *rhs, gdf_column *output) {
    return BinaryOp<float, DeviceAdd<float> >::launch(lhs, rhs, output);
}
