#include <math.h>

#include <gdf/gdf.h>
#include <gdf/errorutils.h>


__device__
bool gdf_is_valid(const gdf_valid_type *valid, gdf_index_type pos) {
    return (valid[pos / GDF_VALID_BITSIZE] >> (pos % GDF_VALID_BITSIZE)) & 1;
}

template<typename T, typename F>
__global__
void gpu_unary_op(const T *data, const gdf_valid_type *valid,
                    gdf_size_type size, T *results, F functor) {
    int tid = threadIdx.x;
    int blkid = blockIdx.x;
    int blksz = blockDim.x;
    int gridsz = gridDim.x;

    int start = tid + blkid * blksz;
    int step = blksz * gridsz;
    if ( valid ) {  // has valid mask
        for (int i=start; i<size; i+=step) {
            if ( gdf_is_valid(valid, i) )
                results[i] = functor.apply(data[i]);
        }
    } else {        // no valid mask
        for (int i=start; i<size; i+=step) {
            results[i] = functor.apply(data[i]);
        }
    }
}

template<typename T, typename F>
struct UnaryOp {
    static
    int launch(gdf_column *input, gdf_column *output) {
        // find optimal blocksize
        int mingridsize, blocksize;
        CUDA_TRY(
            cudaOccupancyMaxPotentialBlockSize(&mingridsize, &blocksize,
                                               gpu_unary_op<T, F>)
        );
        // find needed gridsize
        int gridsize = (input->size + blocksize - 1) / blocksize;

        F functor;
        gpu_unary_op<<<gridsize, blocksize>>>(
            // input
            (const T*)input->data, input->valid, input->size,
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
struct DeviceSin {
    __device__
    T apply(T data) {
        return sin(data);
    }
};

int gdf_sin_generic(gdf_column *input, gdf_column *output) {
    switch ( input->dtype ) {
    case GDF_FLOAT32:
        return gdf_sin_f32(input, output);
    default:
        return 1;
    }
}


int gdf_sin_f32(gdf_column *input, gdf_column *output) {
    return UnaryOp<float, DeviceSin<float> >::launch(input, output);
}
