#include <gdf/gdf.h>
#include <gdf/utils.h>
#include <gdf/errorutils.h>

#include <cub/block/block_reduce.cuh>

#define REDUCTION_BLOCK_SIZE 128

/*
Generic reduction implementation with support for validity mask
*/

template<typename T, typename F>
__global__
void gpu_reduction_op(const T *data, const gdf_valid_type *mask,
                      gdf_size_type size, T *results, F functor, T identity)
{
    typedef cub::BlockReduce<T, REDUCTION_BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int tid = threadIdx.x;
    int blkid = blockIdx.x;
    int blksz = blockDim.x;
    int gridsz = gridDim.x;

    int step = blksz * gridsz;

    T agg = identity;

    for (int base=blkid * blksz; base<size; base+=step) {
        // Threadblock synchronous loop
        int i = base + tid;
        // load
        T loaded = identity;
        if (i < size)
            loaded = data[i];
        // set invalid location to identity
        if ( !gdf_is_valid(mask, i) ) {
             loaded = identity;
        }
        // Block reduce
        T temp = BlockReduce(temp_storage).Reduce(loaded, functor);
        // Add current block
        agg = functor(agg, temp);
    }
    // First thread of each block stores the result.
    if (tid == 0)
        results[blkid] = agg;
}



template<typename T, typename F>
struct ReduceOp {
    static
    gdf_error launch(gdf_column *input, T identity, T *output,
                     gdf_size_type output_size) {

        // 1st round
        //    Partially reduce the input into *output_size* length.
        //    Each block computes one output in *output*.
        //    output_size == gridsize
        launch_once((const T*)input->data, input->valid, input->size,
                    (T*)output, output_size, identity);
        CUDA_CHECK_LAST();

        // 2nd round
        //    Finish the partial reduction (if needed).
        //    A single block reduction that computes one output stored to the
        //    first index in *output*.
        if ( output_size > 1 ) {
            launch_once(output, nullptr, output_size,
                        output, 1, identity);
            CUDA_CHECK_LAST();
        }

        return GDF_SUCCESS;
    }

    static
    void launch_once(const T *data, gdf_valid_type *valid, gdf_size_type size,
                     T *output, gdf_size_type output_size, T identity) {
        // find needed gridsize
        // use atmost REDUCTION_BLOCK_SIZE blocks
        int blocksize = REDUCTION_BLOCK_SIZE;
        int gridsize = (output_size < REDUCTION_BLOCK_SIZE?
                        output_size : REDUCTION_BLOCK_SIZE);

        F functor;
        // launch kernel
        gpu_reduction_op<<<gridsize, blocksize>>>(
            // inputs
            data, valid, size,
            // output
            output,
            // action
            functor,
            // identity
            identity
        );
    }

};


template<typename T>
struct DeviceSum {
    __device__
    T operator() (const T &lhs, const T &rhs) {
        return lhs + rhs;
    }
};

template<typename T>
struct DeviceProduct {
    __device__
    T operator() (const T &lhs, const T &rhs) {
        return lhs * rhs;
    }
};

#define DEF_REDUCE_OP_NUM(F)                                                      \
gdf_error F##_generic(gdf_column *col, void *dev_result,                          \
                          gdf_size_type dev_result_size) {                        \
    switch ( col->dtype ) {                                                       \
    case GDF_FLOAT64: return F##_f64(col, (double*)dev_result, dev_result_size);  \
    case GDF_FLOAT32: return F##_f32(col, (float*)dev_result, dev_result_size);   \
    case GDF_INT64:   return F##_i64(col, (int64_t*)dev_result, dev_result_size); \
    case GDF_INT32:   return F##_i32(col, (int32_t*)dev_result, dev_result_size); \
    default:          return GDF_UNSUPPORTED_DTYPE;                               \
    }                                                                             \
}

#define DEF_REDUCE_IMPL(F, OP, T, ID)                                         \
gdf_error F(gdf_column *col, T *dev_result, gdf_size_type dev_result_size) {  \
    return ReduceOp<T, OP<T> >::launch(col, ID, dev_result, dev_result_size); \
}


unsigned int gdf_reduce_optimal_output_size() {
    return REDUCTION_BLOCK_SIZE;
}


/* Sum */

DEF_REDUCE_OP_NUM(gdf_sum)
DEF_REDUCE_IMPL(gdf_sum_f64, DeviceSum, double, 0)
DEF_REDUCE_IMPL(gdf_sum_f32, DeviceSum, float, 0)
DEF_REDUCE_IMPL(gdf_sum_i64, DeviceSum, int64_t, 0)
DEF_REDUCE_IMPL(gdf_sum_i32, DeviceSum, int32_t, 0)

/* Product */

DEF_REDUCE_OP_NUM(gdf_product)
DEF_REDUCE_IMPL(gdf_product_f64, DeviceProduct, double, 1)
DEF_REDUCE_IMPL(gdf_product_f32, DeviceProduct, float, 1)
DEF_REDUCE_IMPL(gdf_product_i64, DeviceProduct, int64_t, 1)
DEF_REDUCE_IMPL(gdf_product_i32, DeviceProduct, int32_t, 1)

