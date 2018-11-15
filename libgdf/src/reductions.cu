#include <gdf/gdf.h>
#include <gdf/utils.h>
#include <gdf/errorutils.h>

#include <cub/block/block_reduce.cuh>

#include <limits>

#define REDUCTION_BLOCK_SIZE 128


struct IdentityLoader{
    template<typename T>
    __device__
    T operator() (const T *ptr, int pos) const {
        return ptr[pos];
    }
};

/*
Generic reduction implementation with support for validity mask
*/

template<typename T, typename F, typename Ld>
__global__
void gpu_reduction_op(const T *data, const gdf_valid_type *mask,
                      gdf_size_type size, T *results, F functor, T identity,
                      Ld loader)
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
        if (i < size && gdf_is_valid(mask, i))
            loaded = loader(data, i);
            
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
        typedef typename F::Loader Ld1;
        F functor1;
        Ld1 loader1;
        launch_once((const T*)input->data, input->valid, input->size,
                    (T*)output, output_size, identity, functor1, loader1);
        CUDA_CHECK_LAST();

        // 2nd round
        //    Finish the partial reduction (if needed).
        //    A single block reduction that computes one output stored to the
        //    first index in *output*.
        if ( output_size > 1 ) {
            typedef typename F::second F2;
            typedef typename F2::Loader Ld2;
            F2 functor2;
            Ld2 loader2;

            launch_once(output, nullptr, output_size,
                        output, 1, identity, functor2, loader2);
            CUDA_CHECK_LAST();
        }

        return GDF_SUCCESS;
    }

    template <typename Functor, typename Loader>
    static
    void launch_once(const T *data, gdf_valid_type *valid, gdf_size_type size,
                     T *output, gdf_size_type output_size, T identity,
                     Functor functor, Loader loader) {
        // find needed gridsize
        // use atmost REDUCTION_BLOCK_SIZE blocks
        int blocksize = REDUCTION_BLOCK_SIZE;
        int gridsize = (output_size < REDUCTION_BLOCK_SIZE?
                        output_size : REDUCTION_BLOCK_SIZE);

        // launch kernel
        gpu_reduction_op<<<gridsize, blocksize>>>(
            // inputs
            data, valid, size,
            // output
            output,
            // action
            functor,
            // identity
            identity,
            // loader
            loader
        );
    }

};


struct DeviceSum {
    typedef IdentityLoader Loader;
    typedef DeviceSum second;

    template<typename T>
    __device__
    T operator() (const T &lhs, const T &rhs) {
        return lhs + rhs;
    }
};

struct DeviceProduct {
    typedef IdentityLoader Loader;
    typedef DeviceProduct second;

    template<typename T>
    __device__
    T operator() (const T &lhs, const T &rhs) {
        return lhs * rhs;
    }
};


struct DeviceSumSquared {
    struct Loader {
        template <typename T>
        __device__
        T operator() (const T* ptr, int pos) const {
            T val = ptr[pos];   // load
            return val * val;   // squared
        }
    };
    // round 2 just uses the basic sum reduction
    typedef DeviceSum second;

    template<typename T>
    __device__
    T operator() (const T &lhs, const T &rhs) const {
        return lhs + rhs;
    }
};


struct DeviceMin {
    typedef IdentityLoader Loader;
    typedef DeviceMin second;

    template<typename T>
    __device__
    T operator() (const T &lhs, const T &rhs) {
        return lhs <= rhs? lhs: rhs;
    }
};


struct DeviceMax {
    typedef IdentityLoader Loader;
    typedef DeviceMax second;

    template<typename T>
    __device__
    T operator() (const T &lhs, const T &rhs) {
        return lhs >= rhs? lhs: rhs;
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
    case GDF_INT8:    return F##_i8(col,  (int8_t*)dev_result, dev_result_size);  \
    default:          return GDF_UNSUPPORTED_DTYPE;                               \
    }                                                                             \
}

#define DEF_REDUCE_OP_REAL(F)                                                     \
gdf_error F##_generic(gdf_column *col, void *dev_result,                          \
                          gdf_size_type dev_result_size) {                        \
    switch ( col->dtype ) {                                                       \
    case GDF_FLOAT64: return F##_f64(col, (double*)dev_result, dev_result_size);  \
    case GDF_FLOAT32: return F##_f32(col, (float*)dev_result, dev_result_size);   \
    default:          return GDF_UNSUPPORTED_DTYPE;                               \
    }                                                                             \
}

#define DEF_REDUCE_IMPL(F, OP, T, ID)                                         \
gdf_error F(gdf_column *col, T *dev_result, gdf_size_type dev_result_size) {  \
    return ReduceOp<T, OP>::launch(col, ID, dev_result, dev_result_size);     \
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
DEF_REDUCE_IMPL(gdf_sum_i8,  DeviceSum, int8_t, 0)

/* Product */

DEF_REDUCE_OP_NUM(gdf_product)
DEF_REDUCE_IMPL(gdf_product_f64, DeviceProduct, double, 1)
DEF_REDUCE_IMPL(gdf_product_f32, DeviceProduct, float, 1)
DEF_REDUCE_IMPL(gdf_product_i64, DeviceProduct, int64_t, 1)
DEF_REDUCE_IMPL(gdf_product_i32, DeviceProduct, int32_t, 1)
DEF_REDUCE_IMPL(gdf_product_i8,  DeviceProduct, int8_t, 1)

/* Sum Squared */

DEF_REDUCE_OP_REAL(gdf_sum_squared)
DEF_REDUCE_IMPL(gdf_sum_squared_f64, DeviceSumSquared, double, 0)
DEF_REDUCE_IMPL(gdf_sum_squared_f32, DeviceSumSquared, float, 0)

/* Min */

DEF_REDUCE_OP_NUM(gdf_min)
DEF_REDUCE_IMPL(gdf_min_f64, DeviceMin, double, std::numeric_limits<double>::max())
DEF_REDUCE_IMPL(gdf_min_f32, DeviceMin, float, std::numeric_limits<float>::max())
DEF_REDUCE_IMPL(gdf_min_i64, DeviceMin, int64_t, std::numeric_limits<int64_t>::max())
DEF_REDUCE_IMPL(gdf_min_i32, DeviceMin, int32_t, std::numeric_limits<int32_t>::max())
DEF_REDUCE_IMPL(gdf_min_i8, DeviceMin, int8_t, std::numeric_limits<int8_t>::max())

/* Max */

DEF_REDUCE_OP_NUM(gdf_max)
DEF_REDUCE_IMPL(gdf_max_f64, DeviceMax, double, std::numeric_limits<double>::lowest())
DEF_REDUCE_IMPL(gdf_max_f32, DeviceMax, float, std::numeric_limits<float>::lowest())
DEF_REDUCE_IMPL(gdf_max_i64, DeviceMax, int64_t, std::numeric_limits<int64_t>::lowest())
DEF_REDUCE_IMPL(gdf_max_i32, DeviceMax, int32_t, std::numeric_limits<int32_t>::lowest())
DEF_REDUCE_IMPL(gdf_max_i8, DeviceMax, int8_t,  std::numeric_limits<int8_t>::lowest())
