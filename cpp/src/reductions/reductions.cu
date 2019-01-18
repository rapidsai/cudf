#include "cudf.h"
#include "utilities/cudf_utils.h"
#include "utilities/error_utils.h"
#include "utilities/type_dispatcher.hpp"

#include <cub/block/block_reduce.cuh>

#include <limits>
#include <type_traits>

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

    template<typename T>
    static constexpr T identity() { return T{0}; }
};

struct DeviceProduct {
    typedef IdentityLoader Loader;
    typedef DeviceProduct second;

    template<typename T>
    __device__
    T operator() (const T &lhs, const T &rhs) {
        return lhs * rhs;
    }

    template<typename T>
    static constexpr T identity() { return T{1}; }
};

struct DeviceSumOfSquares {
    struct Loader {
        template<typename T>
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

    template<typename T>
    static constexpr T identity() { return T{0}; }
};

struct DeviceMin {
    typedef IdentityLoader Loader;
    typedef DeviceMin second;

    template<typename T>
    __device__
    T operator() (const T &lhs, const T &rhs) {
        return lhs <= rhs? lhs: rhs;
    }

    template<typename T>
    static constexpr T identity() { return std::numeric_limits<T>::max(); }
};

struct DeviceMax {
    typedef IdentityLoader Loader;
    typedef DeviceMax second;

    template<typename T>
    __device__
    T operator() (const T &lhs, const T &rhs) {
        return lhs >= rhs? lhs: rhs;
    }

    template<typename T>
    static constexpr T identity() { return std::numeric_limits<T>::lowest(); }
};

template <typename Op>
struct ReduceDispatcher {
    template <typename T,
              typename std::enable_if_t<std::is_arithmetic<T>::value>* = nullptr>
    gdf_error operator()(gdf_column *col, 
                         void *dev_result, 
                         gdf_size_type dev_result_size) {
        GDF_REQUIRE(col->size > col->null_count, GDF_DATASET_EMPTY);
        T identity = Op::template identity<T>();
        return ReduceOp<T, Op>::launch(col, identity, 
                                       reinterpret_cast<T*>(dev_result), 
                                       dev_result_size); 
    }

    template <typename T,
              typename std::enable_if_t<!std::is_arithmetic<T>::value, T>* = nullptr>
    gdf_error operator()(gdf_column *col, 
                         void *dev_result, 
                         gdf_size_type dev_result_size) {
        return GDF_UNSUPPORTED_DTYPE;
    }
};


gdf_error gdf_sum(gdf_column *col,
                  void *dev_result,
                  gdf_size_type dev_result_size)
{   
    return cudf::type_dispatcher(col->dtype, ReduceDispatcher<DeviceSum>(),
                                 col, dev_result, dev_result_size);
}

gdf_error gdf_product(gdf_column *col,
                      void *dev_result,
                      gdf_size_type dev_result_size)
{
    return cudf::type_dispatcher(col->dtype, ReduceDispatcher<DeviceProduct>(),
                                 col, dev_result, dev_result_size);
}

gdf_error gdf_sum_of_squares(gdf_column *col,
                             void *dev_result,
                             gdf_size_type dev_result_size)
{
    return cudf::type_dispatcher(col->dtype, ReduceDispatcher<DeviceSumOfSquares>(),
                                 col, dev_result, dev_result_size);
}

gdf_error gdf_min(gdf_column *col,
                  void *dev_result,
                  gdf_size_type dev_result_size)
{
    return cudf::type_dispatcher(col->dtype, ReduceDispatcher<DeviceMin>(),
                                 col, dev_result, dev_result_size);
}

gdf_error gdf_max(gdf_column *col,
                  void *dev_result,
                  gdf_size_type dev_result_size)
{
    return cudf::type_dispatcher(col->dtype, ReduceDispatcher<DeviceMax>(),
                                 col, dev_result, dev_result_size);
}


unsigned int gdf_reduce_optimal_output_size() {
    return REDUCTION_BLOCK_SIZE;
}
