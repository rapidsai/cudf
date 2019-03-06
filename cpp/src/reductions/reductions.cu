#include "cudf.h"
#include "utilities/cudf_utils.h"
#include "utilities/error_utils.h"
#include "utilities/type_dispatcher.hpp"

#include <cub/block/block_reduce.cuh>

#include <limits>
#include <type_traits>

#include "reduction_operators.cuh"

#define REDUCTION_BLOCK_SIZE 128

using namespace cudf::reduction;

namespace { // anonymous namespace

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
    if (tid == 0){
        genericAtomicOperation(results[0], agg, functor);
    }
}



template<typename T, typename Op>
gdf_error ReduceOp(gdf_column *input, T *output)
{
    T identity = Op::template identity<T>();

    // initialize output by identity value
    CUDA_TRY(cudaMemcpyAsync(output, &identity,
            sizeof(T), cudaMemcpyHostToDevice, 0));
    CUDA_CHECK_LAST();

    int blocksize = REDUCTION_BLOCK_SIZE;
    int gridsize = (input->size + REDUCTION_BLOCK_SIZE -1 )
        /REDUCTION_BLOCK_SIZE;

    typename Op::Loader loader;
    Op functor;

    // launch kernel
    gpu_reduction_op<<<gridsize, blocksize>>>(
        (const T*)input->data, input->valid, input->size, (T*)output,
        functor, identity, loader
    );

    CUDA_CHECK_LAST();

    return GDF_SUCCESS;
};


template <typename Op>
struct ReduceDispatcher {
    static constexpr bool is_nonarithmetic_op =
        std::is_same<Op, DeviceMin>::value ||
        std::is_same<Op, DeviceMax>::value ;

    template <typename T>
    gdf_error launch(gdf_column *col, void *dev_result)
    {
        GDF_REQUIRE(col->size > col->null_count, GDF_DATASET_EMPTY);
        return ReduceOp<T, Op>(col, static_cast<T*>(dev_result));
    }

    template <typename T, typename std::enable_if<
                std::is_arithmetic<T>::value>::type* = nullptr>
    gdf_error operator()(gdf_column *col, void *dev_result)
    {
        return launch<T>(col, dev_result);
    }

    template <typename T, typename std::enable_if<
                  !std::is_arithmetic<T>::value && is_nonarithmetic_op
              >::type* = nullptr>
    gdf_error operator()(gdf_column *col, void *dev_result)
    {
        using UnderlyingType = typename T::value_type;
        return launch<UnderlyingType>(col, dev_result);
    }

    template <typename T, typename std::enable_if<
                  !std::is_arithmetic<T>::value && !is_nonarithmetic_op
              >::type* = nullptr>
    gdf_error operator()(gdf_column *col, void *dev_result)
    {
        return GDF_UNSUPPORTED_DTYPE;
    }
};


}   // anonymous namespace

typedef enum {
  GDF_REDUCTION_SUM = 0,
  GDF_REDUCTION_MIN,
  GDF_REDUCTION_MAX,
  GDF_REDUCTION_PRODUCTION,
  GDF_REDUCTION_SUMOFSQUARES,
} gdf_reduction_op;


gdf_error gdf_reduction(gdf_column *col,
                  gdf_reduction_op op,
                  void *dev_result)
{
    switch(op){
    case GDF_REDUCTION_SUM:
        return cudf::type_dispatcher(col->dtype,
            ReduceDispatcher<DeviceSum>(), col, dev_result);
    case GDF_REDUCTION_MIN:
        return cudf::type_dispatcher(col->dtype,
            ReduceDispatcher<DeviceMin>(), col, dev_result);
    case GDF_REDUCTION_MAX:
        return cudf::type_dispatcher(col->dtype,
            ReduceDispatcher<DeviceMax>(), col, dev_result);
    case GDF_REDUCTION_PRODUCTION:
        return cudf::type_dispatcher(col->dtype,
            ReduceDispatcher<DeviceProduct>(), col, dev_result);
    case GDF_REDUCTION_SUMOFSQUARES:
        return cudf::type_dispatcher(col->dtype,
            ReduceDispatcher<DeviceSumOfSquares>(), col, dev_result);
    default:
        return GDF_INVALID_API_CALL;
    }
}

// ToDo: remove these APIs
gdf_error gdf_sum(gdf_column *col,
                  void *dev_result,
                  gdf_size_type dev_result_size)
{
    return gdf_reduction(col, GDF_REDUCTION_SUM, dev_result);
}

gdf_error gdf_product(gdf_column *col,
                      void *dev_result,
                      gdf_size_type dev_result_size)
{
    return gdf_reduction(col, GDF_REDUCTION_PRODUCTION, dev_result);
}

gdf_error gdf_sum_of_squares(gdf_column *col,
                             void *dev_result,
                             gdf_size_type dev_result_size)
{
    return gdf_reduction(col, GDF_REDUCTION_SUMOFSQUARES, dev_result);
}

gdf_error gdf_min(gdf_column *col,
                  void *dev_result,
                  gdf_size_type dev_result_size)
{
    return gdf_reduction(col, GDF_REDUCTION_MIN, dev_result);
}

gdf_error gdf_max(gdf_column *col,
                  void *dev_result,
                  gdf_size_type dev_result_size)
{
    return gdf_reduction(col, GDF_REDUCTION_MAX, dev_result);
}


unsigned int gdf_reduction_get_intermediate_output_size() {
    return REDUCTION_BLOCK_SIZE;
}
