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
template<typename T_in, typename T_out, typename F, typename Ld>
__global__
void gpu_reduction_op(const T_in *data, const gdf_valid_type *mask,
                      gdf_size_type size, T_out *results, F functor, T_out identity,
                      Ld loader)
{
    typedef cub::BlockReduce<T_out, REDUCTION_BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int tid = threadIdx.x;
    int blkid = blockIdx.x;
    int blksz = blockDim.x;
    int gridsz = gridDim.x;

    int step = blksz * gridsz;

    T_out agg = identity;

    for (int base=blkid * blksz; base<size; base+=step) {
        // Threadblock synchronous loop
        int i = base + tid;
        // load
        T_out loaded = identity;
        if (i < size && gdf_is_valid(mask, i))
            loaded = loader(data, i);

        // Block reduce
        T_out temp = BlockReduce(temp_storage).Reduce(loaded, functor);
        // Add current block
        agg = functor(agg, temp);
    }

    // First thread of each block stores the result.
    if (tid == 0){
        genericAtomicOperation(results[0], agg, functor);
    }
}

template<typename T_in, typename T_out, typename Op>
gdf_error ReduceOp(const gdf_column *input, T_out *output)
{
    T_out identity = Op::template identity<T_out>();

    // initialize output by identity value
    CUDA_TRY(cudaMemcpyAsync(output, &identity,
            sizeof(T_out), cudaMemcpyHostToDevice, 0));
    CUDA_CHECK_LAST();

    int blocksize = REDUCTION_BLOCK_SIZE;
    int gridsize = (input->size + REDUCTION_BLOCK_SIZE -1 )
        /REDUCTION_BLOCK_SIZE;

    typename Op::Loader loader;
    Op functor;

    gpu_reduction_op<<<gridsize, blocksize>>>(
        & cudf::detail::unwrap(* ((const T_in*)input->data )),
        input->valid, input->size,
        & cudf::detail::unwrap(*output),
        functor, cudf::detail::unwrap(identity), loader
    );

    CUDA_CHECK_LAST();

    return GDF_SUCCESS;
};


template <typename T_in, typename Op>
struct ReduceOutputDispatcher {
private:
    // return true if both are same type (e.g. date, timestamp...)
    // or both are arithmetic types.
    template <typename T, typename U>
    static constexpr bool is_convertable()
    {
        return std::is_same<T, U>::value ||
            ( std::is_arithmetic<T>::value && std::is_arithmetic<U>::value );
    }

public:
    template <typename T_out, typename std::enable_if<
                is_convertable<T_in, T_out>() >::type* = nullptr >
    gdf_error operator()(const gdf_column *col, void *dev_result)
    {
        return ReduceOp<T_in, T_out, Op>
            (col, static_cast<T_out*>(dev_result));
    }

    template <typename T_out, typename std::enable_if<
                !is_convertable<T_in, T_out>() >::type* = nullptr >
    gdf_error operator()(const gdf_column *col, void *dev_result)
    {
        return GDF_UNSUPPORTED_DTYPE;
    }
};

template <typename Op>
struct ReduceDispatcher {
private:
    // return true if T is arithmetic type
    // or if Op is DeviceMin or DeviceMax for non-arithmetic types
    template <typename T>
    static constexpr bool is_supported()
    {
        return std::is_arithmetic<T>::value ||
               std::is_same<Op, DeviceMin>::value ||
               std::is_same<Op, DeviceMax>::value;
    }

public:
    template <typename T, typename std::enable_if<is_supported<T>()>::type* = nullptr>
    gdf_error operator()(const gdf_column *col, void *dev_result, gdf_dtype output_dtype)
    {
        GDF_REQUIRE(col->size > col->null_count, GDF_DATASET_EMPTY);
        return cudf::type_dispatcher(output_dtype,
            ReduceOutputDispatcher<T, Op>(), col, dev_result);
    }

    template <typename T, typename std::enable_if<!is_supported<T>()>::type* = nullptr>
    gdf_error operator()(const gdf_column *col, void *dev_result, gdf_dtype output_dtype)
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


gdf_error gdf_reduction(const gdf_column *col,
                  gdf_reduction_op op,
                  void *dev_result, gdf_dtype output_dtype)
{
    switch(op){
    case GDF_REDUCTION_SUM:
        return cudf::type_dispatcher(col->dtype,
            ReduceDispatcher<DeviceSum>(), col, dev_result, output_dtype);
    case GDF_REDUCTION_MIN:
        return cudf::type_dispatcher(col->dtype,
            ReduceDispatcher<DeviceMin>(), col, dev_result, output_dtype);
    case GDF_REDUCTION_MAX:
        return cudf::type_dispatcher(col->dtype,
            ReduceDispatcher<DeviceMax>(), col, dev_result, output_dtype);
    case GDF_REDUCTION_PRODUCTION:
        return cudf::type_dispatcher(col->dtype,
            ReduceDispatcher<DeviceProduct>(), col, dev_result, output_dtype);
    case GDF_REDUCTION_SUMOFSQUARES:
        return cudf::type_dispatcher(col->dtype,
            ReduceDispatcher<DeviceSumOfSquares>(),
            col, dev_result, output_dtype);
    default:
        return GDF_INVALID_API_CALL;
    }
}

gdf_error gdf_sum(gdf_column *col,
                  void *dev_result,
                  gdf_size_type dev_result_size)
{
    return gdf_reduction(col, GDF_REDUCTION_SUM, dev_result, col->dtype);
}

gdf_error gdf_product(gdf_column *col,
                      void *dev_result,
                      gdf_size_type dev_result_size)
{
    return gdf_reduction(col, GDF_REDUCTION_PRODUCTION, dev_result, col->dtype);
}

gdf_error gdf_sum_of_squares(gdf_column *col,
                             void *dev_result,
                             gdf_size_type dev_result_size)
{
    return gdf_reduction(col, GDF_REDUCTION_SUMOFSQUARES, dev_result, col->dtype);
}

gdf_error gdf_min(gdf_column *col,
                  void *dev_result,
                  gdf_size_type dev_result_size)
{
    return gdf_reduction(col, GDF_REDUCTION_MIN, dev_result, col->dtype);
}

gdf_error gdf_max(gdf_column *col,
                  void *dev_result,
                  gdf_size_type dev_result_size)
{
    return gdf_reduction(col, GDF_REDUCTION_MAX, dev_result, col->dtype);
}


unsigned int gdf_reduction_get_intermediate_output_size() {
    return REDUCTION_BLOCK_SIZE;
}
