#include "cudf.h"
#include "rmm/rmm.h"
#include "utilities/cudf_utils.h"
#include "utilities/error_utils.hpp"
#include "utilities/type_dispatcher.hpp"

#include <cub/block/block_reduce.cuh>

#include <limits>
#include <type_traits>

#include "reduction_operators.cuh"

using namespace cudf::reduction;

namespace { // anonymous namespace

static constexpr int reduction_block_size = 128;

/*
Generic reduction implementation with support for validity mask
*/
template<typename T_in, typename T_out, typename F, typename Ld>
__global__
void gpu_reduction_op(const T_in *data, const gdf_valid_type *mask,
                      gdf_size_type size, T_out *results,
                      F functor, T_out identity, Ld loader)
{
    typedef cub::BlockReduce<T_out, reduction_block_size> BlockReduce;
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
gdf_error ReduceOp(const gdf_column *input,
                   gdf_scalar* scalar, cudaStream_t stream)
{
    T_out identity = Op::template identity<T_out>();

    // allocate temporary memory for the result
    void *result = NULL;
    RMM_TRY(RMM_ALLOC(&result, sizeof(T_out), stream));

    // initialize output by identity value
    CUDA_TRY(cudaMemcpyAsync(result, &identity,
            sizeof(T_out), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK_LAST();

    int blocksize = reduction_block_size;
    int gridsize = (input->size + reduction_block_size -1 )
        /reduction_block_size;

    // kernel call
    gpu_reduction_op<<<gridsize, blocksize, 0, stream>>>(
        (const T_in*)input->data, input->valid, input->size,
        static_cast<T_out*>(result),
        Op{}, identity, typename Op::Loader{});
    CUDA_CHECK_LAST();

    // read back the result to host memory
    CUDA_TRY(cudaMemcpyAsync(&scalar->data, result,
            sizeof(T_out), cudaMemcpyDeviceToHost, stream));

    // cleanup temporary memory
    RMM_TRY(RMM_FREE(result, stream));

    return GDF_SUCCESS;
};


template <typename T_in, typename Op>
struct ReduceOutputDispatcher {
public:
    template <typename T_out, typename std::enable_if<
                std::is_convertible<T_in, T_out>::value >::type* = nullptr >
    gdf_error operator()(const gdf_column *col,
                         gdf_scalar* scalar, cudaStream_t stream)
    {
        return ReduceOp<T_in, T_out, Op>(col, scalar, stream);
    }

    template <typename T_out, typename std::enable_if<
                ! std::is_convertible<T_in, T_out>::value >::type* = nullptr >
    gdf_error operator()(const gdf_column *col,
                         gdf_scalar* scalar, cudaStream_t stream)
    {
        return GDF_UNSUPPORTED_DTYPE;
    }
};

template <typename Op>
struct ReduceDispatcher {
private:
    // return true if T is arithmetic type or
    // Op is DeviceMin or DeviceMax for wrapper (non-arithmetic) types
    template <typename T>
    static constexpr bool is_supported()
    {
        return std::is_arithmetic<T>::value ||
               std::is_same<Op, DeviceMin>::value ||
               std::is_same<Op, DeviceMax>::value ;
    }

public:
    template <typename T, typename std::enable_if<
        is_supported<T>()>::type* = nullptr>
    gdf_error operator()(const gdf_column *col,
                         gdf_scalar* scalar, cudaStream_t stream=0)
    {
        GDF_REQUIRE(col->size > col->null_count, GDF_DATASET_EMPTY);
        return cudf::type_dispatcher(scalar->dtype,
            ReduceOutputDispatcher<T, Op>(), col, scalar, stream);
    }

    template <typename T, typename std::enable_if<
        !is_supported<T>()>::type* = nullptr>
    gdf_error operator()(const gdf_column *col,
                         gdf_scalar* scalar, cudaStream_t stream=0)
    {
        return GDF_UNSUPPORTED_DTYPE;
    }
};

}   // anonymous namespace


gdf_error gdf_reduction(const gdf_column *col,
                  gdf_reduction_op op,
                  gdf_scalar* output)
{
    switch(op){
    case GDF_REDUCTION_SUM:
        return cudf::type_dispatcher(col->dtype,
            ReduceDispatcher<DeviceSum>(), col, output);
    case GDF_REDUCTION_MIN:
        return cudf::type_dispatcher(col->dtype,
            ReduceDispatcher<DeviceMin>(), col, output);
    case GDF_REDUCTION_MAX:
        return cudf::type_dispatcher(col->dtype,
            ReduceDispatcher<DeviceMax>(), col, output);
    case GDF_REDUCTION_PRODUCTION:
        return cudf::type_dispatcher(col->dtype,
            ReduceDispatcher<DeviceProduct>(), col, output);
    case GDF_REDUCTION_SUMOFSQUARES:
        return cudf::type_dispatcher(col->dtype,
            ReduceDispatcher<DeviceSumOfSquares>(), col, output);
    default:
        return GDF_INVALID_API_CALL;
    }
}

gdf_error gdf_reduction_stub(gdf_column *col,
                  gdf_reduction_op op,
                  void *dev_result, gdf_dtype output_dtype)
{
    gdf_scalar scalar;
    scalar.dtype = output_dtype;

    gdf_error ret = gdf_reduction(col, op, &scalar);    // do reduction

    // back to device memory.
    CUDA_TRY(cudaMemcpyAsync(dev_result, &scalar.data,
            8, cudaMemcpyHostToDevice, 0));
    CUDA_CHECK_LAST();

    return ret;
}

gdf_error gdf_sum(gdf_column *col,
                  void *dev_result,
                  gdf_size_type dev_result_size)
{
    return gdf_reduction_stub(col, GDF_REDUCTION_SUM, dev_result, col->dtype);
}

gdf_error gdf_product(gdf_column *col,
                      void *dev_result,
                      gdf_size_type dev_result_size)
{
    return gdf_reduction_stub(col, GDF_REDUCTION_PRODUCTION, dev_result, col->dtype);
}

gdf_error gdf_sum_of_squares(gdf_column *col,
                             void *dev_result,
                             gdf_size_type dev_result_size)
{
    return gdf_reduction_stub(col, GDF_REDUCTION_SUMOFSQUARES, dev_result, col->dtype);
}

gdf_error gdf_min(gdf_column *col,
                  void *dev_result,
                  gdf_size_type dev_result_size)
{
    return gdf_reduction_stub(col, GDF_REDUCTION_MIN, dev_result, col->dtype);
}

gdf_error gdf_max(gdf_column *col,
                  void *dev_result,
                  gdf_size_type dev_result_size)
{
    return gdf_reduction_stub(col, GDF_REDUCTION_MAX, dev_result, col->dtype);
}


unsigned int gdf_reduction_get_intermediate_output_size() {
    return reduction_block_size;
}
