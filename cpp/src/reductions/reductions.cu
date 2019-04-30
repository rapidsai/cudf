#include "cudf.h"
#include "rmm/rmm.h"
#include "utilities/cudf_utils.h"
#include "utilities/error_utils.hpp"
#include "utilities/type_dispatcher.hpp"
#include "bitmask/legacy_bitmask.hpp"

#include <cub/block/block_reduce.cuh>

#include <limits>
#include <type_traits>

#include <reduction.hpp>
#include "reduction_operators.cuh"

namespace { // anonymous namespace

static constexpr int reduction_block_size = 128;

/*
Generic reduction implementation with support for validity mask
*/
template<typename T_in, typename T_out, typename F, typename Ld>
__global__
void gpu_reduction_op(const T_in *data, const gdf_valid_type *mask,
                      gdf_size_type size, T_out *result,
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
            loaded = static_cast<T_out>(loader(data, i));

        // Block reduce
        T_out temp = BlockReduce(temp_storage).Reduce(loaded, functor);
        // Add current block
        agg = functor(agg, temp);
    }

    // First thread of each block stores the result.
    if (tid == 0){
        cudf::genericAtomicOperation(result, agg, functor);
    }
}

template<typename T_in, typename T_out, typename Op>
void ReduceOp(const gdf_column *input,
                   gdf_scalar* scalar, cudaStream_t stream)
{
    T_out identity = Op::Op::template identity<T_out>();

    // allocate temporary memory for the result
    void *result = NULL;
    RMM_TRY(RMM_ALLOC(&result, sizeof(T_out), stream));

    // initialize output by identity value
    CUDA_TRY(cudaMemcpyAsync(result, &identity,
            sizeof(T_out), cudaMemcpyHostToDevice, stream));
    CHECK_STREAM(stream);

    int blocksize = reduction_block_size;
    int gridsize = (input->size + reduction_block_size -1 )
        /reduction_block_size;

    // kernel call
    gpu_reduction_op<<<gridsize, blocksize, 0, stream>>>(
        static_cast<const T_in*>(input->data), input->valid, input->size,
        static_cast<T_out*>(result),
        typename Op::Op{}, identity, typename Op::Loader{});
    CHECK_STREAM(stream);

    // read back the result to host memory
    // TODO: asynchronous copy
    CUDA_TRY(cudaMemcpy(&scalar->data, result,
            sizeof(T_out), cudaMemcpyDeviceToHost));

    // cleanup temporary memory
    RMM_TRY(RMM_FREE(result, stream));

    // set scalar is valid
    scalar->is_valid = true;
};

template <typename T_in, typename Op>
struct ReduceOutputDispatcher {
public:
    template <typename T_out, typename std::enable_if<
        std::is_constructible<T_out, T_in>::value >::type* = nullptr>
    void operator()(const gdf_column *col,
                         gdf_scalar* scalar, cudaStream_t stream)
    {
        ReduceOp<T_in, T_out, Op>(col, scalar, stream);
    }

    template <typename T_out, typename std::enable_if<
        not std::is_constructible<T_out, T_in>::value >::type* = nullptr >
    void operator()(const gdf_column *col,
                         gdf_scalar* scalar, cudaStream_t stream)
    {
        CUDF_FAIL("input data type is not convertible to output data type");
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
               std::is_same<T, cudf::bool8>::value ||
               std::is_same<Op, cudf::reductions::ReductionMin>::value ||
               std::is_same<Op, cudf::reductions::ReductionMax>::value ;
    }

public:
    template <typename T, typename std::enable_if<
        is_supported<T>()>::type* = nullptr>
    void operator()(const gdf_column *col,
                         gdf_scalar* scalar, cudaStream_t stream=0)
    {
        cudf::type_dispatcher(scalar->dtype,
            ReduceOutputDispatcher<T, Op>(), col, scalar, stream);
    }

    template <typename T, typename std::enable_if<
        not is_supported<T>()>::type* = nullptr>
    void operator()(const gdf_column *col,
                         gdf_scalar* scalar, cudaStream_t stream=0)
    {
        CUDF_FAIL("Reduction operators other than `min` and `max`"
                  " are not supported for non-arithmetic types");
    }
};

}   // anonymous namespace


namespace cudf{


gdf_scalar reduction(const gdf_column *col,
                  gdf_reduction_op op, gdf_dtype output_dtype)
{
    gdf_scalar scalar;
    scalar.dtype = output_dtype;
    scalar.is_valid = false; // the scalar is not valid for error case

    CUDF_EXPECTS(col != nullptr, "Input column is null");
    // check if input column is empty
    if( col->size <= col->null_count )return scalar;

    switch(op){
    case GDF_REDUCTION_SUM:
        cudf::type_dispatcher(col->dtype,
            ReduceDispatcher<cudf::reductions::ReductionSum>(), col, &scalar);
        break;
    case GDF_REDUCTION_MIN:
        cudf::type_dispatcher(col->dtype,
            ReduceDispatcher<cudf::reductions::ReductionMin>(), col, &scalar);
        break;
    case GDF_REDUCTION_MAX:
        cudf::type_dispatcher(col->dtype,
            ReduceDispatcher<cudf::reductions::ReductionMax>(), col, &scalar);
        break;
    case GDF_REDUCTION_PRODUCT:
        cudf::type_dispatcher(col->dtype,
            ReduceDispatcher<cudf::reductions::ReductionProduct>(), col, &scalar);
        break;
    case GDF_REDUCTION_SUMOFSQUARES:
        cudf::type_dispatcher(col->dtype,
            ReduceDispatcher<cudf::reductions::ReductionSumOfSquares>(), col, &scalar);
        break;
    default:
        CUDF_FAIL("Unsupported reduction operator");
    }

    return scalar;
}

}   // cudf namespace

