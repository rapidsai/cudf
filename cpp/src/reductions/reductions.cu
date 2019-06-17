#include <cudf/cudf.h>
#include <rmm/rmm.h>
#include <utilities/cudf_utils.h>
#include <utilities/error_utils.hpp>
#include <utilities/type_dispatcher.hpp>
#include <bitmask/legacy_bitmask.hpp>
#include <iterator/iterator.cuh>

#include <cub/device/device_reduce.cuh>
#include <cub/block/block_reduce.cuh>
#include <thrust/reduce.h>

#include <limits>
#include <type_traits>

#include <cudf/reduction.hpp>
#include "reduction_operators.cuh"

namespace { // anonymous namespace

template <typename Op, typename InputIterator, typename T_output>
void reduction_op(T_output* dev_result, InputIterator d_in, gdf_size_type num_items,
    T_output init, Op op, cudaStream_t stream)
{
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;

    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, dev_result, num_items,
        op, init, stream);
    // Allocate temporary storage
    RMM_TRY(RMM_ALLOC(&d_temp_storage, temp_storage_bytes, stream));

    // Run reduction
    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, dev_result, num_items,
        op, init, stream);

    // Free temporary storage
    RMM_TRY(RMM_FREE(d_temp_storage, stream));
}

template<typename T_in, typename T_out, typename Op, bool has_nulls>
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

    if( std::is_same<Op, cudf::reductions::ReductionSumOfSquares>::value ){
        auto it_raw = cudf::make_iterator<has_nulls, T_in, T_out>(*input, identity);
        auto it = thrust::make_transform_iterator(it_raw, cudf::transformer_squared<T_out>{});
        reduction_op(static_cast<T_out*>(result), it, input->size, identity,
            typename Op::Op{}, stream);
    }else{
        auto it = cudf::make_iterator<has_nulls, T_in, T_out>(*input, identity);
        reduction_op(static_cast<T_out*>(result), it, input->size, identity,
            typename Op::Op{}, stream);
    }

    // read back the result to host memory
    // TODO: asynchronous copy
    CUDA_TRY(cudaMemcpy(&scalar->data, result,
            sizeof(T_out), cudaMemcpyDeviceToHost));

    // cleanup temporary memory
    RMM_TRY(RMM_FREE(result, stream));

    // set scalar is valid
    scalar->is_valid = true;
};

// Reduction for mean, var, std
// It requires extra step after single step reduction call
template<typename T_in, typename T_out, typename Op, bool has_nulls>
void ReduceMultiStepOp(const gdf_column *input,
                   gdf_scalar* scalar, cudaStream_t stream)
{
    gdf_size_type valid_count = input->size - input->null_count;

    T_out identity = Op::Op::template identity<T_out>();
    using intermediateOp = typename Op::template Intermediate<T_out>;
    using Itype = typename intermediateOp::IType;
    Itype intermediate{0};

    // allocate temporary memory for the result
    void *result = NULL;
    RMM_TRY(RMM_ALLOC(&result, sizeof(Itype), stream));

    // initialize output by identity value
    CUDA_TRY(cudaMemcpyAsync(result, &intermediate,
            sizeof(Itype), cudaMemcpyHostToDevice, stream));
    CHECK_STREAM(stream);

    auto transformer = intermediateOp::get_transformer();
    auto it_raw = cudf::make_iterator<has_nulls, T_in, T_out>(*input, identity);
    auto it = thrust::make_transform_iterator(it_raw, transformer);
    reduction_op(static_cast<Itype*>(result), it, input->size, intermediate,
        typename Op::Op{}, stream);


    // read back the result to host memory
    // TODO: asynchronous copy
    CUDA_TRY(cudaMemcpy(&intermediate, result,
            sizeof(Itype), cudaMemcpyDeviceToHost));

    // compute the result value from intermediate value.
    T_out hos_result = intermediateOp::ComputeResult(intermediate, valid_count);
    memcpy(&scalar->data, result, sizeof(T_out));

    // cleanup temporary memory
    RMM_TRY(RMM_FREE(result, stream));

    // set scalar is valid
    scalar->is_valid = true;
};

template <typename T_in, typename Op>
struct ReduceOutputDispatcher {
private:
    static constexpr bool is_multistep_reduction()
    {
        return  std::is_same<Op, cudf::reductions::ReductionMean>::value ||
                std::is_same<Op, cudf::reductions::ReductionVar >::value ||
                std::is_same<Op, cudf::reductions::ReductionStd >::value;
    }

    template <typename T_out>
    static constexpr bool is_convertible_v()
    {
        return  std::is_arithmetic<T_in>::value && std::is_arithmetic<T_out>::value;
    }

public:
    template <typename T_out, typename std::enable_if<
        is_convertible_v<T_out>() >::type* = nullptr>
    void operator()(const gdf_column *col,
                         gdf_scalar* scalar, cudaStream_t stream)
    {
        if( is_multistep_reduction() ){
            if( col->valid == nullptr ){
                ReduceMultiStepOp<T_in, T_out, Op, false>(col, scalar, stream);
            }else{
                ReduceMultiStepOp<T_in, T_out, Op, true >(col, scalar, stream);
            }
        }else{
            if( col->valid == nullptr ){
                ReduceOp<T_in, T_out, Op, false>(col, scalar, stream);
            }else{
                ReduceOp<T_in, T_out, Op, true >(col, scalar, stream);
            }
        }
    }

    template <typename T_out, typename std::enable_if<
        not is_convertible_v<T_out>() >::type* = nullptr >
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
#if 0
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
    case GDF_REDUCTION_MEAN:
        cudf::type_dispatcher(col->dtype,
            ReduceDispatcher<cudf::reductions::ReductionMean>(), col, &scalar);
        break;
    case GDF_REDUCTION_VAR:
        cudf::type_dispatcher(col->dtype,
            ReduceDispatcher<cudf::reductions::ReductionVar>(), col, &scalar);
        break;
    case GDF_REDUCTION_STD:
        cudf::type_dispatcher(col->dtype,
            ReduceDispatcher<cudf::reductions::ReductionStd>(), col, &scalar);
        break;
#else
    case GDF_REDUCTION_MEAN:
        cudf::type_dispatcher(col->dtype,
            ReduceDispatcher<cudf::reductions::ReductionMean>(), col, &scalar);
        break;
#endif
    default:
        CUDF_FAIL("Unsupported reduction operator");
    }

    return scalar;
}

}   // cudf namespace

