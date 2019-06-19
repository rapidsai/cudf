/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CUDF_REDUCTION_DISPATCHER_MULTISTEP_CUH
#define CUDF_REDUCTION_DISPATCHER_MULTISTEP_CUH

#include "reduction_functions.cuh"

namespace cudf {
namespace reductions {

// Reduction for mean, var, std
// It requires extra step after single step reduction call
template<typename T_in, typename T_out, typename Op, bool has_nulls>
void ReduceMultiStepOp(const gdf_column *input,
                   gdf_scalar* scalar, gdf_size_type ddof, cudaStream_t stream)
{
    gdf_size_type valid_count = input->size - input->null_count;

    T_out identity = Op::Op::template identity<T_out>();
    using intermediateOp = typename Op::template Intermediate<T_out>;
    // Itype: intermediate structure, output type of `reduction_op` and
    // input type of `intermediateOp::ComputeResult`
    using Itype = typename intermediateOp::IType;
    Itype intermediate{0};

    // allocate temporary memory for the dev_result
    void *dev_result = NULL;
    RMM_TRY(RMM_ALLOC(&dev_result, sizeof(Itype), stream));

    // initialize output by identity value
    CUDA_TRY(cudaMemcpyAsync(dev_result, &intermediate,
            sizeof(Itype), cudaMemcpyHostToDevice, stream));
    CHECK_STREAM(stream);

    // reduction by iterator
    auto it = intermediateOp::template make_iterator<has_nulls, T_in, T_out>(input, identity);
    reduction_op(static_cast<Itype*>(dev_result), it, input->size, intermediate,
        typename Op::Op{}, stream);

    // read back the dev_result to host memory
    // TODO: asynchronous copy
    CUDA_TRY(cudaMemcpy(&intermediate, dev_result,
            sizeof(Itype), cudaMemcpyDeviceToHost));

    // compute the dev_result value from intermediate value.
    T_out hos_result = intermediateOp::ComputeResult(intermediate, valid_count, ddof);
    memcpy(&scalar->data, &hos_result, sizeof(T_out));

    // cleanup temporary memory
    RMM_TRY(RMM_FREE(dev_result, stream));

    // set scalar is valid
    scalar->is_valid = true;
};


template <typename T_in, typename Op>
struct ReduceMultiStepOutputDispatcher {
private:
    template <typename T_out>
    static constexpr bool is_supported_v()
    {
        // the operator `mean`, `var`, `std` only accepts
        // floating points as output dtype
        return  std::is_floating_point<T_out>::value;
    }

public:
    template <typename T_out, typename std::enable_if<
        is_supported_v<T_out>() >::type* = nullptr>
    void operator()(const gdf_column *col,
                         gdf_scalar* scalar, gdf_size_type ddof, cudaStream_t stream)
    {
        if( col->valid == nullptr ){
            ReduceMultiStepOp<T_in, T_out, Op, false>(col, scalar, ddof, stream);
        }else{
            ReduceMultiStepOp<T_in, T_out, Op, true >(col, scalar, ddof, stream);
        }
    }

    template <typename T_out, typename std::enable_if<
        not is_supported_v<T_out>() >::type* = nullptr >
    void operator()(const gdf_column *col,
                         gdf_scalar* scalar, gdf_size_type ddof, cudaStream_t stream)
    {
        CUDF_FAIL("Unsupported output data type");
    }
};

template <typename Op>
struct ReduceMultiStepDispatcher {
private:
    // return true if T is arithmetic type or cudf::bool8
    template <typename T>
    static constexpr bool is_supported()
    {
        return std::is_arithmetic<T>::value ||
               std::is_same<T, cudf::bool8>::value;
    }

public:
    template <typename T, typename std::enable_if<
        is_supported<T>()>::type* = nullptr>
    void operator()(const gdf_column *col,
                         gdf_scalar* scalar, gdf_size_type ddof, cudaStream_t stream=0)
    {
        cudf::type_dispatcher(scalar->dtype,
            ReduceMultiStepOutputDispatcher<T, Op>(), col, scalar, ddof, stream);
    }

    template <typename T, typename std::enable_if<
        not is_supported<T>()>::type* = nullptr>
    void operator()(const gdf_column *col,
                         gdf_scalar* scalar, gdf_size_type ddof, cudaStream_t stream=0)
    {
        CUDF_FAIL("Reduction operators other than `min` and `max`"
                  " are not supported for non-arithmetic types");
    }
};

} // namespace reductions
} // namespace cudf
#endif