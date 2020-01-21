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

#ifndef CUDF_REDUCTION_DISPATCHER_CUH
#define CUDF_REDUCTION_DISPATCHER_CUH

#include "reduction_functions.cuh"

namespace cudf {
namespace reduction {
namespace simple {

/** --------------------------------------------------------------------------*    
 * @brief Reduction for 'sum', 'product', 'min', 'max', 'sum of squares'
 * which directly compute the reduction by a single step reduction call
 *
 * @param[in] col    input column
 * @param[out] scalar  output scalar data
 * @param[in] stream cuda stream
 *
 * @tparam ElementType  the input column cudf dtype
 * @tparam ResultType   the output cudf dtype
 * @tparam Op           the operator of cudf::reduction::op::
 * @tparam has_nulls    true if column has nulls
 * ----------------------------------------------------------------------------**/
template<typename ElementType, typename ResultType, typename Op, bool has_nulls>
gdf_scalar simple_reduction(gdf_column const& col, gdf_dtype const output_dtype, cudaStream_t stream)
{
    gdf_scalar scalar;
    scalar.dtype = output_dtype;
    scalar.is_valid = false; // the scalar is not valid for error case
    ResultType identity = Op::Op::template identity<ResultType>();

    //  allocate temporary memory for the result
    ResultType *result = NULL;
    RMM_TRY(RMM_ALLOC(&result, sizeof(ResultType), stream));

    // initialize output by identity value
    CUDA_TRY(cudaMemcpyAsync(result, &identity,
            sizeof(ResultType), cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(stream);

    // reduction by iterator
    auto it = Op::template make_iterator<has_nulls, ElementType, ResultType>(col);
    cudf::reduction::detail::reduce(result, it, col.size, identity,
        typename Op::Op{}, stream);

    // read back the result to host memory
    // TODO: asynchronous copy
    CUDA_TRY(cudaMemcpy(&scalar.data, result,
            sizeof(ResultType), cudaMemcpyDeviceToHost));

    // cleanup temporary memory
    RMM_TRY(RMM_FREE(result, stream));

    // set scalar is valid
    if (col.null_count < col.size)
      scalar.is_valid = true;
    return scalar;
};

// @brief result type dispatcher for simple reduction (a.k.a. sum, prod, min...)
template <typename ElementType, typename Op>
struct result_type_dispatcher {
private:
    template <typename ResultType>
    static constexpr bool is_supported_v()
    {
        // for single step reductions,
        // the available combination of input and output dtypes are
        //  - same dtypes (including cudf::wrappers)
        //  - any arithmetic dtype to any arithmetic dtype
        //  - cudf::bool8 to/from any arithmetic dtype
        return  std::is_convertible<ElementType, ResultType>::value ||
        ( std::is_arithmetic<ElementType >::value && std::is_same<ResultType, cudf::bool8>::value ) ||
        ( std::is_arithmetic<ResultType>::value && std::is_same<ElementType , cudf::bool8>::value );
    }

public:
    template <typename ResultType, std::enable_if_t<is_supported_v<ResultType>()>* = nullptr>
    gdf_scalar operator()(gdf_column const& col, gdf_dtype const output_dtype, cudaStream_t stream)
    {
        if( cudf::has_nulls(col) ){
          return simple_reduction<ElementType, ResultType, Op, true >(col, output_dtype, stream);
        }else{
          return simple_reduction<ElementType, ResultType, Op, false>(col, output_dtype, stream);
        }
    }

    template <typename ResultType, std::enable_if_t<not is_supported_v<ResultType>()>* = nullptr>
    gdf_scalar operator()(gdf_column const& col, gdf_dtype const output_dtype, cudaStream_t stream)
    {
        CUDF_FAIL("input data type is not convertible to output data type");
    }
};

// @brief input column element for simple reduction (a.k.a. sum, prod, min...)
template <typename Op>
struct element_type_dispatcher {
private:
    // return true if ElementType is arithmetic type or bool8, or
    // Op is DeviceMin or DeviceMax for wrapper (non-arithmetic) types
    template <typename ElementType>
    static constexpr bool is_supported_v()
    {
        return std::is_arithmetic<ElementType>::value ||
               std::is_same<ElementType, cudf::bool8>::value ||
               std::is_same<Op, cudf::reduction::op::min>::value ||
               std::is_same<Op, cudf::reduction::op::max>::value ;
    }

public:
    template <typename ElementType, std::enable_if_t<is_supported_v<ElementType>()>* = nullptr>
    gdf_scalar operator()(gdf_column const& col, gdf_dtype const output_dtype, cudaStream_t stream)
    {
        return cudf::type_dispatcher(output_dtype,
            result_type_dispatcher<ElementType, Op>(), col, output_dtype, stream);
    }

    template <typename ElementType, std::enable_if_t<not is_supported_v<ElementType>()>* = nullptr>
    gdf_scalar operator()(gdf_column const& col, gdf_dtype const output_dtype, cudaStream_t stream)
    {
        CUDF_FAIL("Reduction operators other than `min` and `max`"
                  " are not supported for non-arithmetic types");
    }
};

} // namespace simple
} // namespace reduction
} // namespace cudf
#endif

