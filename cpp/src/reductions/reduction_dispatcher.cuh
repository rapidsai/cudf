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
namespace reductions {

// make an column iterator
template<bool has_nulls, typename T_in, typename T_out, typename Op>
auto make_iterator(const gdf_column &input, T_out identity, Op op)
{
    return cudf::make_iterator<has_nulls, T_in, T_out>(input, identity);
}

// make an column iterator, specialized for `cudf::reductions::op::sum_of_squares`
template<bool has_nulls, typename T_in, typename T_out>
auto make_iterator(const gdf_column &input, T_out identity,
    cudf::reductions::op::sum_of_squares op)
{
    auto it_raw = cudf::make_iterator<has_nulls, T_in, T_out>(input, identity);
    return thrust::make_transform_iterator(it_raw,
        cudf::transformer_squared<T_out>{});
}

// Reduction for 'sum', 'product', 'min', 'max', 'sum of squares'
// which directly compute the reduction by a single step reduction call
template<typename T_in, typename T_out, typename Op, bool has_nulls>
void simple_reduction(gdf_column const& col, gdf_scalar& scalar, cudaStream_t stream)
{
    T_out identity = Op::Op::template identity<T_out>();

    // allocate temporary memory for the result
    void *result = NULL;
    RMM_TRY(RMM_ALLOC(&result, sizeof(T_out), stream));

    // initialize output by identity value
    CUDA_TRY(cudaMemcpyAsync(result, &identity,
            sizeof(T_out), cudaMemcpyHostToDevice, stream));
    CHECK_STREAM(stream);

    // reduction by iterator
    auto it = make_iterator<has_nulls, T_in, T_out>(col, identity, Op{});
    reduce(static_cast<T_out*>(result), it, col.size, identity,
        typename Op::Op{}, stream);

    // read back the result to host memory
    // TODO: asynchronous copy
    CUDA_TRY(cudaMemcpy(&scalar.data, result,
            sizeof(T_out), cudaMemcpyDeviceToHost));

    // cleanup temporary memory
    RMM_TRY(RMM_FREE(result, stream));

    // set scalar is valid
    scalar.is_valid = true;
};


template <typename T_in, typename Op>
struct simple_reduction_result_type_dispatcher {
private:
    template <typename T_out>
    static constexpr bool is_supported_v()
    {
        // for single step reductions,
        // the available combination of input and output dtypes are
        //  - same dtypes (including cudf::wrappers)
        //  - any arithmetic dtype to any arithmetic dtype
        //  - cudf::bool8 to/from any arithmetic dtype
        return  std::is_convertible<T_in, T_out>::value ||
        ( std::is_arithmetic<T_in >::value && std::is_same<T_out, cudf::bool8>::value ) ||
        ( std::is_arithmetic<T_out>::value && std::is_same<T_in , cudf::bool8>::value );
    }

public:
    template <typename T_out, std::enable_if_t<is_supported_v<T_out>()>* = nullptr>
    void operator()(gdf_column const& col, gdf_scalar& scalar, cudaStream_t stream)
    {
        if( cudf::has_nulls(col) ){
            simple_reduction<T_in, T_out, Op, false>(col, scalar, stream);
        }else{
            simple_reduction<T_in, T_out, Op, true >(col, scalar, stream);
        }
    }

    template <typename T_out, std::enable_if_t<not is_supported_v<T_out>()>* = nullptr>
    void operator()(gdf_column const& col, gdf_scalar& scalar, cudaStream_t stream)
    {
        CUDF_FAIL("input data type is not convertible to output data type");
    }
};

template <typename Op>
struct simple_reduction_dispatcher {
private:
    // return true if T is arithmetic type or
    // Op is DeviceMin or DeviceMax for wrapper (non-arithmetic) types
    template <typename T>
    static constexpr bool is_supported_v()
    {
        return std::is_arithmetic<T>::value ||
               std::is_same<T, cudf::bool8>::value ||
               std::is_same<Op, cudf::reductions::op::min>::value ||
               std::is_same<Op, cudf::reductions::op::max>::value ;
    }

public:
    template <typename T, std::enable_if_t<is_supported_v<T>()>* = nullptr>
    void operator()(gdf_column const& col, gdf_scalar& scalar, cudaStream_t stream)
    {
        cudf::type_dispatcher(scalar.dtype,
            simple_reduction_result_type_dispatcher<T, Op>(), col, scalar, stream);
    }

    template <typename T, std::enable_if_t<not is_supported_v<T>()>* = nullptr>
    void operator()(gdf_column const& col, gdf_scalar& scalar, cudaStream_t stream)
    {
        CUDF_FAIL("Reduction operators other than `min` and `max`"
                  " are not supported for non-arithmetic types");
    }
};

} // namespace reductions
} // namespace cudf
#endif