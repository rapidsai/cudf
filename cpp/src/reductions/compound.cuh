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

#pragma once 

#include "reduction.cuh"
#include "reduction_operators.cuh"

#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/utilities/legacy/type_dispatcher.hpp> //TODO remove gdf_dtype_of after cudf::scalar support

namespace cudf {
namespace experimental {
namespace reduction {
namespace compound {

/** --------------------------------------------------------------------------*
 * @brief Reduction for mean, var, std
 * It requires extra step after single step reduction call
 *
 * @param[in] col    input column view
 * @param[out] scalar  output scalar data
 * @param[in] ddof   `Delta Degrees of Freedom` used for `std`, `var`.
 *                   The divisor used in calculations is N - ddof, where N represents the number of elements.
 * @param[in] stream cuda stream
 *
 * @tparam ElementType  the input column cudf dtype
 * @tparam ResultType   the output cudf dtype
 * @tparam Op           the operator of cudf::experimental::reduction::op::
 * @tparam has_nulls    true if column has nulls
 * ----------------------------------------------------------------------------**/
template <typename ElementType, typename ResultType, typename Op, bool has_nulls>
gdf_scalar compound_reduction(column_view const& col,
                              data_type const output_dtype,
                              cudf::size_type ddof, cudaStream_t stream) {
  gdf_scalar scalar;
  scalar.dtype = gdf_dtype_of<ResultType>();
  // TODO: after cudf::scalar support
  // scalar.dtype = output_dtype;
  scalar.is_valid = false;  // the scalar is not valid for error case
  cudf::size_type valid_count = col.size() - col.null_count();

  using intermediateOp = typename Op::template intermediate<ResultType>;
  // IntermediateType: intermediate structure, output type of `reduction_op` and
  // input type of `intermediateOp::ComputeResult`
  using IntermediateType = typename intermediateOp::IntermediateType;
  IntermediateType intermediate{0};

  // allocate temporary memory for the dev_result
  void* dev_result = NULL;
  RMM_TRY(RMM_ALLOC(&dev_result, sizeof(IntermediateType), stream));

  // initialize output by identity value
  CUDA_TRY(cudaMemcpyAsync(dev_result, &intermediate, sizeof(IntermediateType),
                           cudaMemcpyHostToDevice, stream));
  CHECK_STREAM(stream);

  // reduction by iterator
  auto dcol = cudf::column_device_view::create(col, stream);
  if (col.nullable()) {
    auto it = thrust::make_transform_iterator(
        dcol->nbegin(Op::Op::template identity<ElementType>()),
        typename Op::template transformer<ResultType>{});

    detail::reduce(static_cast<IntermediateType*>(dev_result), it, col.size(),
                   intermediate, typename Op::Op{}, stream);
  } else {
    auto it = thrust::make_transform_iterator(
        dcol->begin<ElementType>(),
        typename Op::template transformer<ResultType>{});

    detail::reduce(static_cast<IntermediateType*>(dev_result), it, col.size(),
                   intermediate, typename Op::Op{}, stream);
  }

  // read back the dev_result to host memory
  // TODO: asynchronous copy
  CUDA_TRY(cudaMemcpy(&intermediate, dev_result, sizeof(IntermediateType),
                      cudaMemcpyDeviceToHost));

  // compute the dev_result value from intermediate value.
  ResultType hos_result = intermediateOp::compute_result(intermediate, valid_count, ddof);
  memcpy(&scalar.data, &hos_result, sizeof(ResultType));

  // cleanup temporary memory
  RMM_TRY(RMM_FREE(dev_result, stream));

  // set scalar is valid
  if (col.null_count() < col.size()) scalar.is_valid = true;
  return scalar;
};

// @brief result type dispatcher for compound reduction (a.k.a. mean, var, std)
template <typename ElementType, typename Op>
struct result_type_dispatcher {
private:
    template <typename ResultType>
    static constexpr bool is_supported_v()
    {
        // the operator `mean`, `var`, `std` only accepts
        // floating points as output dtype
        return  std::is_floating_point<ResultType>::value;
    }

public:
    template <typename ResultType, std::enable_if_t<is_supported_v<ResultType>()>* = nullptr>
    gdf_scalar operator()(column_view const& col, cudf::data_type const output_dtype, cudf::size_type ddof, cudaStream_t stream)
    {
        if(col.has_nulls()) {
            return compound_reduction<ElementType, ResultType, Op, true >(col, output_dtype, ddof, stream);
        } else {
            return compound_reduction<ElementType, ResultType, Op, false>(col, output_dtype, ddof, stream);
        }
    }

    template <typename ResultType, std::enable_if_t<not is_supported_v<ResultType>()>* = nullptr >
    gdf_scalar operator()(column_view const& col, cudf::data_type const output_dtype, cudf::size_type ddof, cudaStream_t stream)
    {
        CUDF_FAIL("Unsupported output data type");
    }
};

// @brief input column element dispatcher for compound reduction (a.k.a. mean, var, std)
template <typename Op>
struct element_type_dispatcher {
private:
    // return true if ElementType is arithmetic type 
    template <typename ElementType>
    static constexpr bool is_supported_v()
    {
        return std::is_arithmetic<ElementType>::value;
    }

public:
    template <typename ElementType, std::enable_if_t<is_supported_v<ElementType>()>* = nullptr>
    gdf_scalar operator()(column_view const& col, cudf::data_type const output_dtype, cudf::size_type ddof, cudaStream_t stream)
    {
        return cudf::experimental::type_dispatcher(output_dtype,
            result_type_dispatcher<ElementType, Op>(), col, output_dtype, ddof, stream);
    }

    template <typename ElementType, std::enable_if_t<not is_supported_v<ElementType>()>* = nullptr>
    gdf_scalar operator()(column_view const& col, cudf::data_type const output_dtype, cudf::size_type ddof, cudaStream_t stream)
    {
        CUDF_FAIL("Reduction operators other than `min` and `max`"
                  " are not supported for non-arithmetic types");
    }
};

} // namespace compound
} // namespace reduction
} // namespace experimental
} // namespace cudf

