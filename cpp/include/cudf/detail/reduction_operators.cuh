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

#include <cudf/types.hpp>  //for CUDA_HOST_DEVICE_CALLABLE
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/transform_unary_functions.cuh>
#include <cudf/detail/utilities/device_operators.cuh>

#include <thrust/functional.h>
#include <cmath>

namespace cudf {
namespace experimental {
namespace reduction {

// intermediate data structure to compute `var`, `std`
template <typename ResultType>
struct var_std {
  ResultType value;          /// the value
  ResultType value_squared;  /// the value of squared

  CUDA_HOST_DEVICE_CALLABLE
  var_std(ResultType _value = 0, ResultType _value_squared = 0)
      : value(_value), value_squared(_value_squared){};

  using this_t = var_std<ResultType>;

  CUDA_HOST_DEVICE_CALLABLE
  this_t operator+(this_t const& rhs) const {
    return this_t((this->value + rhs.value),
                  (this->value_squared + rhs.value_squared));
  };
};

// transformer for `struct var_std` in order to compute `var`, `std`
template <typename ResultType>
struct transformer_var_std {
  using OutputType = var_std<ResultType>;

  CUDA_HOST_DEVICE_CALLABLE
  OutputType operator()(ResultType const& value) {
    return OutputType(value, value * value);
  };
};

// ------------------------------------------------------------------------
// Definitions of device struct for reduction operation
// all `op::xxx` must have `op` and `transformer`
// `op`  is used to compute the reduction at device
// `transformer` is used to convert elements for computing the reduction at device.
// By default `transformer` is static type conversion to ResultType.
// In some cases, it could be square or abs or complex operations
namespace op {

/**
 * @brief  Simple reduction operator CRTP Base class
 *
 * @tparam Derived operator with simple_op interface
 */
template <typename Derived>
struct simple_op {
  /**
   * @brief Get binary operator functor for reduction
   *
   * @return binary operator functor object
   */
  auto get_binary_op() {
    using binary_op = typename Derived::op;
    return binary_op{};
  }

  /**
   * @brief Get transformer functor for transforming input column
   * which inturn is used by reduction binary operator
   *
   * @tparam ResultType output type for element transformer
   *
   * @return element transformer functor object
   */
  template <typename ResultType>
  auto get_element_transformer() {
    using element_transformer = typename Derived::transformer<ResultType>;
    return element_transformer{};
  }

  /**
   * @brief get identity value of type `T` for binary reduction operator
   *
   * @tparam T data type of identity value
   *
   * @return identity value 
   */
  template <typename T>
  constexpr T get_identity() {
    return Derived::op::template identity<T>();
  }
};

// `sum`, `product`, `sum_of_squares`, `min`, `max` are used at simple_reduction
// inferface is defined by CRTP calss simple_op

// operator for `sum`
struct sum : public simple_op<sum> {
  using op = cudf::DeviceSum;

  template <typename ResultType>
  using transformer = thrust::identity<ResultType>;
};

// operator for `product`
struct product : public simple_op<product> {
  using op = cudf::DeviceProduct;

  template <typename ResultType>
  using transformer = thrust::identity<ResultType>;
};

// operator for `sum_of_squares`
struct sum_of_squares : public simple_op<sum_of_squares> {
  using op = cudf::DeviceSum;

  template <typename ResultType>
  using transformer = cudf::transformer_squared<ResultType>;
};

// operator for `min`
struct min : public simple_op<min> {
  using op = cudf::DeviceMin;

  template <typename ResultType>
  using transformer = thrust::identity<ResultType>;
};

// operator for `max`
struct max : public simple_op<max> {
  using op = cudf::DeviceMax;

  template <typename ResultType>
  using transformer = thrust::identity<ResultType>;
};

/**
 * @brief  Compound reduction operator CRTP Base class
 * This template class defines the interface for compound operators
 * In addition to interface defined by simple_op CRTP, this class defines
 * interface for final result transformation.
 *
 * @tparam Derived compound operators derived from compound_op
 */
template <typename Derived>
struct compound_op : public simple_op<Derived> {
  /**
   * @brief  computes the transformed result from result of simple operator.
   *
   * @tparam ResultType output type of compound reduction operator
   * @tparam IntermediateType output type of simple reduction operator
   * @param input output of simple reduction as input for result transformation
   * @param count validity count
   * @param ddof  `ddof` parameter used by variance and standard deviation
   *
   * @return transformed output result of compount operator
   */
  template <typename ResultType, typename IntermediateType>
  CUDA_HOST_DEVICE_CALLABLE static ResultType compute_result(
      const IntermediateType& input, const cudf::size_type& count,
      const cudf::size_type& ddof) {
    // Enforced interface
    return Derived::template intermediate<ResultType>::compute_result(input, count, ddof);
  }
};

// `mean`, `variance`, `standard_deviation` are used at compound_reduction
// compound_reduction requires intermediate::IntermediateType and
// intermediate::compute_result IntermediateType is the intermediate data
// structure type of a single reduction call, it is also used as OutputType of
// cudf::reduction::detail::reduce at compound_reduction. compute_result
// computes the final ResultType from the IntermediateType.
// intemediate::compute_result method is enforced by CRTP base class compound_op

// operator for `mean`
struct mean : public compound_op<mean> {
  using op = cudf::DeviceSum;

  template <typename ResultType>
  using transformer = thrust::identity<ResultType>;

  template <typename ResultType>
  struct intermediate {
    using IntermediateType = ResultType;  // sum value

    // compute `mean` from intermediate type `IntermediateType`
    CUDA_HOST_DEVICE_CALLABLE
    static ResultType compute_result(const IntermediateType& input,
                                     const cudf::size_type& count,
                                     const cudf::size_type& ddof) {
      return (input / count);
    };
  };
};

// operator for `variance`
struct variance : public compound_op<variance> {
  using op = cudf::DeviceSum;

  template <typename ResultType>
  using transformer = cudf::experimental::reduction::transformer_var_std<ResultType>;

  template <typename ResultType>
  struct intermediate {
    using IntermediateType =
        var_std<ResultType>;  // with sum of value, and sum of squared value

    // compute `variance` from intermediate type `IntermediateType`
    CUDA_HOST_DEVICE_CALLABLE
    static ResultType compute_result(const IntermediateType& input,
                                     const cudf::size_type& count,
                                     const cudf::size_type& ddof) {
      ResultType mean = input.value / count;
      ResultType asum = input.value_squared;
      cudf::size_type div = count - ddof;
      ResultType var = asum / div - ((mean * mean) * count) / div;

      return var;
    };
  };
};

// operator for `standard deviation`
struct standard_deviation : public compound_op<standard_deviation> {
  using op = cudf::DeviceSum;

  template <typename ResultType>
  using transformer = cudf::experimental::reduction::transformer_var_std<ResultType>;

  template <typename ResultType>
  struct intermediate {
    using IntermediateType =
        var_std<ResultType>;  // with sum of value, and sum of squared value

    // compute `standard deviation` from intermediate type `IntermediateType`
    CUDA_HOST_DEVICE_CALLABLE
    static ResultType compute_result(const IntermediateType& input,
                                     const cudf::size_type& count,
                                     const cudf::size_type& ddof) {
      using intermediateOp = variance::template intermediate<ResultType>;
      ResultType var = intermediateOp::compute_result(input, count, ddof);

      return static_cast<ResultType>(std::sqrt(var));
    };
  };
};

}  // namespace op
}  // namespace reduction
}  // namespace experimental
}  // namespace cudf
