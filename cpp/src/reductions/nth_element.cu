/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/reduction_functions.hpp>
#include <cudf/utilities/traits.hpp>
#include <rmm/device_scalar.hpp>

namespace cudf {
namespace experimental {
namespace reduction {
namespace {

template <typename ElementType>
struct nth_element_functor {
  template <typename OutputType>
  static constexpr bool is_supported_v() {
    return std::is_convertible<ElementType, OutputType>::value;
  }

  template <typename OutputType>
  std::enable_if_t<(is_supported_v<OutputType>() &&
                    is_fixed_width<OutputType>()),
                   std::unique_ptr<scalar>>
  operator()(column_device_view const& dcol,
             size_type n,
             rmm::mr::device_memory_resource* mr,
             cudaStream_t stream) {
    bool const n_valid = n < dcol.size() && n >= -dcol.size();
    using ScalarType = cudf::experimental::scalar_type_t<OutputType>;
    auto result = new ScalarType(OutputType{0}, n_valid, stream, mr);
    if (n_valid) {
      n = (n < 0 ? dcol.size() + n : n);
      //device to device copy
      thrust::for_each_n(
          rmm::exec_policy(stream)->on(stream),
          thrust::make_counting_iterator<size_type>(n),1,
          [dres = result->data(), dvalid = result->validity_data(),
           dcol] __device__(auto n) {
            *dres = static_cast<OutputType>(dcol.element<ElementType>(n));
            *dvalid = dcol.is_valid(n);
          });
    }
    return std::unique_ptr<scalar>(result);
  }

  template <typename OutputType>
  static constexpr bool is_supported_s() {
    return std::is_same<OutputType, cudf::string_view>::value &&
           std::is_same<ElementType, cudf::string_view>::value;
  }

  template <typename OutputType>
  static constexpr bool is_not_supported() {
    return  !is_supported_v<OutputType>() ||
           (!is_fixed_width<OutputType>() && !is_supported_s<OutputType>());
  }

  template <typename OutputType>
  std::enable_if_t<is_supported_s<OutputType>(), std::unique_ptr<scalar>>
  operator()(column_device_view const& dcol,
             size_type n,
             rmm::mr::device_memory_resource* mr, cudaStream_t stream) {
    bool const n_valid = n < dcol.size() && n >= -dcol.size();
    using OutputPair = thrust::pair<OutputType, bool>;
    OutputPair host_result{{}, n_valid};
    if (n_valid) {
      n = (n < 0 ? dcol.size() + n : n);
      if (dcol.nullable()) {
        auto it = dcol.pair_begin<ElementType, true>() + n;
        rmm::device_scalar<OutputPair> dev(host_result, stream, mr);
        thrust::copy(rmm::exec_policy(stream)->on(stream), it, it + 1, dev.data());
        host_result = dev.value();
      } else {
        auto it = dcol.pair_begin<ElementType, false>() + n;
        rmm::device_scalar<OutputPair> dev(host_result, stream, mr);
        thrust::copy(rmm::exec_policy(stream)->on(stream), it, it + 1, dev.data());
        host_result = dev.value();
      }
    }
    using ScalarType = cudf::experimental::scalar_type_t<OutputType>;
    // only for string_view, data is copied
    auto result = new ScalarType(host_result.first, host_result.second, stream, mr);
    return std::unique_ptr<scalar>(result);
  }

  template <typename OutputType>
  std::enable_if_t<is_not_supported<OutputType>(), std::unique_ptr<scalar>>
  operator()(column_device_view const& dcol,
             size_type n,
             rmm::mr::device_memory_resource* mr,
             cudaStream_t stream) {
    CUDF_FAIL("Only fixed type is supported for nth element");
  }
};

struct element_dispatch_nth_element_functor {
  template <typename ElementType>
  std::unique_ptr<cudf::scalar> operator()(column_device_view const& dcol,
                                           data_type const output_dtype,
                                           size_type n,
                                           rmm::mr::device_memory_resource* mr,
                                           cudaStream_t stream) {
    return cudf::experimental::type_dispatcher(
        output_dtype, nth_element_functor<ElementType>{}, 
        dcol, n, mr, stream);
  }
};
}  // namespace anonymous

std::unique_ptr<cudf::scalar> nth_element(column_view const& col,
                                          data_type const output_dtype,
                                          size_type n,
                                          rmm::mr::device_memory_resource* mr,
                                          cudaStream_t stream) {
  auto dcol = cudf::column_device_view::create(col, stream);
  return cudf::experimental::type_dispatcher(
      col.type(), element_dispatch_nth_element_functor{}, 
      *dcol, output_dtype, n, mr, stream);
}

}  // namespace reduction
}  // namespace experimental
}  // namespace cudf
