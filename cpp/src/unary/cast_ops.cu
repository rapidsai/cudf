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

#include <cudf/column/column.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/detail/unary.hpp>

#include <rmm/thrust_rmm_allocator.h>

namespace cudf {
namespace experimental {
namespace detail {

template <typename _T, typename _R>
struct unary_cast {
  template <
      typename T = _T,
      typename R = _R,
      typename std::enable_if_t<
          (cudf::is_numeric<T>() && cudf::is_numeric<R>()) ||
          (cudf::is_timestamp<T>() && cudf::is_timestamp<R>())>* = nullptr>
  CUDA_DEVICE_CALLABLE R operator()(T const element) {
    return static_cast<R>(element);
  }
  template <typename T = _T,
            typename R = _R,
            typename std::enable_if_t<cudf::is_numeric<T>() &&
                                      cudf::is_timestamp<R>()>* = nullptr>
  CUDA_DEVICE_CALLABLE R operator()(T const element) {
    return static_cast<R>(static_cast<typename R::rep>(element));
  }
  template <typename T = _T,
            typename R = _R,
            typename std::enable_if_t<cudf::is_timestamp<T>() &&
                                      cudf::is_numeric<R>()>* = nullptr>
  CUDA_DEVICE_CALLABLE R operator()(T const element) {
    return static_cast<R>(element.time_since_epoch().count());
  }
};

template <typename T>
struct dispatch_unary_cast_to {
  column_view input;

  dispatch_unary_cast_to(column_view inp) : input(inp) {}

  template <typename R,
            typename std::enable_if_t<cudf::is_numeric<R>() ||
                                      cudf::is_timestamp<R>()>* = nullptr>
  std::unique_ptr<column> operator()(data_type type,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream) {
    auto size = input.size();
    auto output = std::make_unique<column>(
        type, size, rmm::device_buffer{size * cudf::size_of(type), 0, mr},
        copy_bitmask(input, 0, mr), input.null_count());

    mutable_column_view output_mutable = *output;

    thrust::transform(rmm::exec_policy(stream)->on(stream), input.begin<T>(),
                      input.end<T>(), output_mutable.begin<R>(),
                      unary_cast<T, R>{});

    return output;
  }

  template <typename R,
            typename std::enable_if_t<!cudf::is_numeric<R>() &&
                                      !cudf::is_timestamp<R>()>* = nullptr>
  std::unique_ptr<column> operator()(data_type type,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream) {
    CUDF_FAIL("Column type must be numeric or timestamp");
  }
};

struct dispatch_unary_cast_from {
  column_view input;

  dispatch_unary_cast_from(column_view inp) : input(inp) {}

  template <typename T,
            typename std::enable_if_t<cudf::is_numeric<T>() ||
                                      cudf::is_timestamp<T>()>* = nullptr>
  std::unique_ptr<column> operator()(data_type type,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream) {
    return experimental::type_dispatcher(type, dispatch_unary_cast_to<T>{input},
                                         type, mr, stream);
  }

  template <typename T,
            typename std::enable_if_t<!cudf::is_timestamp<T>() &&
                                      !cudf::is_numeric<T>()>* = nullptr>
  std::unique_ptr<column> operator()(data_type type,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream) {
    CUDF_FAIL("Column type must be numeric or timestamp");
  }
};

std::unique_ptr<column> cast(column_view const& input,
                             data_type type,
                             rmm::mr::device_memory_resource* mr,
                             cudaStream_t stream) {
  CUDF_EXPECTS(is_fixed_width(type), "Unary cast type must be fixed-width.");

  return experimental::type_dispatcher(input.type(),
                                       detail::dispatch_unary_cast_from{input},
                                       type, mr, stream);
}

}  // namespace detail

std::unique_ptr<column> cast(column_view const& input,
                             data_type type,
                             rmm::mr::device_memory_resource* mr) {
  return detail::cast(input, type, mr);
}


}  // namespace experimental
}  // namespace cudf
