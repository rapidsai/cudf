/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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
#include <cudf/strings/detail/convert/convert_booleans.hpp>
#include <cudf/strings/detail/convert/convert_datetime.hpp>
#include <cudf/strings/detail/convert/convert_floats.hpp>
#include <cudf/strings/detail/convert/convert_integers.hpp>

#include <rmm/thrust_rmm_allocator.h>

namespace cudf {
namespace experimental {
namespace detail {
template <typename T, typename R>
static constexpr bool is_numeric_or_date_conversion() {
    return (cudf::is_numeric<T>() || cudf::is_timestamp<T>()) &&
           (cudf::is_numeric<R>() || cudf::is_timestamp<R>());
}

template <typename T, typename R>
static constexpr bool is_string_conversion() {
    return (std::is_same<T, cudf::string_view>::value && (cudf::is_numeric<R>() || cudf::is_timestamp<R>() || cudf::is_boolean<R>())) ||
           ((cudf::is_numeric<T>() || cudf::is_timestamp<T>() || cudf::is_boolean<T>()) && std::is_same<R, cudf::string_view>::value);
}

template <typename _T, typename _R>
struct unary_cast {
  template <
      typename T = _T,
      typename R = _R,
      typename std::enable_if_t<
          (cudf::is_numeric<T>() && cudf::is_numeric<R>())>* = nullptr>
  CUDA_DEVICE_CALLABLE R operator()(T const element) {
    return static_cast<R>(element);
  }
  template <
      typename T = _T,
      typename R = _R,
      typename std::enable_if_t<
          (cudf::is_timestamp<T>() && cudf::is_timestamp<R>())>* = nullptr>
  CUDA_DEVICE_CALLABLE R operator()(T const element) {
    return static_cast<R>(simt::std::chrono::floor<R::duration>(element));
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

template <typename _T>
struct dispatch_unary_cast_to {
  column_view input;

  dispatch_unary_cast_to(column_view inp) : input(inp) {}

  template <typename R, typename T = _T,
            typename std::enable_if_t<is_numeric_or_date_conversion<T, R>() &&
                                      !is_string_conversion<T, R>()>* = nullptr>
  std::unique_ptr<column> operator()(data_type type,
                                     std::string const& timestamp_format,
                                     string_scalar const& true_string,
                                     string_scalar const& false_string,
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

  template <typename R, typename T = _T,
            typename std::enable_if_t<is_string_conversion<T, R>() &&
                                      !is_numeric_or_date_conversion<T, R>()>* = nullptr>
  std::unique_ptr<column> operator()(data_type type,
                                     std::string const& timestamp_format,
                                     string_scalar const& true_string,
                                     string_scalar const& false_string,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream) {
      // Converting to string
      if (std::is_same<R, cudf::string_view>::value) {
          if ( cudf::is_boolean<T>()) {
              return strings::detail::from_booleans(input, true_string, false_string, mr, stream);
          } else if(std::is_integral<T>::value) {
              return strings::detail::from_integers(input, mr, stream);
          } else if (std::is_floating_point<T>::value) {
              return strings::detail::from_floats(input, mr, stream);
          } else if (cudf::is_timestamp<T>()) {
              return strings::detail::from_timestamps(input, timestamp_format, mr, stream);
          } else {
              CUDF_FAIL("Not a valid type to be converted to string");
          }
      }
      //conversting from string
      else {
          if ( cudf::is_boolean<R>()) {
              return strings::detail::to_booleans(input, true_string, mr, stream);
          } else if(std::is_integral<R>::value) {
              return strings::detail::to_integers(input, type, mr, stream);
          } else if (std::is_floating_point<R>::value) {
              return strings::detail::to_floats(input, type, mr, stream);
          } else if (cudf::is_timestamp<R>()) {
              return strings::detail::to_timestamps(input, type, timestamp_format, mr, stream);
          } else {
              CUDF_FAIL("Not a valid type to be converted from string");
          }
      }
  }

  template <typename R, typename T = _T,
            typename std::enable_if_t<!is_numeric_or_date_conversion<T, R>() &&
                                      !is_string_conversion<T, R>()>* = nullptr>
  std::unique_ptr<column> operator()(data_type type,
                                     std::string const& timestamp_format,
                                     string_scalar const& true_string,
                                     string_scalar const& false_string,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream) {
    CUDF_FAIL("Column type must be numeric or timestamp or boolean or string");
  }
};

struct dispatch_unary_cast_from {
  column_view input;

  dispatch_unary_cast_from(column_view inp) : input(inp) {}

  template <typename T,
            typename std::enable_if_t<cudf::is_numeric<T>() ||
                                      cudf::is_timestamp<T>() ||
                                      cudf::is_boolean<T>() ||
                                      std::is_same<T, cudf::string_view>::value>* = nullptr>
  std::unique_ptr<column> operator()(data_type type,
                                     std::string const& timestamp_format,
                                     string_scalar const& true_string,
                                     string_scalar const& false_string,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream) {
    return experimental::type_dispatcher(type, dispatch_unary_cast_to<T>{input},
                                         type, timestamp_format, true_string, false_string, mr, stream);
  }

  template <typename T,
            typename std::enable_if_t<!cudf::is_timestamp<T>() &&
                                      !cudf::is_numeric<T>() &&
                                      !cudf::is_boolean<T>() &&
                                      !std::is_same<T, cudf::string_view>::value>* = nullptr>
  std::unique_ptr<column> operator()(data_type type,
                                     std::string const& timestamp_format,
                                     string_scalar const& true_string,
                                     string_scalar const& false_string,
                                     rmm::mr::device_memory_resource* mr,
                                     cudaStream_t stream) {
    CUDF_FAIL("Column type must be numeric or timestamp or boolean or string");
  }
};

std::unique_ptr<column> cast(column_view const& input,
                             data_type type,
                             std::string const& timestamp_format,
                             string_scalar const& true_string,
                             string_scalar const& false_string,
                             rmm::mr::device_memory_resource* mr,
                             cudaStream_t stream) {

  return experimental::type_dispatcher(input.type(),
                                       detail::dispatch_unary_cast_from{input},
                                       type, timestamp_format, true_string, false_string, mr, stream);
}

}  // namespace detail

std::unique_ptr<column> cast(column_view const& input,
                             data_type type,
                             std::string const& timestamp_format,
                             string_scalar const& true_string,
                             string_scalar const& false_string,
                             rmm::mr::device_memory_resource* mr) {
  return detail::cast(input, type, timestamp_format, true_string, false_string, mr);
}

}  // namespace experimental
}  // namespace cudf
