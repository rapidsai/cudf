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
#include <cudf/copying.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/scalar/scalar_factories.hpp>

#include <cudf/detail/utilities/cuda.cuh>

namespace cudf {
namespace detail {

namespace {

struct get_element_functor {
  template <typename T, std::enable_if_t<is_fixed_width<T>() && !is_fixed_point<T>()> *p = nullptr>
  std::unique_ptr<scalar> operator()(
    column_view const &input,
    size_type index,
    cudaStream_t stream                 = 0,
    rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource())
  {
    auto s = make_fixed_width_scalar(data_type(type_to_id<T>()), stream, mr);

    using ScalarType = cudf::scalar_type_t<T>;
    auto typed_s     = static_cast<ScalarType *>(s.get());

    auto device_s   = get_scalar_device_view(*typed_s);
    auto device_col = column_device_view::create(input, stream);

    device_single_thread(
      [device_s, d_col = *device_col, index] __device__() mutable {
        device_s.set_value(d_col.element<T>(index));
        device_s.set_valid(d_col.is_valid(index));
      },
      stream);
    return s;
  }

  template <typename T, std::enable_if_t<std::is_same<T, string_view>::value> *p = nullptr>
  std::unique_ptr<scalar> operator()(
    column_view const &input,
    size_type index,
    cudaStream_t stream                 = 0,
    rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource())
  {
    auto device_col = column_device_view::create(input, stream);

    rmm::device_scalar<string_view> temp_data;
    rmm::device_scalar<bool> temp_valid;

    device_single_thread(
      [buffer   = temp_data.data(),
       validity = temp_valid.data(),
       d_col    = *device_col,
       index] __device__() mutable {
        *buffer   = d_col.element<string_view>(index);
        *validity = d_col.is_valid(index);
      },
      stream);

    return std::make_unique<string_scalar>(temp_data, temp_valid.value(stream), stream, mr);
  }

  template <typename T, std::enable_if_t<std::is_same<T, dictionary32>::value> *p = nullptr>
  std::unique_ptr<scalar> operator()(
    column_view const &input,
    size_type index,
    cudaStream_t stream                 = 0,
    rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource())
  {
    auto dict_view = dictionary_column_view(input);
    auto key_index_scalar =
      get_element_functor{}.operator()<int32_t>(dict_view.indices(), index, stream);

    size_type key_index =
      static_cast<numeric_scalar<int32_t> const *>(key_index_scalar.get())->value(stream);
    auto result = type_dispatcher(
      dict_view.keys().type(), get_element_functor{}, dict_view.keys(), key_index, stream, mr);

    auto result_validity = result->validity_data();
    auto device_col      = column_device_view::create(input, stream);

    device_single_thread(
      [result_validity, d_col = *device_col, index] __device__() mutable {
        *result_validity = d_col.is_valid(index);
      },
      stream);

    return result;
  }

  template <typename T, std::enable_if_t<std::is_same<T, list_view>::value> *p = nullptr>
  std::unique_ptr<scalar> operator()(
    column_view const &input,
    size_type index,
    cudaStream_t stream                 = 0,
    rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource())
  {
    CUDF_FAIL("get_element_functor not supported for list_view");
  }

  template <typename T, std::enable_if_t<std::is_same<T, numeric::decimal32>::value> *p = nullptr>
  std::unique_ptr<scalar> operator()(
    column_view const &input,
    size_type index,
    cudaStream_t stream                 = 0,
    rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource())
  {
    CUDF_FAIL("get_element_functor not supported for decimal32");
  }

  template <typename T, std::enable_if_t<std::is_same<T, numeric::decimal64>::value> *p = nullptr>
  std::unique_ptr<scalar> operator()(
    column_view const &input,
    size_type index,
    cudaStream_t stream                 = 0,
    rmm::mr::device_memory_resource *mr = rmm::mr::get_default_resource())
  {
    CUDF_FAIL("get_element_functor not supported for decimal64");
  }
};

}  // namespace

std::unique_ptr<scalar> get_element(column_view const &input,
                                    size_type index,
                                    cudaStream_t stream,
                                    rmm::mr::device_memory_resource *mr)
{
  CUDF_EXPECTS(index >= 0 and index < input.size(), "Index out of bounds");
  return type_dispatcher(input.type(), get_element_functor{}, input, index, stream, mr);
}

}  // namespace detail

std::unique_ptr<scalar> get_element(column_view const &input,
                                    size_type index,
                                    rmm::mr::device_memory_resource *mr)
{
  return detail::get_element(input, index, 0, mr);
}

}  // namespace cudf
