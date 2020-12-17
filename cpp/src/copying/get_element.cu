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
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/scalar/scalar_factories.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {

namespace {

struct get_element_functor {
  template <typename T, std::enable_if_t<is_fixed_width<T>() && !is_fixed_point<T>()> *p = nullptr>
  std::unique_ptr<scalar> operator()(
    column_view const &input,
    size_type index,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource())
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
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource())
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
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource())
  {
    auto dict_view    = dictionary_column_view(input);
    auto indices_iter = detail::indexalator_factory::make_input_iterator(dict_view.indices());
    numeric_scalar<size_type> key_index_scalar{index, true, stream};
    auto d_key_index = get_scalar_device_view(key_index_scalar);
    auto d_col       = column_device_view::create(input, stream);

    // retrieve the indices value at index
    device_single_thread(
      [d_key_index, d_col = *d_col, indices_iter, index] __device__() mutable {
        d_key_index.set_value(indices_iter[index]);
        d_key_index.set_valid(d_col.is_valid(index));
      },
      stream);

    if (!key_index_scalar.is_valid(stream)) {
      auto null_result = make_default_constructed_scalar(dict_view.keys().type());
      null_result->set_valid(false, stream);
      return null_result;
    }

    // retrieve the key element using the key-index
    return type_dispatcher(dict_view.keys().type(),
                           get_element_functor{},
                           dict_view.keys(),
                           key_index_scalar.value(stream),
                           stream,
                           mr);
  }

  template <typename T, std::enable_if_t<std::is_same<T, list_view>::value> *p = nullptr>
  std::unique_ptr<scalar> operator()(
    column_view const &input,
    size_type index,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource())
  {
    CUDF_FAIL("get_element_functor not supported for list_view");
  }

  template <typename T, std::enable_if_t<cudf::is_fixed_point<T>()> *p = nullptr>
  std::unique_ptr<scalar> operator()(
    column_view const &input,
    size_type index,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource())
  {
    using Type = typename T::rep;

    auto device_col = column_device_view::create(input, stream);

    rmm::device_scalar<Type> temp_data;
    rmm::device_scalar<bool> temp_valid;

    device_single_thread(
      [buffer   = temp_data.data(),
       validity = temp_valid.data(),
       d_col    = *device_col,
       index] __device__() mutable {
        *buffer   = d_col.element<Type>(index);
        *validity = d_col.is_valid(index);
      },
      stream);

    return std::make_unique<fixed_point_scalar<T>>(std::move(temp_data),
                                                   numeric::scale_type{input.type().scale()},
                                                   temp_valid.value(stream),
                                                   stream,
                                                   mr);
  }

  template <typename T, std::enable_if_t<std::is_same<T, struct_view>::value> *p = nullptr>
  std::unique_ptr<scalar> operator()(
    column_view const &input,
    size_type index,
    rmm::cuda_stream_view stream,
    rmm::mr::device_memory_resource *mr = rmm::mr::get_current_device_resource())
  {
    CUDF_FAIL("get_element_functor not supported for struct_view");
  }
};

}  // namespace

std::unique_ptr<scalar> get_element(column_view const &input,
                                    size_type index,
                                    rmm::cuda_stream_view stream,
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
  return detail::get_element(input, index, rmm::cuda_stream_default, mr);
}

}  // namespace cudf
