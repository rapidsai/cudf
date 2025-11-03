/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/utilities/cuda_memcpy.hpp>

#include <jit/span.cuh>
#include <jit_preprocessed_files/transform/jit/kernel.cu.jit.hpp>

#include <algorithm>

namespace cudf {
namespace jit {

constexpr bool is_scalar(cudf::size_type base_column_size, cudf::size_type column_size)
{
  return column_size == 1 && column_size != base_column_size;
}

typename std::vector<column_view>::const_iterator get_transform_base_column(
  std::vector<column_view> const& inputs);
struct input_column_reflection {
  std::string type_name;
  bool is_scalar = false;

  [[nodiscard]] std::string accessor(int32_t index) const
  {
    auto column_accessor =
      jitify2::reflection::Template("cudf::jit::column_accessor").instantiate(type_name, index);

    return is_scalar ? jitify2::reflection::Template("cudf::jit::scalar_accessor")
                         .instantiate(column_accessor)
                     : column_accessor;
  }
};

jitify2::StringVec build_jit_template_params(
  bool has_user_data,
  null_aware is_null_aware,
  std::vector<std::string> const& span_outputs,
  std::vector<std::string> const& column_outputs,
  std::vector<input_column_reflection> const& column_inputs);

std::map<uint32_t, std::string> build_ptx_params(std::vector<std::string> const& output_typenames,
                                                 std::vector<std::string> const& input_typenames,
                                                 bool has_user_data);

template <typename T>
rmm::device_uvector<T> to_device_vector(std::vector<T> const& host,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  rmm::device_uvector<T> device{host.size(), stream, mr};
  cudf::detail::cuda_memcpy_async<T>(device, host, stream);
  return device;
}

template <typename DeviceView, typename ColumnView>
std::tuple<std::vector<std::unique_ptr<DeviceView, std::function<void(DeviceView*)>>>,
           rmm::device_uvector<DeviceView>>
column_views_to_device(std::vector<ColumnView> const& views,
                       rmm::cuda_stream_view stream,
                       rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<DeviceView, std::function<void(DeviceView*)>>> handles;

  std::transform(views.begin(), views.end(), std::back_inserter(handles), [&](auto const& view) {
    return DeviceView::create(view, stream);
  });

  std::vector<DeviceView> host_array;

  std::transform(
    handles.begin(), handles.end(), std::back_inserter(host_array), [](auto const& handle) {
      return *handle;
    });

  auto device_array = to_device_vector(host_array, stream, mr);

  return std::make_tuple(std::move(handles), std::move(device_array));
}

input_column_reflection reflect_input_column(size_type base_column_size, column_view column);

std::vector<input_column_reflection> reflect_input_columns(size_type base_column_size,
                                                           std::vector<column_view> const& inputs);

template <typename ColumnView>
std::vector<std::string> column_type_names(std::vector<ColumnView> const& views)
{
  std::vector<std::string> names;

  std::transform(views.begin(), views.end(), std::back_inserter(names), [](auto const& view) {
    return type_to_name(view.type());
  });

  return names;
}

}  // namespace jit
}  // namespace cudf
