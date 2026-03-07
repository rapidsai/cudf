/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/scalar_column_view.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>

#include <thrust/iterator/transform_iterator.h>

#include <jit/cache.hpp>
#include <jit/span.cuh>
#include <jit_preprocessed_files/transform/jit/kernel.cu.jit.hpp>

#include <algorithm>
#include <span>
#include <variant>
#include <vector>

namespace cudf {
namespace jit {

[[deprecated("Use scalar_column_view instead")]] bool is_scalar(cudf::size_type base_column_size,
                                                                cudf::size_type column_size);

[[deprecated("Use get_projection_size instead")]] typename std::vector<column_view>::const_iterator
get_transform_base_column(std::vector<column_view> const& inputs);

size_type get_projection_size(
  std::span<std::variant<column_view, scalar_column_view> const> inputs);

std::map<uint32_t, std::string> build_ptx_params(std::span<std::string const> output_typenames,
                                                 std::span<std::string const> input_typenames,
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
column_views_to_device(std::span<ColumnView const> views,
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

std::vector<std::string> input_type_names(
  std::span<std::variant<column_view, scalar_column_view> const> views);

// TODO: how do we support mutable column view?, the arguments need to be casted
/**
 * @brief Reflects the input column or scalar into a string that can be used in the generated CUDA
 * code.
 *
 * @param index The index of the input in the list of inputs.
 * @param input The input column or scalar to reflect.
 * @param may_be_nullable Whether to allow runtime checks of the nullability of the input
 * @return A string representing the input in the generated CUDA code.
 */
std::string reflect_input(
  int32_t index,
  std::variant<column_view, scalar_column_view, mutable_column_view> const& input,
  bool may_be_nullable);

/**
 * @brief Reflects the input columns or scalars into strings that can be used in the generated CUDA
 * code.
 *
 * @param inputs The input columns or scalars to reflect.
 * @param may_be_nullable Whether to allow runtime checks of the nullability of the inputs
 * @return A vector of strings representing the inputs in the generated CUDA code.
 */
std::vector<std::string> reflect_inputs(
  std::span<std::variant<column_view, scalar_column_view, mutable_column_view> const> inputs,
  std::span<bool const> may_be_nullable);

jitify2::Kernel get_udf_kernel(jitify2::PreprocessedProgramData const& preprocessed_program_data,
                               std::string const& kernel_name,
                               std::string const& cuda_source);

}  // namespace jit
}  // namespace cudf
