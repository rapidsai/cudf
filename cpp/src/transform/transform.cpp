/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.
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

#include "jit/cache.hpp"
#include "jit/parser.hpp"
#include "jit/span.cuh"
#include "jit/util.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/jit/runtime_support.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <jit_preprocessed_files/transform/jit/kernel.cu.jit.hpp>

namespace cudf {
namespace transformation {
namespace jit {
namespace {

struct input_column_reflection {
  std::string type_name;
  bool is_scalar = false;

  [[nodiscard]] std::string accessor(int32_t index) const
  {
    auto column_accessor =
      jitify2::reflection::Template("cudf::transformation::jit::column_accessor")
        .instantiate(type_name, index);

    return is_scalar ? jitify2::reflection::Template("cudf::transformation::jit::scalar")
                         .instantiate(column_accessor)
                     : column_accessor;
  }
};

jitify2::StringVec build_jit_template_params(
  bool has_user_data,
  std::vector<std::string> const& span_outputs,
  std::vector<std::string> const& column_outputs,
  std::vector<input_column_reflection> const& column_inputs)
{
  jitify2::StringVec tparams;

  tparams.emplace_back(jitify2::reflection::reflect(has_user_data));

  std::transform(thrust::make_counting_iterator<size_t>(0),
                 thrust::make_counting_iterator(span_outputs.size()),
                 std::back_inserter(tparams),
                 [&](auto i) {
                   return jitify2::reflection::Template("cudf::transformation::jit::span_accessor")
                     .instantiate(span_outputs[i], i);
                 });

  std::transform(
    thrust::make_counting_iterator<size_t>(0),
    thrust::make_counting_iterator(column_outputs.size()),
    std::back_inserter(tparams),
    [&](auto i) {
      return jitify2::reflection::Template("cudf::transformation::jit::column_accessor")
        .instantiate(column_outputs[i], i);
    });

  std::transform(thrust::make_counting_iterator<size_t>(0),
                 thrust::make_counting_iterator(column_inputs.size()),
                 std::back_inserter(tparams),
                 [&](auto i) { return column_inputs[i].accessor(i); });

  return tparams;
}

std::map<uint32_t, std::string> build_ptx_params(std::vector<std::string> const& output_typenames,
                                                 std::vector<std::string> const& input_typenames,
                                                 bool has_user_data)
{
  std::map<uint32_t, std::string> params;
  uint32_t index = 0;

  if (has_user_data) {
    params.emplace(index++, "void *");
    params.emplace(index++, jitify2::reflection::reflect<cudf::size_type>());
  }

  for (auto& name : output_typenames) {
    params.emplace(index++, name + "*");
  }

  for (auto& name : input_typenames) {
    params.emplace(index++, name);
  }

  return params;
}

template <typename T>
rmm::device_uvector<T> to_device_vector(std::vector<T> const& host,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  rmm::device_uvector<T> device{host.size(), stream, mr};
  detail::cuda_memcpy_async(device_span<T>{device}, host_span<T const>{host}, stream);
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

jitify2::Kernel get_kernel(std::string const& kernel_name, std::string const& cuda_source)
{
  return cudf::jit::get_program_cache(*transform_jit_kernel_cu_jit)
    .get_kernel(kernel_name,
                {},
                {{"transform/jit/operation-udf.hpp", cuda_source}},
                {"-arch=sm_.",
                 "--device-int128",
                 // TODO: remove when we upgrade to CCCL >= 3.0

                 // CCCL WAR for not using the correct INT128 feature macro:
                 // https://github.com/NVIDIA/cccl/issues/3801
                 "-D__SIZEOF_INT128__=16"});
}

input_column_reflection reflect_input_column(size_type base_column_size, column_view column)
{
  CUDF_EXPECTS(column.size() == 1 || column.size() == base_column_size, "");
  return input_column_reflection{type_to_name(column.type()), column.size() != base_column_size};
}

std::vector<input_column_reflection> reflect_input_columns(size_type base_column_size,
                                                           std::vector<column_view> const& inputs)
{
  std::vector<input_column_reflection> reflections;
  std::transform(
    inputs.begin(), inputs.end(), std::back_inserter(reflections), [&](auto const& view) {
      return reflect_input_column(base_column_size, view);
    });

  return reflections;
}

template <typename ColumnView>
std::vector<std::string> column_type_names(std::vector<ColumnView> const& views)
{
  std::vector<std::string> names;

  std::transform(views.begin(), views.end(), std::back_inserter(names), [](auto const& view) {
    return type_to_name(view.type());
  });

  return names;
}

jitify2::ConfiguredKernel build_transform_kernel(
  std::string const& kernel_name,
  size_type base_column_size,
  std::vector<mutable_column_view> const& output_columns,
  std::vector<column_view> const& input_columns,
  bool has_user_data,
  std::string const& udf,
  bool is_ptx,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto const cuda_source =
    is_ptx
      ? cudf::jit::parse_single_function_ptx(
          udf,
          "GENERIC_TRANSFORM_OP",
          build_ptx_params(
            column_type_names(output_columns), column_type_names(input_columns), has_user_data))
      : cudf::jit::parse_single_function_cuda(udf, "GENERIC_TRANSFORM_OP");

  return get_kernel(jitify2::reflection::Template(kernel_name)
                      .instantiate(build_jit_template_params(
                        has_user_data,
                        {},
                        column_type_names(output_columns),
                        reflect_input_columns(base_column_size, input_columns))),
                    cuda_source)
    ->configure_1d_max_occupancy(0, 0, nullptr, stream.value());
}

jitify2::ConfiguredKernel build_span_kernel(std::string const& kernel_name,
                                            size_type base_column_size,
                                            std::vector<std::string> const& span_outputs,
                                            std::vector<column_view> const& input_columns,
                                            bool has_user_data,
                                            std::string const& udf,
                                            bool is_ptx,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  auto const cuda_source =
    is_ptx ? cudf::jit::parse_single_function_ptx(
               udf,
               "GENERIC_TRANSFORM_OP",
               build_ptx_params(span_outputs, column_type_names(input_columns), has_user_data))
           : cudf::jit::parse_single_function_cuda(udf, "GENERIC_TRANSFORM_OP");

  return get_kernel(jitify2::reflection::Template(kernel_name)
                      .instantiate(build_jit_template_params(
                        has_user_data,
                        span_outputs,
                        {},
                        reflect_input_columns(base_column_size, input_columns))),
                    cuda_source)
    ->configure_1d_max_occupancy(0, 0, nullptr, stream.value());
}

void launch_column_output_kernel(jitify2::ConfiguredKernel& kernel,
                                 std::vector<mutable_column_view> const& output_columns,
                                 std::vector<column_view> const& input_columns,
                                 std::optional<void*> user_data,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  auto [output_handles, outputs] =
    column_views_to_device<mutable_column_device_view, mutable_column_view>(
      output_columns, stream, mr);
  auto [input_handles, inputs] =
    column_views_to_device<column_device_view, column_view>(input_columns, stream, mr);

  mutable_column_device_view const* outputs_ptr = outputs.data();
  column_device_view const* inputs_ptr          = inputs.data();
  void* p_user_data                             = user_data.value_or(nullptr);

  std::array<void*, 3> args{&outputs_ptr, &inputs_ptr, &p_user_data};

  kernel->launch(args.data());
}

template <typename T>
void launch_span_kernel(jitify2::ConfiguredKernel& kernel,
                        device_span<T> output,
                        rmm::device_buffer& null_mask,
                        std::vector<column_view> const& input_cols,
                        std::optional<void*> user_data,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr)
{
  auto outputs = to_device_vector(std::vector{cudf::jit::device_optional_span<T>{
                                    cudf::jit::device_span<T>{output.data(), output.size()},
                                    static_cast<bitmask_type*>(null_mask.data())}},
                                  stream,
                                  mr);

  auto [input_handles, inputs] =
    column_views_to_device<column_device_view, column_view>(input_cols, stream, mr);

  cudf::jit::device_optional_span<T> const* outputs_ptr = outputs.data();
  column_device_view const* inputs_ptr                  = inputs.data();
  void* p_user_data                                     = user_data.value_or(nullptr);

  std::array<void*, 3> args{&outputs_ptr, &inputs_ptr, &p_user_data};

  kernel->launch(args.data());
}

bool is_scalar(cudf::size_type base_column_size, cudf::size_type column_size)
{
  return column_size == 1 && column_size != base_column_size;
}

std::tuple<rmm::device_buffer, size_type> make_transform_null_mask(
  column_view base_column,
  std::vector<column_view> const& inputs,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  // collect the non-scalar elements that contribute to the resulting bitmask
  std::vector<column_view> bitmask_columns;

  // to handle null masks for scalars, we just check if the scalar element is null. If it is null,
  // then all the rows of the transform output will be null. This helps us prevent creating
  // column-sized bitmasks for each scalar.
  for (column_view const& col : inputs) {
    if (is_scalar(base_column.size(), col.size())) {
      // all nulls
      if (col.has_nulls()) {
        return std::make_tuple(
          create_null_mask(base_column.size(), mask_state::ALL_NULL, stream, mr),
          base_column.size());
      }
    } else {
      bitmask_columns.emplace_back(col);
    }
  }

  return cudf::bitmask_and(table_view{bitmask_columns}, stream, mr);
}

std::unique_ptr<column> transform_operation(column_view base_column,
                                            data_type output_type,
                                            std::vector<column_view> const& inputs,
                                            std::string const& udf,
                                            bool is_ptx,
                                            std::optional<void*> user_data,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  auto [null_mask, null_count] = make_transform_null_mask(base_column, inputs, stream, mr);

  auto output = make_fixed_width_column(
    output_type, base_column.size(), std::move(null_mask), null_count, stream, mr);

  auto kernel = build_transform_kernel(is_fixed_point(output_type)
                                         ? "cudf::transformation::jit::fixed_point_kernel"
                                         : "cudf::transformation::jit::kernel",
                                       base_column.size(),
                                       {*output},
                                       inputs,
                                       user_data.has_value(),
                                       udf,
                                       is_ptx,
                                       stream,
                                       mr);

  launch_column_output_kernel(kernel, {*output}, inputs, user_data, stream, mr);

  return output;
}

std::unique_ptr<column> string_view_operation(column_view base_column,
                                              std::vector<column_view> const& inputs,
                                              std::string const& udf,
                                              bool is_ptx,
                                              std::optional<void*> user_data,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  auto [null_mask, null_count] = make_transform_null_mask(base_column, inputs, stream, mr);

  auto kernel = build_span_kernel("cudf::transformation::jit::span_kernel",
                                  base_column.size(),
                                  {"cudf::string_view"},
                                  inputs,
                                  user_data.has_value(),
                                  udf,
                                  is_ptx,
                                  stream,
                                  mr);

  rmm::device_uvector<string_view> string_views(base_column.size(), stream, mr);

  launch_span_kernel<string_view>(kernel, string_views, null_mask, inputs, user_data, stream, mr);

  auto output = make_strings_column(string_views, string_view{}, stream, mr);

  output->set_null_mask(std::move(null_mask), null_count);

  return output;
}

void perform_checks(column_view base_column,
                    data_type output_type,
                    std::vector<column_view> const& inputs)
{
  CUDF_EXPECTS(is_runtime_jit_supported(), "Runtime JIT is only supported on CUDA Runtime 11.5+");
  CUDF_EXPECTS(is_fixed_width(output_type) || output_type.id() == type_id::STRING,
               "Transforms only support output of fixed-width or string types",
               std::invalid_argument);
  CUDF_EXPECTS(std::all_of(inputs.begin(),
                           inputs.end(),
                           [](auto& input) {
                             return is_fixed_width(input.type()) ||
                                    (input.type().id() == type_id::STRING);
                           }),
               "Transforms only support input of fixed-width or string types",
               std::invalid_argument);

  CUDF_EXPECTS(std::all_of(inputs.begin(),
                           inputs.end(),
                           [&](auto const& input) {
                             return (input.size() == 1) || (input.size() == base_column.size());
                           }),
               "All transform input columns must have the same size or be scalar (have size 1)",
               std::invalid_argument);
}

}  // namespace

}  // namespace jit
}  // namespace transformation

namespace detail {

std::unique_ptr<column> transform(std::vector<column_view> const& inputs,
                                  std::string const& udf,
                                  data_type output_type,
                                  bool is_ptx,
                                  std::optional<void*> user_data,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(
    !inputs.empty(), "Transform must have at least 1 input column", std::invalid_argument);

  auto const base_column = std::max_element(
    inputs.begin(), inputs.end(), [](auto& a, auto& b) { return a.size() < b.size(); });

  transformation::jit::perform_checks(*base_column, output_type, inputs);

  if (is_fixed_width(output_type)) {
    return transformation::jit::transform_operation(
      *base_column, output_type, inputs, udf, is_ptx, user_data, stream, mr);
  } else if (output_type.id() == type_id::STRING) {
    return transformation::jit::string_view_operation(
      *base_column, inputs, udf, is_ptx, user_data, stream, mr);
  } else {
    CUDF_FAIL("Unsupported output type for transform operation");
  }
}

}  // namespace detail

std::unique_ptr<column> transform(std::vector<column_view> const& inputs,
                                  std::string const& udf,
                                  data_type output_type,
                                  bool is_ptx,
                                  std::optional<void*> user_data,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::transform(inputs, udf, output_type, is_ptx, user_data, stream, mr);
}

}  // namespace cudf
