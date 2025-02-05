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
#include "jit/util.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <jit_preprocessed_files/transform/jit/kernel.cu.jit.hpp>

namespace cudf {
namespace transformation {
namespace jit {
namespace {

using device_data_t = void*;

std::vector<std::string> build_jit_typenames(mutable_column_view output,
                                             std::vector<column_view> const& inputs)
{
  static constexpr auto SCALAR_STRIDE = 0;
  static constexpr auto COLUMN_STRIDE = 1;

  auto const column_type_name = [](data_type data_type, bool is_scalar) {
    return jitify2::reflection::Template("cudf::transformation::jit::strided")
      .instantiate(type_to_name(data_type), is_scalar ? SCALAR_STRIDE : COLUMN_STRIDE);
  };

  std::vector<std::string> typenames;

  typenames.push_back(column_type_name(output.type(), false));
  std::transform(
    inputs.begin(), inputs.end(), std::back_inserter(typenames), [&](auto const& input) {
      bool const is_scalar = input.size() != output.size();
      return column_type_name(input.type(), is_scalar);
    });

  return typenames;
}

std::map<uint32_t, std::string> build_ptx_params(mutable_column_view output,
                                                 std::vector<column_view> const& inputs)
{
  std::map<uint32_t, std::string> params;
  uint32_t index = 0;

  auto const add_column = [&](bool is_output, data_type type) {
    auto const param_type = type_to_name(type);
    params.emplace(index++, is_output ? (param_type + "*") : param_type);
  };

  add_column(true, output.type());

  for (auto& input : inputs) {
    add_column(false, input.type());
  }

  return params;
}

std::vector<device_data_t> build_device_data(mutable_column_view output,
                                             std::vector<column_view> const& inputs)
{
  std::vector<device_data_t> data;

  data.push_back(const_cast<device_data_t>(cudf::jit::get_data_ptr(output)));

  std::transform(inputs.begin(), inputs.end(), std::back_inserter(data), [](auto const& input) {
    return const_cast<device_data_t>(cudf::jit::get_data_ptr(input));
  });

  return data;
}

std::vector<void*> build_launch_args(cudf::size_type& size, std::vector<device_data_t>& device_data)
{
  // JITIFY and NVRTC need non-const pointers even if they aren't written to
  std::vector<void*> args;
  args.push_back(&size);
  std::transform(
    device_data.begin(), device_data.end(), std::back_inserter(args), [](auto& data) -> void* {
      return &data;
    });

  return args;
}

void transform_operation(size_type base_column_size,
                         mutable_column_view output,
                         std::vector<column_view> const& inputs,
                         std::string const& udf,
                         data_type output_type,
                         bool is_ptx,
                         rmm::cuda_stream_view stream,
                         rmm::device_async_resource_ref mr)
{
  std::string const kernel_name = jitify2::reflection::Template("cudf::transformation::jit::kernel")
                                    .instantiate(build_jit_typenames(output, inputs));

  std::string const cuda_source =
    is_ptx ? cudf::jit::parse_single_function_ptx(
               udf, "GENERIC_TRANSFORM_OP", build_ptx_params(output, inputs))
           : cudf::jit::parse_single_function_cuda(udf, "GENERIC_TRANSFORM_OP");

  auto device_data = build_device_data(output, inputs);

  auto args = build_launch_args(base_column_size, device_data);

  cudf::jit::get_program_cache(*transform_jit_kernel_cu_jit)
    .get_kernel(
      kernel_name, {}, {{"transform/jit/operation-udf.hpp", cuda_source}}, {"-arch=sm_."})  //
    ->configure_1d_max_occupancy(0, 0, nullptr, stream.value())                             //
    ->launch(args.data());
}
}  // namespace

}  // namespace jit
}  // namespace transformation

namespace detail {
std::unique_ptr<column> transform(std::vector<column_view> const& inputs,
                                  std::string const& transform_udf,
                                  data_type output_type,
                                  bool is_ptx,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(is_fixed_width(output_type), "Transforms only support fixed-width types");
  CUDF_EXPECTS(
    std::all_of(
      inputs.begin(), inputs.end(), [](auto& input) { return is_fixed_width(input.type()); }),
    "Transforms only support fixed-width types");

  auto const base_column = std::max_element(
    inputs.begin(), inputs.end(), [](auto& a, auto& b) { return a.size() < b.size(); });

  CUDF_EXPECTS(std::all_of(inputs.begin(),
                           inputs.end(),
                           [&](auto const& input) {
                             return (input.size() == 1) || (input.size() == base_column->size());
                           }),
               "All transform input columns must have the same size or be scalar (have size 1)");

  CUDF_EXPECTS(std::all_of(inputs.begin(),
                           inputs.end(),
                           [&](auto const& input) {
                             return (input.size() == 1 && input.null_count() == 0) ||
                                    (input.null_count() == base_column->null_count());
                           }),
               "All transform input columns must have the same null-count");

  auto output = make_fixed_width_column(output_type,
                                        base_column->size(),
                                        copy_bitmask(*base_column, stream, mr),
                                        base_column->null_count(),
                                        stream,
                                        mr);

  if (base_column->is_empty()) { return output; }

  mutable_column_view const output_view = *output;

  // transform
  transformation::jit::transform_operation(
    base_column->size(), output_view, inputs, transform_udf, output_type, is_ptx, stream, mr);

  return output;
}

}  // namespace detail

std::unique_ptr<column> transform(std::vector<column_view> const& inputs,
                                  std::string const& transform_udf,
                                  data_type output_type,
                                  bool is_ptx,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::transform(inputs, transform_udf, output_type, is_ptx, stream, mr);
}

}  // namespace cudf
