/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

typedef void const* device_data_t;

void transform_operation(mutable_column_view output,
                         host_span<column_view const> inputs,
                         std::string const& udf,
                         data_type output_type,
                         bool is_ptx,
                         rmm::cuda_stream_view stream)
{
  jitify2::StringVec template_arguments;
  template_arguments.push_back(cudf::type_to_name(output.type()));
  std::transform(inputs.begin(),
                 inputs.end(),
                 std::back_inserter(template_arguments),
                 [](cudf::column_view input) { return cudf::type_to_name(input.type()); });

  std::string const kernel_name =
    jitify2::reflection::Template("cudf::transformation::jit::kernel")  //
      .instantiate(template_arguments);

  std::string cuda_source =
    is_ptx ? cudf::jit::parse_single_function_ptx(udf,  //
                                                  "GENERIC_TRANSFORM_OP",
                                                  cudf::type_to_name(output_type),
                                                  {0})
           : cudf::jit::parse_single_function_cuda(udf,  //
                                                   "GENERIC_TRANSFORM_OP");

  std::vector<device_data_t> device_data;

  device_data.push_back(cudf::jit::get_data_ptr(output));
  std::transform(
    inputs.begin(), inputs.end(), std::back_inserter(device_data), [](column_view view) {
      return cudf::jit::get_data_ptr(view);
    });

  cudf::size_type const size = output.size();

  std::vector<void const*> args;
  args.push_back(&size);
  std::transform(device_data.begin(),
                 device_data.end(),
                 std::back_inserter(args),
                 [](device_data_t& device_data) -> void const* { return &device_data; });

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
std::unique_ptr<column> transform(host_span<column_view const> inputs,
                                  std::string const& unary_udf,
                                  data_type output_type,
                                  bool is_ptx,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(!inputs.empty(), "Number of input columns must be atleast 1");
  CUDF_EXPECTS(
    std::all_of(
      inputs.begin(), inputs.end(), [](auto view) { return is_fixed_width(view.type()); }),
    "All input columns must contain fixed-width types");

  column_view const primary_column = inputs[0];

  std::unique_ptr<column> output = make_fixed_width_column(output_type,
                                                           primary_column.size(),
                                                           copy_bitmask(primary_column, stream, mr),
                                                           primary_column.null_count(),
                                                           stream,
                                                           mr);

  if (primary_column.is_empty()) { return output; }

  mutable_column_view const output_view = *output;

  // transform
  transformation::jit::transform_operation(
    output_view, inputs, unary_udf, output_type, is_ptx, stream);

  return output;
}

}  // namespace detail

std::unique_ptr<column> transform(host_span<column_view const> inputs,
                                  std::string const& unary_udf,
                                  data_type output_type,
                                  bool is_ptx,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::transform(inputs, unary_udf, output_type, is_ptx, stream, mr);
}

}  // namespace cudf
