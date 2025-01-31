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

std::map<unsigned int, std::string> ptx_params(mutable_column_view output,
                                               std::vector<column_view> const& inputs

)
{
  std::map<unsigned int, std::string> types;

  unsigned int i = 0;
  types.emplace(i, cudf::type_to_name(output.type()) + " *");
  i++;

  for (auto& input : inputs) {
    types.emplace(i, cudf::type_to_name(input.type()));
    i++;
  }

  return types;
}

void transform_operation(mutable_column_view output,
                         std::vector<column_view> const& inputs,
                         std::string const& udf,
                         data_type output_type,
                         bool is_ptx,
                         rmm::cuda_stream_view stream,
                         cudf::size_type base_column_size)
{
  std::vector<std::string> typenames;
  typenames.push_back(jitify2::reflection::Template("cudf::transformation::jit::strided")
                        .instantiate(cudf::type_to_name(output.type()), 1));

  for (auto& input : inputs) {
    bool const should_advance = input.size() == base_column_size;
    typenames.push_back(jitify2::reflection::Template("cudf::transformation::jit::strided")
                          .instantiate(cudf::type_to_name(input.type()), should_advance ? 1 : 0));
  }

  std::string const kernel_name =
    jitify2::reflection::Template("cudf::transformation::jit::kernel")  //
      .instantiate(typenames);

  std::string cuda_source = is_ptx
                              ? cudf::jit::parse_single_function_ptx(udf,  //
                                                                     "GENERIC_TRANSFORM_OP",
                                                                     ptx_params(output, inputs))
                              : cudf::jit::parse_single_function_cuda(udf,  //
                                                                      "GENERIC_TRANSFORM_OP");

  {
    std::vector<device_data_t> device_data;

    device_data.push_back(const_cast<device_data_t>(cudf::jit::get_data_ptr(output)));
    std::transform(
      inputs.begin(), inputs.end(), std::back_inserter(device_data), [](column_view view) {
        return const_cast<device_data_t>(cudf::jit::get_data_ptr(view));
      });

    int64_t size = output.size();

    std::vector<void*> args;
    args.push_back(&size);
    std::transform(device_data.begin(),
                   device_data.end(),
                   std::back_inserter(args),
                   [](device_data_t& data) -> void* { return &data; });

    cudf::jit::get_program_cache(*transform_jit_kernel_cu_jit)
      .get_kernel(
        kernel_name, {}, {{"transform/jit/operation-udf.hpp", cuda_source}}, {"-arch=sm_."})  //
      ->configure_1d_max_occupancy(0, 0, nullptr, stream.value())                             //
      ->launch(args.data());
  }
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
  CUDF_EXPECTS(is_fixed_width(output_type), "Unexpected non-fixed-width type.");
  std::for_each(inputs.begin(), inputs.end(), [](auto& col) {
    CUDF_EXPECTS(is_fixed_width(col.type()), "Unexpected non-fixed-width type.");
  });

  auto base_column = std::max_element(
    inputs.begin(), inputs.end(), [](auto& a, auto& b) { return a.size() < b.size(); });

  std::for_each(inputs.begin(), inputs.end(), [&](column_view const& col) {
    CUDF_EXPECTS((col.size() == 1) || (col.size() == base_column->size()),
                 "All columns must have the same size or have size 1 (scalar)");
    CUDF_EXPECTS(col.null_count() == 0, "All columns must have non-null values");
  });

  std::unique_ptr<column> output = make_fixed_width_column(output_type,
                                                           base_column->size(),
                                                           copy_bitmask(*base_column, stream, mr),
                                                           base_column->null_count(),
                                                           stream,
                                                           mr);

  if (base_column->is_empty()) { return output; }

  mutable_column_view const output_view = *output;

  // transform
  transformation::jit::transform_operation(
    output_view, inputs, transform_udf, output_type, is_ptx, stream, base_column->size());

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
