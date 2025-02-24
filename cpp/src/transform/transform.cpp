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
#include <cudf/jit/types.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <jit_preprocessed_files/transform/jit/kernel.cu.jit.hpp>

namespace cudf {
namespace transformation {
namespace jit {
namespace {

using scale_repr = std::underlying_type_t<numeric::scale_type>;

jitify2::StringVec build_jit_typenames(mutable_column_view output,
                                       std::vector<column_view> const& inputs)
{
  jitify2::StringVec typenames;
  int32_t index = 0;

  auto const add_column = [&](cudf::data_type data_type, bool is_scalar) {
    std::string accessor = jitify2::reflection::Template("cudf::transformation::jit::accessor")
                             .instantiate(cudf::type_to_name(data_type), index++);

    typenames.push_back(
      is_scalar
        ? jitify2::reflection::Template("cudf::transformation::jit::scalar").instantiate(accessor)
        : accessor);
  };

  add_column(output.type(), false);
  std::for_each(inputs.begin(), inputs.end(), [&](auto const& input) {
    bool const is_scalar = input.size() != output.size();
    add_column(input.type(), is_scalar);
  });

  return typenames;
}

std::map<uint32_t, std::string> build_ptx_params(mutable_column_view output,
                                                 std::vector<column_view> const& inputs)
{
  std::map<uint32_t, std::string> params;
  uint32_t index = 0;

  auto const add_column = [&](bool is_output, data_type type) {
    auto const type_name = cudf::type_to_name(type);

    if (is_output) {
      params.emplace(index++, type_name + "*");
    } else {
      params.emplace(index++, type_name);
    }
  };

  add_column(true, output.type());

  for (auto& input : inputs) {
    add_column(false, input.type());
  }

  return params;
}

cudf::jit::column_device_view to_jit_view(column_view const& view)
{
  return {const_cast<void*>(view.head()), nullptr, view.null_mask(), view.type()};
}

rmm::device_uvector<cudf::jit::column_device_view> build_views(
  mutable_column_view output,
  std::vector<column_view> const& inputs,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  std::vector<cudf::jit::column_device_view> host;

  host.emplace_back(to_jit_view(output));

  std::transform(inputs.begin(), inputs.end(), std::back_inserter(host), to_jit_view);

  rmm::device_uvector<cudf::jit::column_device_view> views{host.size(), stream, mr};

  detail::cuda_memcpy_async(cudf::device_span<cudf::jit::column_device_view>{views},
                            cudf::host_span<cudf::jit::column_device_view const>{host},
                            stream);

  return views;
}

void launch(jitify2::ConfiguredKernel& kernel,
            cudf::size_type size,
            rmm::device_uvector<cudf::jit::column_device_view>& views)
{
  // JITIFY and NVRTC need non-const pointers even if they aren't written to

  cudf::jit::column_device_view* views_ptr = views.data();

  std::array<void*, 2> args{&size, &views_ptr};

  kernel->launch(args.data());
}

jitify2::Kernel get_kernel(mutable_column_view output,
                           std::vector<column_view> const& inputs,
                           std::string const& cuda_source)
{
  std::string const kernel_name =
    jitify2::reflection::Template(cudf::is_fixed_point(output.type())
                                    ? "cudf::transformation::jit::fixed_point_kernel"
                                    : "cudf::transformation::jit::kernel")
      .instantiate(build_jit_typenames(output, inputs));

  return cudf::jit::get_program_cache(*transform_jit_kernel_cu_jit)
    .get_kernel(
      kernel_name, {}, {{"transform/jit/operation-udf.hpp", cuda_source}}, {"-arch=sm_."});
}

void transform_operation(size_type base_column_size,
                         mutable_column_view output,
                         std::vector<column_view> const& inputs,
                         std::string const& udf,
                         bool is_ptx,
                         rmm::cuda_stream_view stream,
                         rmm::device_async_resource_ref mr)
{
  std::string const cuda_source =
    is_ptx ? cudf::jit::parse_single_function_ptx(
               udf, "GENERIC_TRANSFORM_OP", build_ptx_params(output, inputs))
           : cudf::jit::parse_single_function_cuda(udf, "GENERIC_TRANSFORM_OP");

  auto views = build_views(output, inputs, stream, mr);

  auto kernel = get_kernel(output, inputs, cuda_source)
                  ->configure_1d_max_occupancy(0, 0, nullptr, stream.value());

  launch(kernel, base_column_size, views);
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
    base_column->size(), output_view, inputs, transform_udf, is_ptx, stream, mr);

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
