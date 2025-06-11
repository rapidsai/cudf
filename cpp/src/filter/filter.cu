

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
#include "transform/utils.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/filter.hpp>
#include <cudf/jit/runtime_support.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cub/device/device_select.cuh>

#include <jit_preprocessed_files/filter/jit/kernel.cu.jit.hpp>

namespace cudf {

namespace {
template <typename IndexType>
void launch_filter_kernel(jitify2::ConfiguredKernel& kernel,
                          cudf::jit::filter_output<IndexType> output,
                          std::vector<column_view> const& input_cols,
                          std::optional<void*> user_data,
                          rmm::cuda_stream_view stream,
                          rmm::device_async_resource_ref mr)
{
  auto outputs = to_device_vector(std::vector{output}, stream, mr);

  auto [input_handles, inputs] =
    cudf::transformation::column_views_to_device<column_device_view, column_view>(
      input_cols, stream, mr);

  cudf::jit::filter_indices<IndexType> const* outputs_ptr = outputs.data();
  column_device_view const* inputs_ptr                    = inputs.data();
  void* p_user_data                                       = user_data.value_or(nullptr);

  std::array<void*, 3> args{&outputs_ptr, &inputs_ptr, &p_user_data};

  kernel->launch(args.data());
}

void perform_checks(column_view base_column,
                    data_type output_type,
                    std::vector<column_view> const& target_columns,
                    std::vector<column_view> const& predicate_columns)
{
  CUDF_EXPECTS(is_runtime_jit_supported(), "Runtime JIT is only supported on CUDA Runtime 11.5+");
  CUDF_EXPECTS(is_fixed_width(output_type) || output_type.id() == type_id::STRING,
               "Filters only support output of fixed-width or string types",
               std::invalid_argument);
  CUDF_EXPECTS(std::all_of(inputs.begin(),
                           inputs.end(),
                           [](auto& input) {
                             return is_fixed_width(input.type()) ||
                                    (input.type().id() == type_id::STRING);
                           }),
               "Filters only support input of fixed-width or string types",
               std::invalid_argument);

  CUDF_EXPECTS(std::all_of(inputs.begin(),
                           inputs.end(),
                           [&](auto const& input) {
                             return (input.size() == 1) || (input.size() == base_column.size());
                           }),
               "All transform input columns must have the same size or be scalar (have size 1)",
               std::invalid_argument);
}

jitify2::Kernel get_kernel(std::string const& kernel_name, std::string const& cuda_source)
{
  return cudf::jit::get_program_cache(*filter_jit_kernel_cu_jit)
    .get_kernel(kernel_name,
                {},
                {{"filter/jit/operation-udf.hpp", cuda_source}},
                {"-arch=sm_.",
                 "--device-int128",
                 // TODO: remove when we upgrade to CCCL >= 3.0

                 // CCCL WAR for not using the correct INT128 feature macro:
                 // https://github.com/NVIDIA/cccl/issues/3801
                 "-D__SIZEOF_INT128__=16"});
}

jitify2::ConfiguredKernel build_kernel(std::string const& kernel_name,
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
    is_ptx
      ? cudf::jit::parse_single_function_ptx(
          udf,
          "GENERIC_TRANSFORM_OP",
          cudf::transformation::build_ptx_params(
            span_outputs, cudf::transformation::column_type_names(input_columns), has_user_data))
      : cudf::jit::parse_single_function_cuda(udf, "GENERIC_TRANSFORM_OP");

  return get_kernel(
           jitify2::reflection::Template(kernel_name)
             .instantiate(build_jit_template_params(
               has_user_data,
               span_outputs,
               {},
               cudf::transformation::reflect_input_columns(base_column_size, input_columns))),
           cuda_source)
    ->configure_1d_max_occupancy(0, 0, nullptr, stream.value());
}

std::vector<std::unique_ptr<column>> filter_operation(
  column_view base_column,
  data_type output_type,
  std::vector<column_view> const& target_columns,
  std::vector<column_view> const& predicate_columns,
  std::string const& udf,
  bool is_ptx,
  std::optional<void*> user_data,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  rmm::device_uvector<cudf::size_type> filter_indices{
    static_cast<size_t>(base_column.size()), stream, mr};
  rmm::device_scalar<cudf::size_type> not_applied_count{0, stream, mr};

  auto kernel = build_kernel("cudf::jit::filter_kernel",
                             base_column.size(),
                             {"cudf::size_type"},
                             predicate_columns,
                             user_data.has_value(),
                             udf,
                             is_ptx,
                             stream,
                             mr);

  cudf::jit::device_span<cudf::size_type> indices_view{filter_indices.data(),
                                                       filter_indices.size()};

  cudf::jit::filter_output<cudf::size_type> filter_output{indices_view, not_applied_count.data()};

  launch_filter_kernel<cudf::size_type>(
    kernel, filter_output, predicate_columns, user_data, stream, mr);

  // [ ] fixed-width flagged-if cub::deviceselect::flaggedif. fixed-width dispatchers. create size -
  // na-sized column.

  std::vector<std::unique_ptr<column>> output;

  return output;
}

}  // namespace

namespace detail {

std::vector<std::unique_ptr<column>> filter(std::vector<column_view> const& target_columns,
                                            std::vector<column_view> const& predicate_columns,
                                            std::string const& predicate_udf,
                                            bool is_ptx,
                                            std::optional<void*> user_data,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(
    !target_columns.empty(), "Filters must have at least 1 target column", std::invalid_argument);
  CUDF_EXPECTS(!predicate_columns.empty(),
               "Filters must have at least 1 predicate column",
               std::invalid_argument);

  auto const base_column = std::max_element(predicate_columns.begin(),
                                            predicate_columns.end(),
                                            [](auto& a, auto& b) { return a.size() < b.size(); });

  CUDF_EXPECTS(
    std::all_of(target_columns.begin(),
                target_columns.end(),
                [&](column_view const& col) { return col.size() == base_column->size(); }),
    "Filter's target columns must have the same size as the predicate base column",
    std::invalid_argument);

  auto const scatter_type = cudf::data_type{cudf::type_to_id<cudf::size_type>()};
  rmm::device_scalar<cudf::size_type> not_applied_count{0, stream, mr};

  perform_checks(*base_column, scatter_type, target_columns, predicate_columns);

  auto filtered = filter_operation(*base_column,
                                   scatter_type,
                                   target_columns,
                                   predicate_columns,
                                   predicate_udf,
                                   is_ptx,
                                   user_data,
                                   stream,
                                   mr);

  return filtered;
}

}  // namespace detail

std::vector<std::unique_ptr<column>> filter(std::vector<column_view> const& target_columns,
                                            std::vector<column_view> const& predicate_columns,
                                            std::string const& predicate_udf,
                                            bool is_ptx,
                                            std::optional<void*> user_data,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::filter(
    target_columns, predicate_columns, predicate_udf, is_ptx, user_data, stream, mr);
}

}  // namespace cudf
