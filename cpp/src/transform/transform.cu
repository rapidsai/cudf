/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <jit/cache.hpp>
#include <jit/helpers.hpp>
#include <jit/parser.hpp>
#include <jit/row_ir.hpp>
#include <jit/span.cuh>
#include <jit/util.hpp>
#include <jit_preprocessed_files/jit/lto/library.cuh.jit.hpp>

namespace cudf {
namespace transformation {
namespace jit {
namespace {

jitify2::Kernel get_kernel(std::string const& kernel_name, std::string const& cuda_source)
{
  CUDF_FUNC_RANGE();
  return cudf::jit::get_program_cache(*transform_jit_kernel_cu_jit)
    .get_kernel(kernel_name, {}, {{"cudf/detail/operation-udf.hpp", cuda_source}}, {"-arch=sm_."});
}

// [ ] load the cudf transformation LTO library; return Culibrary object representing it
// void get_transform_library();

// [ ] take CUDA/PTX code as input
// [ ] support different function signature styles
// [ ] handle include directories
// [ ] C++/C-ABI symbol name for locating function to link against
// [ ] take input and output data types as parameters
// [ ] take function_info struct as input; describe: input/output types, null-awareness, user-data
// requirement, etc.
// [ ] return Culibrary object representing compiled operator
// void compile_library();
// void compile_transform_operator_thunk();

// [ ] take LTO-IR compiled binary of operator as input
// [ ] link it into the lto library; return
// void link_transform_thunk();

// [ ] if we can, we should not depend on jitify for LTO-IR compilation, linking, and caching; use
// nvJITLink directly
// void cache_transform_thunk();

jitify2::StringVec build_jit_template_params(
  null_aware is_null_aware,
  bool may_evaluate_null,
  bool has_user_data,
  std::vector<std::string> const& span_outputs,
  std::vector<std::string> const& column_outputs,
  std::vector<cudf::jit::input_column_reflection> const& column_inputs)
{
  jitify2::StringVec tparams;

  tparams.emplace_back(jitify2::reflection::reflect(is_null_aware));
  tparams.emplace_back(jitify2::reflection::reflect(may_evaluate_null));
  tparams.emplace_back(jitify2::reflection::reflect(has_user_data));

  std::transform(thrust::counting_iterator<size_t>(0),
                 thrust::counting_iterator(span_outputs.size()),
                 std::back_inserter(tparams),
                 [&](auto i) {
                   return jitify2::reflection::Template("cudf::jit::span_accessor")
                     .instantiate(span_outputs[i], i);
                 });

  std::transform(thrust::counting_iterator<size_t>(0),
                 thrust::counting_iterator(column_outputs.size()),
                 std::back_inserter(tparams),
                 [&](auto i) {
                   return jitify2::reflection::Template("cudf::jit::column_accessor")
                     .instantiate(column_outputs[i], i);
                 });

  std::transform(thrust::counting_iterator<size_t>(0),
                 thrust::counting_iterator(column_inputs.size()),
                 std::back_inserter(tparams),
                 [&](auto i) { return column_inputs[i].accessor(i); });

  return tparams;
}

jitify2::ConfiguredKernel build_transform_kernel(
  std::string const& kernel_name,
  size_type base_column_size,
  std::vector<mutable_column_view> const& output_columns,
  std::vector<column_view> const& input_columns,
  null_aware is_null_aware,
  bool may_evaluate_null,
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
               cudf::jit::build_ptx_params(cudf::jit::column_type_names(output_columns),
                                           cudf::jit::column_type_names(input_columns),
                                           has_user_data))
           : cudf::jit::parse_single_function_cuda(udf, "GENERIC_TRANSFORM_OP");

  return get_kernel(jitify2::reflection::Template(kernel_name)
                      .instantiate(build_jit_template_params(
                        is_null_aware,
                        may_evaluate_null,
                        has_user_data,
                        {},
                        cudf::jit::column_type_names(output_columns),
                        cudf::jit::reflect_input_columns(base_column_size, input_columns))),
                    cuda_source)
    ->configure_1d_max_occupancy(0, 0, nullptr, stream.value());
}

jitify2::ConfiguredKernel build_span_kernel(std::string const& kernel_name,
                                            size_type base_column_size,
                                            std::vector<std::string> const& span_outputs,
                                            std::vector<column_view> const& input_columns,
                                            null_aware is_null_aware,
                                            bool may_evaluate_null,
                                            bool has_user_data,
                                            std::string const& udf,
                                            bool is_ptx,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto const cuda_source =
    is_ptx ? cudf::jit::parse_single_function_ptx(
               udf,
               "GENERIC_TRANSFORM_OP",
               cudf::jit::build_ptx_params(
                 span_outputs, cudf::jit::column_type_names(input_columns), has_user_data))
           : cudf::jit::parse_single_function_cuda(udf, "GENERIC_TRANSFORM_OP");

  return get_kernel(jitify2::reflection::Template(kernel_name)
                      .instantiate(build_jit_template_params(
                        is_null_aware,
                        may_evaluate_null,
                        has_user_data,
                        span_outputs,
                        {},
                        cudf::jit::reflect_input_columns(base_column_size, input_columns))),
                    cuda_source)
    ->configure_1d_max_occupancy(0, 0, nullptr, stream.value());
}

void launch_column_output_kernel(jitify2::ConfiguredKernel& kernel,
                                 std::vector<mutable_column_view> const& output_columns,
                                 std::vector<column_view> const& input_columns,
                                 std::optional<bool*> intermediate_null_mask,
                                 std::optional<void*> user_data,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  auto [output_handles, outputs] =
    cudf::jit::column_views_to_device<mutable_column_device_view, mutable_column_view>(
      output_columns, stream, mr);
  auto [input_handles, inputs] =
    cudf::jit::column_views_to_device<column_device_view, column_view>(input_columns, stream, mr);

  mutable_column_device_view const* p_outputs = outputs.data();
  column_device_view const* p_inputs          = inputs.data();
  bool* p_intermediate_null_mask              = intermediate_null_mask.value_or(nullptr);
  void* p_user_data                           = user_data.value_or(nullptr);

  std::array<void*, 4> args{&p_outputs, &p_inputs, &p_intermediate_null_mask, &p_user_data};

  kernel->launch_raw(args.data());
}

template <typename T>
void launch_span_kernel(jitify2::ConfiguredKernel& kernel,
                        cudf::jit::device_optional_span<T> const& output,
                        std::vector<column_view> const& input_cols,
                        std::optional<bool*> intermediate_null_mask,
                        std::optional<void*> user_data,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr)
{
  auto outputs = cudf::jit::to_device_vector(std::vector{output}, stream, mr);

  auto [input_handles, inputs] =
    cudf::jit::column_views_to_device<column_device_view, column_view>(input_cols, stream, mr);

  cudf::jit::device_optional_span<T> const* p_outputs = outputs.data();
  column_device_view const* p_inputs                  = inputs.data();
  bool* p_intermediate_null_mask                      = intermediate_null_mask.value_or(nullptr);
  void* p_user_data                                   = user_data.value_or(nullptr);

  std::array<void*, 4> args{&p_outputs, &p_inputs, &p_intermediate_null_mask, &p_user_data};

  kernel->launch_raw(args.data());
}

std::tuple<rmm::device_buffer, size_type> and_null_mask(column_view base_column,
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
    if (cudf::jit::is_scalar(base_column.size(), col.size())) {
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

bool may_evaluate_null(column_view base_column,
                       std::vector<column_view> const& inputs,
                       null_aware is_null_aware,
                       output_nullability null_out)
{
  // null-aware UDFs will evaluate nulls unless explicitly marked as not producing nulls
  if (is_null_aware == null_aware::YES) {
    return null_out != output_nullability::ALL_VALID;
  } else {
    /// null-unaware UDFs will evaluate nulls if any input is nullable unless explicitly marked
    /// as not producing nulls
    bool any_nullable =
      std::any_of(inputs.begin(), inputs.end(), [](auto const& col) { return col.nullable(); });
    return any_nullable && null_out == output_nullability::PRESERVE;
  }
}

std::unique_ptr<column> transform_operation(column_view base_column,
                                            data_type output_type,
                                            std::vector<column_view> const& inputs,
                                            std::string const& udf,
                                            bool is_ptx,
                                            std::optional<void*> user_data,
                                            null_aware is_null_aware,
                                            output_nullability null_policy,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  auto output = make_fixed_width_column(
    output_type, base_column.size(), cudf::mask_state::UNALLOCATED, stream, mr);

  auto may_return_nulls = may_evaluate_null(base_column, inputs, is_null_aware, null_policy);

  std::optional<rmm::device_uvector<bool>> intermediate_null_mask = std::nullopt;

  if (is_null_aware == null_aware::NO) {
    if (may_return_nulls) {
      auto [and_mask_buffer, and_mask_null_count] = and_null_mask(base_column, inputs, stream, mr);
      output->set_null_mask(std::move(and_mask_buffer), and_mask_null_count);
    }
  } else if (is_null_aware == null_aware::YES) {
    if (may_return_nulls) {
      intermediate_null_mask =
        std::make_optional<rmm::device_uvector<bool>>(base_column.size(), stream, mr);
    }
  }

  auto kernel = build_transform_kernel(is_fixed_point(output_type)
                                         ? "cudf::transformation::jit::fixed_point_kernel"
                                         : "cudf::transformation::jit::kernel",
                                       base_column.size(),
                                       {*output},
                                       inputs,
                                       is_null_aware,
                                       may_return_nulls,
                                       user_data.has_value(),
                                       udf,
                                       is_ptx,
                                       stream,
                                       mr);

  launch_column_output_kernel(kernel,
                              {*output},
                              inputs,
                              intermediate_null_mask.has_value()
                                ? std::optional<bool*>(intermediate_null_mask->data())
                                : std::nullopt,
                              user_data,
                              stream,
                              mr);

  if (intermediate_null_mask.has_value()) {
    auto [null_mask, null_count] = detail::valid_if(
      intermediate_null_mask->begin(),
      intermediate_null_mask->end(),
      [] __device__(bool element) { return element; },
      stream,
      mr);

    output->set_null_mask(std::move(null_mask), null_count);
  }

  return output;
}

std::unique_ptr<column> string_view_operation(column_view base_column,
                                              std::vector<column_view> const& inputs,
                                              std::string const& udf,
                                              bool is_ptx,
                                              std::optional<void*> user_data,
                                              null_aware is_null_aware,
                                              output_nullability null_policy,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  rmm::device_uvector<string_view> string_views(base_column.size(), stream, mr);

  auto may_return_nulls = may_evaluate_null(base_column, inputs, is_null_aware, null_policy);

  std::optional<rmm::device_uvector<bool>> intermediate_null_mask   = std::nullopt;
  std::optional<std::tuple<rmm::device_buffer, size_type>> and_mask = std::nullopt;

  if (is_null_aware == null_aware::NO) {
    if (may_return_nulls) { and_mask = and_null_mask(base_column, inputs, stream, mr); }
  } else if (is_null_aware == null_aware::YES) {
    if (may_return_nulls) {
      intermediate_null_mask =
        std::make_optional<rmm::device_uvector<bool>>(base_column.size(), stream, mr);
    }
  }

  auto output_span = cudf::jit::device_optional_span<string_view>{
    cudf::jit::device_span<string_view>{string_views.data(), string_views.size()},
    and_mask.has_value() ? static_cast<bitmask_type*>(std::get<0>(*and_mask).data()) : nullptr};

  auto kernel = build_span_kernel("cudf::transformation::jit::span_kernel",
                                  base_column.size(),
                                  {"cudf::string_view"},
                                  inputs,
                                  is_null_aware,
                                  may_return_nulls,
                                  user_data.has_value(),
                                  udf,
                                  is_ptx,
                                  stream,
                                  mr);

  launch_span_kernel<string_view>(kernel,
                                  output_span,
                                  inputs,
                                  intermediate_null_mask.has_value()
                                    ? std::optional<bool*>(intermediate_null_mask->data())
                                    : std::nullopt,
                                  user_data,
                                  stream,
                                  mr);

  auto output = make_strings_column(string_views, cudf::string_view{}, stream, mr);

  if (and_mask.has_value()) {
    auto [and_mask_buffer, and_mask_null_count] = std::move(*and_mask);
    output->set_null_mask(std::move(and_mask_buffer), and_mask_null_count);
  } else if (intermediate_null_mask.has_value()) {
    auto [null_mask, null_count] = detail::valid_if(
      intermediate_null_mask->begin(),
      intermediate_null_mask->end(),
      [] __device__(bool element) { return element; },
      stream,
      mr);

    output->set_null_mask(std::move(null_mask), null_count);
  }

  return output;
}

void perform_checks(column_view base_column,
                    data_type output_type,
                    std::vector<column_view> const& inputs)
{
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
                                  null_aware is_null_aware,
                                  output_nullability null_policy,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(
    !inputs.empty(), "Transform must have at least 1 input column", std::invalid_argument);
  CUDF_EXPECTS(!(is_null_aware == null_aware::YES && is_ptx),
               "Optional types are not supported in PTX UDFs",
               std::invalid_argument);

  auto const base_column = cudf::jit::get_transform_base_column(inputs);

  transformation::jit::perform_checks(*base_column, output_type, inputs);

  if (is_fixed_width(output_type)) {
    return transformation::jit::transform_operation(*base_column,
                                                    output_type,
                                                    inputs,
                                                    udf,
                                                    is_ptx,
                                                    user_data,
                                                    is_null_aware,
                                                    null_policy,
                                                    stream,
                                                    mr);
  } else if (output_type.id() == type_id::STRING) {
    return transformation::jit::string_view_operation(
      *base_column, inputs, udf, is_ptx, user_data, is_null_aware, null_policy, stream, mr);
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
                                  null_aware is_null_aware,
                                  output_nullability null_policy,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::transform(
    inputs, udf, output_type, is_ptx, user_data, is_null_aware, null_policy, stream, mr);
}

std::unique_ptr<column> compute_column_jit(table_view const& table,
                                           ast::expression const& expr,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  cudf::detail::row_ir::ast_args ast_args{.table = table};
  auto args = cudf::detail::row_ir::ast_converter::compute_column(
    cudf::detail::row_ir::target::CUDA, expr, ast_args, stream, mr);

  return cudf::transform(args.columns,
                         args.udf,
                         args.output_type,
                         args.is_ptx,
                         args.user_data,
                         args.is_null_aware,
                         args.null_policy,
                         stream,
                         mr);
}

}  // namespace cudf
