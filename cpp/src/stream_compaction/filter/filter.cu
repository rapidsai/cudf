/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "jit/row_ir.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <jit/cache.hpp>
#include <jit/helpers.hpp>
#include <jit/parser.hpp>
#include <jit/span.cuh>
#include <jit_preprocessed_files/stream_compaction/filter/jit/kernel.cu.jit.hpp>

#include <utility>
#include <vector>

namespace cudf {

namespace {

void launch_filter_kernel(jitify2::ConfiguredKernel& kernel,
                          cudf::jit::device_span<bool> output,
                          std::vector<column_view> const& input_columns,
                          std::optional<void*> user_data,
                          rmm::cuda_stream_view stream,
                          rmm::device_async_resource_ref mr)
{
  auto outputs = cudf::jit::to_device_vector(
    std::vector{cudf::jit::device_optional_span<bool>{output, nullptr}}, stream, mr);

  auto [input_handles, inputs] =
    cudf::jit::column_views_to_device<column_device_view, column_view>(input_columns, stream, mr);

  cudf::jit::device_optional_span<bool> const* outputs_ptr = outputs.data();
  column_device_view const* inputs_ptr                     = inputs.data();
  void* p_user_data                                        = user_data.value_or(nullptr);

  std::array<void*, 3> args{&outputs_ptr, &inputs_ptr, &p_user_data};

  kernel->launch_raw(args.data());
}

void perform_checks(column_view base_column,
                    std::vector<column_view> const& predicate_columns,
                    std::vector<column_view> const& filter_columns)
{
  auto check_columns = [&](std::vector<column_view> const& columns) {
    CUDF_EXPECTS(std::all_of(columns.begin(),
                             columns.end(),
                             [](auto& input) {
                               return is_fixed_width(input.type()) ||
                                      (input.type().id() == type_id::STRING);
                             }),
                 "Filters only support fixed-width and string types",
                 std::invalid_argument);

    CUDF_EXPECTS(std::all_of(columns.begin(),
                             columns.end(),
                             [&](auto& input) {
                               return cudf::jit::is_scalar(base_column.size(), input.size()) ||
                                      (input.size() == base_column.size());
                             }),
                 "All filter columns must have the same size or be scalar (have size 1)",
                 std::invalid_argument);
  };

  for (auto const& col : filter_columns) {
    CUDF_EXPECTS(col.size() == base_column.size(),
                 "All filter columns must have the same size",
                 std::invalid_argument);
  }

  check_columns(predicate_columns);
  check_columns(filter_columns);
}

jitify2::Kernel get_kernel(std::string const& kernel_name, std::string const& cuda_source)
{
  CUDF_FUNC_RANGE();

  return cudf::jit::get_program_cache(*stream_compaction_filter_jit_kernel_cu_jit)
    .get_kernel(kernel_name, {}, {{"cudf/detail/operation-udf.hpp", cuda_source}}, {"-arch=sm_."});
}

jitify2::ConfiguredKernel build_kernel(std::string const& kernel_name,
                                       size_type base_column_size,
                                       std::vector<std::string> const& span_outputs,
                                       std::vector<column_view> const& input_columns,
                                       bool has_user_data,
                                       null_aware is_null_aware,
                                       std::string const& udf,
                                       bool is_ptx,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  CUDF_EXPECTS(!(is_null_aware == null_aware::YES && is_ptx),
               "Optional types are not supported in PTX UDFs",
               std::invalid_argument);
  auto const cuda_source =
    is_ptx ? cudf::jit::parse_single_function_ptx(
               udf,
               "GENERIC_FILTER_OP",
               cudf::jit::build_ptx_params(
                 span_outputs, cudf::jit::column_type_names(input_columns), has_user_data))
           : cudf::jit::parse_single_function_cuda(udf, "GENERIC_FILTER_OP");

  return get_kernel(jitify2::reflection::Template(kernel_name)
                      .instantiate(cudf::jit::build_jit_template_params(
                        has_user_data,
                        is_null_aware,
                        span_outputs,
                        {},
                        cudf::jit::reflect_input_columns(base_column_size, input_columns))),
                    cuda_source)
    ->configure_1d_max_occupancy(0, 0, nullptr, stream.value());
}

std::vector<std::unique_ptr<column>> filter_operation(
  column_view base_column,
  std::vector<column_view> const& predicate_columns,
  std::string const& predicate_udf,
  std::vector<column_view> const& filter_columns,
  bool is_ptx,
  std::optional<void*> user_data,
  null_aware is_null_aware,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto filter_bools =
    rmm::device_uvector<uint8_t>{static_cast<size_t>(base_column.size()), stream, mr};

  auto kernel = build_kernel("cudf::filtering::jit::kernel",
                             base_column.size(),
                             {"bool"},
                             predicate_columns,
                             user_data.has_value(),
                             is_null_aware,
                             predicate_udf,
                             is_ptx,
                             stream,
                             mr);

  auto filter_bools_span =
    cudf::jit::device_span<bool>{reinterpret_cast<bool*>(filter_bools.data()), filter_bools.size()};

  launch_filter_kernel(kernel, filter_bools_span, predicate_columns, user_data, stream, mr);

  return apply_boolean_mask(
           cudf::table_view{filter_columns},
           cudf::column_view{cudf::data_type{type_id::BOOL8},
                             static_cast<cudf::size_type>(filter_bools_span.size()),
                             filter_bools.data(),
                             nullptr,
                             0},
           stream,
           mr)
    ->release();
}

}  // namespace

namespace detail {
std::vector<std::unique_ptr<column>> filter(std::vector<column_view> const& predicate_columns,
                                            std::string const& predicate_udf,
                                            std::vector<column_view> const& filter_columns,
                                            bool is_ptx,
                                            std::optional<void*> user_data,
                                            null_aware is_null_aware,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(
    !predicate_columns.empty(), "Filters must have at least 1 column", std::invalid_argument);
  CUDF_EXPECTS(
    !filter_columns.empty(), "Filters must have at least 1 column", std::invalid_argument);

  auto const base_column = cudf::jit::get_transform_base_column(predicate_columns);

  perform_checks(*base_column, predicate_columns, filter_columns);

  auto filtered = filter_operation(*base_column,
                                   predicate_columns,
                                   predicate_udf,
                                   filter_columns,
                                   is_ptx,
                                   user_data,
                                   is_null_aware,
                                   stream,
                                   mr);

  return filtered;
}

std::unique_ptr<table> filter(table_view const& predicate_table,
                              ast::expression const& predicate_expr,
                              table_view const& filter_table,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  cudf::detail::row_ir::ast_args ast_args{.table = predicate_table};
  auto args = cudf::detail::row_ir::ast_converter::filter(
    cudf::detail::row_ir::target::CUDA, predicate_expr, ast_args, filter_table, stream, mr);

  return std::make_unique<table>(cudf::detail::filter(args.predicate_columns,
                                                      args.predicate_udf,
                                                      args.filter_columns,
                                                      args.is_ptx,
                                                      args.user_data,
                                                      args.is_null_aware,
                                                      stream,
                                                      mr));
}

}  // namespace detail

std::vector<std::unique_ptr<column>> filter(std::vector<column_view> const& predicate_columns,
                                            std::string const& predicate_udf,
                                            std::vector<column_view> const& filter_columns,
                                            bool is_ptx,
                                            std::optional<void*> user_data,
                                            null_aware is_null_aware,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::filter(
    predicate_columns, predicate_udf, filter_columns, is_ptx, user_data, is_null_aware, stream, mr);
}

std::unique_ptr<table> filter(table_view const& predicate_table,
                              ast::expression const& predicate_expr,
                              table_view const& filter_table,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::filter(predicate_table, predicate_expr, filter_table, stream, mr);
}

}  // namespace cudf
