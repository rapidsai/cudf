/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "jit/row_ir.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/stream_compaction.hpp>

namespace cudf {

namespace detail {
std::vector<std::unique_ptr<column>> filter(
  std::vector<std::variant<column_view, scalar_column_view>> const& predicate_inputs,
  std::string const& predicate_udf,
  std::vector<column_view> const& filter_columns,
  bool is_ptx,
  std::optional<void*> user_data,
  null_aware is_null_aware,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto predicate = cudf::transform(predicate_inputs, predicate_udf, is_ptx, user_data, stream, mr);

  return apply_boolean_mask(cudf::table_view{filter_columns}, predicate->view(), stream, mr)
    ->release();
}

}  // namespace detail

std::unique_ptr<table> filter(table_view const& predicate_table,
                              ast::expression const& predicate_expr,
                              table_view const& filter_table,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  cudf::detail::row_ir::ast_args ast_args{.table = predicate_table};
  auto args = cudf::detail::row_ir::ast_converter::filter(
    cudf::detail::row_ir::target::CUDA, predicate_expr, ast_args, filter_table, stream, mr);

  return std::make_unique<table>(cudf::detail::filter(args.predicate_inputs,
                                                      args.predicate_udf,
                                                      args.filter_columns,
                                                      args.is_ptx,
                                                      args.user_data,
                                                      args.is_null_aware,
                                                      stream,
                                                      mr));
}

std::vector<std::unique_ptr<column>> filter(
  std::vector<std::variant<column_view, scalar_column_view>> const& predicate_inputs,
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
    predicate_inputs, predicate_udf, filter_columns, is_ptx, user_data, is_null_aware, stream, mr);
}

std::vector<std::unique_ptr<column>> filter(std::vector<column_view> const& predicate_columns,
                                            std::string const& predicate_udf,
                                            std::vector<column_view> const& filter_columns,
                                            bool is_ptx,
                                            std::optional<void*> user_data,
                                            null_aware is_null_aware,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  // TODO: preserve legacy behaviour
}

}  // namespace cudf
