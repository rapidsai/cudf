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
#include <cudf/reshape.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/iterator>
#include <thrust/copy.h>

#include <jit/cache.hpp>
#include <jit/helpers.hpp>
#include <jit/parser.hpp>
#include <jit/span.cuh>
#include <jit_preprocessed_files/stream_compaction/filter/jit/kernel.cu.jit.hpp>

#include <utility>
#include <vector>

namespace cudf {

namespace {

struct filter_predicate {
  static constexpr cudf::size_type NOT_APPLIED = -1;

  constexpr __device__ bool operator()(cudf::size_type flag) const { return flag != NOT_APPLIED; }
};

/// @brief counts the number of items that are not marked as NOT_APPLIED
struct filter_histogram {
  constexpr __device__ cudf::size_type operator()(cudf::size_type flag) const
  {
    return (flag == filter_predicate::NOT_APPLIED) ? 0 : 1;
  }
};

/// @brief A gather-filter operation that filters the elements of an input range base on a
/// pre-computed stencil. The stencil serves as both a stencil and a gather map.
template <typename InputIterator>
auto filter_by(InputIterator input_begin,
               InputIterator input_end,
               cudf::size_type num_selected,
               cudf::size_type const* stencil,
               rmm::cuda_stream_view stream,
               rmm::device_async_resource_ref mr)
{
  rmm::device_uvector<typename std::iterator_traits<InputIterator>::value_type> output(
    static_cast<size_t>(num_selected), stream, mr);

  auto output_size = cuda::std::distance(output.begin(),
                                         thrust::copy_if(rmm::exec_policy(stream),
                                                         input_begin,
                                                         input_end,
                                                         stencil,
                                                         output.begin(),
                                                         filter_predicate{}));

  CUDF_EXPECTS(output_size == num_selected,
               "The number of selected items does not match the expected count.",  // < This should
                                                                                   // never happen
               std::runtime_error);

  return output;
}

template <bool result_has_nulls>
struct filter_dispatcher;

template <>
struct filter_dispatcher<false> {
  template <typename T>
    requires(cudf::is_rep_layout_compatible<T>() && cudf::is_fixed_width<T>())
  std::unique_ptr<cudf::column> operator()(cudf::column_device_view const& col,
                                           cudf::size_type num_selected,
                                           cudf::size_type const* stencil_iterator,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const
  {
    auto filtered = filter_by(
      col.data<T>(), col.data<T>() + col.size(), num_selected, stencil_iterator, stream, mr);
    auto filtered_size = filtered.size();
    auto out =
      cudf::column(col.type(), filtered_size, filtered.release(), rmm::device_buffer{}, 0, {});
    return std::make_unique<cudf::column>(std::move(out));
  }

  template <typename T>
    requires(std::is_same_v<T, cudf::string_view>)
  std::unique_ptr<cudf::column> operator()(cudf::column_device_view const& col,
                                           cudf::size_type num_selected,
                                           cudf::size_type const* stencil_iterator,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const
  {
    auto filtered = filter_by(col.begin<cudf::string_view>(),
                              col.begin<cudf::string_view>() + col.size(),
                              num_selected,
                              stencil_iterator,
                              stream,
                              mr);
    return cudf::make_strings_column(filtered, cudf::string_view{nullptr, 0}, stream, mr);
  }

  template <typename T>
  std::unique_ptr<cudf::column> operator()(cudf::column_device_view const& col,
                                           cudf::size_type num_selected,
                                           cudf::size_type const* stencil_iterator,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const
  {
    CUDF_FAIL("Unsupported column type for filter operation: " +
                cudf::type_to_name(cudf::data_type{cudf::type_to_id<T>()}),
              std::invalid_argument);
  }
};

std::unique_ptr<cudf::column> filter_column(
  cudf::column_device_view const& column,
  cudf::size_type num_selected,
  cudf::jit::device_span<cudf::size_type const> filter_indices,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  return cudf::type_dispatcher<dispatch_storage_type>(column.type(),
                                                      filter_dispatcher<false>{},
                                                      column,
                                                      num_selected,
                                                      filter_indices.begin(),
                                                      stream,
                                                      mr);
}

void launch_filter_kernel(jitify2::ConfiguredKernel& kernel,
                          cudf::jit::device_span<cudf::size_type> output,
                          std::vector<column_view> const& input_columns,
                          std::optional<void*> user_data,
                          rmm::cuda_stream_view stream,
                          rmm::device_async_resource_ref mr)
{
  auto outputs = cudf::jit::to_device_vector(
    std::vector{cudf::jit::device_optional_span<cudf::size_type>{output, nullptr}}, stream, mr);

  auto [input_handles, inputs] =
    cudf::jit::column_views_to_device<column_device_view, column_view>(input_columns, stream, mr);

  cudf::jit::device_optional_span<cudf::size_type> const* outputs_ptr = outputs.data();
  column_device_view const* inputs_ptr                                = inputs.data();
  void* p_user_data                                                   = user_data.value_or(nullptr);

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

  check_columns(predicate_columns);
  check_columns(filter_columns);
}

jitify2::Kernel get_kernel(std::string const& kernel_name, std::string const& cuda_source)
{
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
  rmm::device_uvector<cudf::size_type> filter_indices{
    static_cast<size_t>(base_column.size()), stream, mr};

  auto kernel = build_kernel("cudf::filtering::jit::kernel",
                             base_column.size(),
                             {"cudf::size_type"},
                             predicate_columns,
                             user_data.has_value(),
                             is_null_aware,
                             predicate_udf,
                             is_ptx,
                             stream,
                             mr);

  cudf::jit::device_span<cudf::size_type> const filter_indices_span{filter_indices.data(),
                                                                    filter_indices.size()};

  launch_filter_kernel(kernel, filter_indices_span, predicate_columns, user_data, stream, mr);

  auto num_selected = thrust::transform_reduce(rmm::exec_policy(stream),
                                               filter_indices.begin(),
                                               filter_indices.end(),
                                               filter_histogram{},
                                               cudf::size_type{0},
                                               std::plus<cudf::size_type>{});

  std::vector<std::unique_ptr<column>> filtered;

  std::transform(filter_columns.begin(),
                 filter_columns.end(),
                 std::back_inserter(filtered),
                 [&](auto const& column) {
                   if (cudf::jit::is_scalar(base_column.size(), column.size())) {
                     // broadcast scalar columns
                     auto tiled = cudf::tile(cudf::table_view{{column}}, num_selected, stream, mr);
                     auto tiled_columns = tiled->release();
                     return std::move(tiled_columns.front());
                   } else {
                     auto d_column = cudf::column_device_view::create(column, stream);
                     return filter_column(
                       *d_column, num_selected, filter_indices_span.as_const(), stream, mr);
                   }
                 });

  return filtered;
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
