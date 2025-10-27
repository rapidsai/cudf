/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "jit/row_ir.hpp"

// [ ] use int32_t for transform null mask intermediate

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/reshape.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/detail/gather.cuh>
#include <cudf/table/table.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/iterator>
#include <thrust/copy.h>
#include <thrust/gather.h>

#include <jit/cache.hpp>
#include <jit/helpers.hpp>
#include <jit/parser.hpp>
#include <jit/span.cuh>
#include <jit_preprocessed_files/stream_compaction/filter/jit/kernel.cu.jit.hpp>

#include <utility>
#include <vector>

namespace cudf {

namespace {

template <typename InputIterator, typename MapIterator>
auto filter_fixed_width(InputIterator input_begin,
                        InputIterator input_end,
                        cudf::size_type num_selected,
                        MapIterator map_iterator,
                        rmm::cuda_stream_view stream,
                        rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  auto output = rmm::device_uvector<typename std::iterator_traits<InputIterator>::value_type>(
    static_cast<size_t>(num_selected), stream, mr);

  thrust::gather(rmm::exec_policy(stream),
                 map_iterator,
                 map_iterator + num_selected,
                 input_begin,
                 output.begin());

  return output;
}

template <typename MapIterator>
std::pair<rmm::device_buffer, cudf::size_type> null_mask_filter(cudf::column_device_view col,
                                                                cudf::size_type null_count,
                                                                cudf::size_type num_selected,
                                                                MapIterator map_iterator,
                                                                rmm::cuda_stream_view stream,
                                                                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  if (!col.nullable() || null_count == 0) { return std::pair(rmm::device_buffer{}, 0); }

  auto null_mask = col.null_mask();
  auto offset    = col.offset();

  auto [compacted_null_mask, compacted_null_count] = cudf::detail::valid_if(
    map_iterator,
    map_iterator + num_selected,
    [null_mask, offset] __device__(cudf::size_type i) -> bool {
      return cudf::bit_is_set(null_mask, i + offset);
    },
    stream,
    mr);

  // drop null mask if there are no nulls
  if (compacted_null_count == 0) { return std::pair(rmm::device_buffer{}, 0); }

  return std::pair(std::move(compacted_null_mask), compacted_null_count);
}

struct filter_dispatcher {
  template <typename T, typename MapIterator>
    requires(cudf::is_rep_layout_compatible<T>() && cudf::is_fixed_width<T>())
  std::unique_ptr<cudf::column> operator()(cudf::column_view col,
                                           cudf::size_type num_selected,
                                           MapIterator map_iterator,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const
  {
    CUDF_FUNC_RANGE();

    auto d_col    = cudf::column_device_view::create(col, stream);
    auto filtered = filter_fixed_width(
      d_col->data<T>(), d_col->data<T>() + d_col->size(), num_selected, map_iterator, stream, mr);
    auto filtered_size = filtered.size();
    auto [null_mask, null_count] =
      null_mask_filter(*d_col, col.null_count(), num_selected, map_iterator, stream, mr);

    auto out = cudf::column(
      d_col->type(), filtered_size, filtered.release(), std::move(null_mask), null_count, {});
    return std::make_unique<cudf::column>(std::move(out));
  }

  template <typename T, typename MapIterator>
    requires(cudf::is_fixed_point<T>())
  std::unique_ptr<cudf::column> operator()(cudf::column_view col,
                                           cudf::size_type num_selected,
                                           MapIterator map_iterator,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const
  {
    CUDF_FUNC_RANGE();

    using Rep          = typename T::rep;
    auto d_col         = cudf::column_device_view::create(col, stream);
    auto filtered      = filter_fixed_width(d_col->data<Rep>(),
                                       d_col->data<Rep>() + d_col->size(),
                                       num_selected,
                                       map_iterator,
                                       stream,
                                       mr);
    auto filtered_size = filtered.size();
    auto [null_mask, null_count] =
      null_mask_filter(*d_col, col.null_count(), num_selected, map_iterator, stream, mr);

    auto out = cudf::column(
      d_col->type(), filtered_size, filtered.release(), std::move(null_mask), null_count, {});
    return std::make_unique<cudf::column>(std::move(out));
  }

  template <typename T, typename MapIterator>
    requires(std::is_same_v<T, cudf::string_view>)
  std::unique_ptr<cudf::column> operator()(cudf::column_view col,
                                           cudf::size_type num_selected,
                                           MapIterator map_iterator,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const
  {
    CUDF_FUNC_RANGE();

    return cudf::strings::detail::gather(
      col, map_iterator, map_iterator + num_selected, false, stream, mr);
  }

  template <typename T, typename MapIterator>
  std::unique_ptr<cudf::column> operator()(cudf::column_view col,
                                           cudf::size_type num_selected,
                                           MapIterator map_iterator,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr) const
  {
    CUDF_FAIL("Unsupported column type for filter operation: " +
                cudf::type_to_name(cudf::data_type{cudf::type_to_id<T>()}),
              std::invalid_argument);
  }
};

template <typename MapIterator>
std::unique_ptr<cudf::column> filter_column(cudf::column_view col,
                                            cudf::size_type num_selected,
                                            MapIterator map_iterator,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  return cudf::type_dispatcher(
    col.type(), filter_dispatcher{}, col, num_selected, map_iterator, stream, mr);
}

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

std::tuple<rmm::device_uvector<cudf::size_type>, cudf::size_type> create_gather_map(
  column_view base_column,
  cudf::jit::device_span<bool const> filters,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  // this is a scratch buffer, it doesn't have to be exact-sized
  auto compacted_indices =
    rmm::device_uvector<cudf::size_type>{static_cast<size_t>(base_column.size()), stream, mr};
  auto selection_end = thrust::copy_if(
    rmm::exec_policy(stream),
    thrust::make_counting_iterator<cudf::size_type>(0),
    thrust::make_counting_iterator(base_column.size()),
    compacted_indices.begin(),
    [p_filters = filters.data()] __device__(cudf::size_type i) -> bool { return p_filters[i]; });
  auto num_selected =
    static_cast<cudf::size_type>(cuda::std::distance(compacted_indices.begin(), selection_end));
  return {std::move(compacted_indices), num_selected};
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

  auto [gather_map, num_selected] =
    create_gather_map(base_column, filter_bools_span.as_const(), stream, mr);

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
                     return filter_column(column, num_selected, gather_map.cbegin(), stream, mr);
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
