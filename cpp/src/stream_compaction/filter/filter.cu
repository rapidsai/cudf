/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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
#include "jit/helpers.hpp"
#include "jit/parser.hpp"
#include "jit/span.cuh"

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

  kernel->launch(args.data());
}

void perform_checks(column_view base_column,
                    std::vector<column_view> const& columns,
                    std::optional<std::vector<bool>> copy_mask)
{
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

  CUDF_EXPECTS(!copy_mask.has_value() || (copy_mask->size() == columns.size()),
               "The size of the copy mask must match the number of input columns",
               std::invalid_argument);
}

jitify2::Kernel get_kernel(std::string const& kernel_name, std::string const& cuda_source)
{
  return cudf::jit::get_program_cache(*stream_compaction_filter_jit_kernel_cu_jit)
    .get_kernel(kernel_name,
                {},
                {{"cudf/detail/operation-udf.hpp", cuda_source}},
                {"-arch=sm_.", "--device-int128"});
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
    is_ptx ? cudf::jit::parse_single_function_ptx(
               udf,
               "GENERIC_FILTER_OP",
               cudf::jit::build_ptx_params(
                 span_outputs, cudf::jit::column_type_names(input_columns), has_user_data))
           : cudf::jit::parse_single_function_cuda(udf, "GENERIC_FILTER_OP");

  return get_kernel(jitify2::reflection::Template(kernel_name)
                      .instantiate(cudf::jit::build_jit_template_params(
                        has_user_data,
                        span_outputs,
                        {},
                        cudf::jit::reflect_input_columns(base_column_size, input_columns))),
                    cuda_source)
    ->configure_1d_max_occupancy(0, 0, nullptr, stream.value());
}

std::vector<std::unique_ptr<column>> filter_operation(column_view base_column,
                                                      std::vector<column_view> const& columns,
                                                      std::string const& predicate_udf,
                                                      bool is_ptx,
                                                      std::optional<void*> user_data,
                                                      std::optional<std::vector<bool>> copy_mask,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr)
{
  rmm::device_uvector<cudf::size_type> filter_indices{
    static_cast<size_t>(base_column.size()), stream, mr};

  auto kernel = build_kernel("cudf::filtering::jit::kernel",
                             base_column.size(),
                             {"cudf::size_type"},
                             columns,
                             user_data.has_value(),
                             predicate_udf,
                             is_ptx,
                             stream,
                             mr);

  cudf::jit::device_span<cudf::size_type> const filter_indices_span{filter_indices.data(),
                                                                    filter_indices.size()};

  launch_filter_kernel(kernel, filter_indices_span, columns, user_data, stream, mr);

  auto num_selected = thrust::transform_reduce(rmm::exec_policy(stream),
                                               filter_indices.begin(),
                                               filter_indices.end(),
                                               filter_histogram{},
                                               cudf::size_type{0},
                                               std::plus<cudf::size_type>{});

  std::vector<std::unique_ptr<column>> filtered;

  std::for_each(thrust::counting_iterator<size_t>(0),
                thrust::counting_iterator<size_t>(columns.size()),
                [&](auto i) {
                  auto column      = columns[i];
                  auto should_copy = !copy_mask.has_value() || (*copy_mask)[i];

                  if (should_copy) {
                    if (cudf::jit::is_scalar(base_column.size(), column.size())) {
                      // broadcast scalar columns
                      auto tiled = cudf::tile(cudf::table_view{{column}}, num_selected, stream, mr);
                      auto tiled_columns = tiled->release();
                      filtered.push_back(std::move(tiled_columns[0]));
                    } else {
                      auto d_column        = cudf::column_device_view::create(column, stream);
                      auto filtered_column = filter_column(
                        *d_column, num_selected, filter_indices_span.as_const(), stream, mr);
                      filtered.push_back(std::move(filtered_column));
                    }
                  }
                });

  return filtered;
}

}  // namespace

namespace detail {
std::vector<std::unique_ptr<column>> filter(std::vector<column_view> const& columns,
                                            std::string const& predicate_udf,
                                            bool is_ptx,
                                            std::optional<void*> user_data,
                                            std::optional<std::vector<bool>> copy_mask,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(!columns.empty(), "Filters must have at least 1 column", std::invalid_argument);

  auto const base_column = std::max_element(
    columns.begin(), columns.end(), [](auto& a, auto& b) { return a.size() < b.size(); });

  perform_checks(*base_column, columns, copy_mask);

  auto filtered = filter_operation(
    *base_column, columns, predicate_udf, is_ptx, user_data, copy_mask, stream, mr);

  return filtered;
}

}  // namespace detail

std::vector<std::unique_ptr<column>> filter(std::vector<column_view> const& columns,
                                            std::string const& predicate_udf,
                                            bool is_ptx,
                                            std::optional<void*> user_data,
                                            std::optional<std::vector<bool>> copy_mask,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::filter(columns, predicate_udf, is_ptx, user_data, copy_mask, stream, mr);
}

}  // namespace cudf
