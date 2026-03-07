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
#include <jit_preprocessed_files/transform/jit/kernel.cu.jit.hpp>

#include <span>
#include <variant>

namespace cudf {
namespace {

// TODO: update parameter names and reflection
template <typename T>
struct mutable_vector_column_view {
  T* _data{nullptr};
  size_type _size{0};
  bitmask_type const* _null_mask{nullptr};
  size_type _offset{0};
  size_type _null_count{0};

  auto to_device() const
  {
    auto deleter = +[](mutable_column_device_view* ptr) { delete ptr; };
    return std::unique_ptr<mutable_column_device_view,
                           std::function<void(mutable_column_device_view*)>>(
      new mutable_column_device_view{mutable_column_device_view::from_parts(
        data_type{type_id::EMPTY}, _size, _data, _null_mask, _offset, nullptr, 0)},
      deleter);
  }
};

template <typename T>
struct vector_column {
  using mutable_view_type = mutable_vector_column_view<T>;

  rmm::device_uvector<T> _data{};
  rmm::device_buffer _null_mask{};
  size_type _offset{0};
  size_type _null_count{0};

  static auto make(size_type size,
                   null_mask_state null_state,
                   rmm::cuda_stream_view stream,
                   rmm::device_async_resource_ref mr)
  {
    rmm::device_uvector<T> data{size, stream, mr};
    auto null_mask = create_null_mask(size, null_state, stream, mr);

    return vector_column<T>{std::move(data), std::move(null_mask), 0, size, 0};
  }

  auto mutable_view()
  {
    return mutable_view_type{_data.data(),
                             _data.size(),
                             static_cast<bitmask_type*>(_null_mask.data()),
                             _offset,
                             _null_count};
  }
};

using string_view_column              = vector_column<string_view>;
using mutable_string_view_column_view = typename string_view_column::mutable_view_type;

using OutputColumn = std::variant<std::unique_ptr<column>, std::unique_ptr<string_view_column>>;
using InputView    = std::variant<column_view, scalar_column_view>;
using OutputView   = std::variant<mutable_column_view, mutable_string_view_column_view>;
using InputViews   = std::span<InputView const>;
using OutputViews  = std::span<OutputView const>;
using InHandle     = std::unique_ptr<column_device_view, std::function<void(column_device_view*)>>;
using OutHandle =
  std::unique_ptr<mutable_column_device_view, std::function<void(mutable_column_device_view*)>>;
using Handle = std::variant<InHandle, OutHandle>;

std::string reflect_output_accessor(OutputView const& output)
{
  auto element = std::visit([](auto& a) { return type_to_name(a.type()); }, output);
  auto column =
    std::holds_alternative<mutable_string_view_column_view>(output)
      ? "cudf::jit::column_device_view_span_wrapper<cudf::mutable_column_device_view_core>"
      : "cudf::mutable_column_device_view_core";

  return jitify2::Template("cudf::jit::column_accessor").instantiate(column, element, false);
}

std::vector<std::string> reflect_output_accessors(OutputViews outputs)
{
  std::vector<std::string> res;
  std::transform(thrust::counting_iterator<size_t>(0),
                 thrust::counting_iterator(outputs.size()),
                 std::back_inserter(res),
                 [&](auto i) { return reflect_output_accessor(outputs[i]); });

  return res;
}

struct TransformKernel {
  static jitify2::Kernel instantiate(null_aware is_null_aware,
                                     bool may_evaluate_null,
                                     bool has_user_data,
                                     std::string const& ins,
                                     std::string const& outs,
                                     std::string const& udf,
                                     udf_source_type source_type)
  {
    CUDF_FUNC_RANGE();
    auto cuda_source =
      (source_type == udf_source_type::PTX)
        ? jit::parse_single_function_ptx(
            udf,
            "GENERIC_TRANSFORM_OP",
            jit::build_ptx_params(output_typenames, input_typenames, has_user_data))
        : jit::parse_single_function_cuda(udf, "GENERIC_TRANSFORM_OP");

    auto kernel = jitify2::reflection::Template("cudf::jit::transform_kernel")
                    .instantiate(is_null_aware, may_evaluate_null, has_user_data, ins, outs);

    return jit::get_udf_kernel(*transform_jit_kernel_cu_jit, kernel, cuda_source);
  }

  static void launch(jitify2::Kernel const& kernel,
                     size_type row_size,
                     void* user_data,
                     column_device_view_core const* inputs,
                     bitmask_type const* null_mask_and,
                     mutable_column_device_view_core const* outputs,
                     rmm::cuda_stream_view stream)
  {
    CUDF_FUNC_RANGE();
    void* args[] = {&row_size, &user_data, &inputs, &null_mask_and, &outputs};
    kernel->configure_1d_max_occupancy(0, 0, nullptr, stream.value())->launch_raw(args);
  }

  static auto to_device_args(InputViews inputs,
                             OutputViews outputs,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
  {
    std::vector<Handle> handles;
    std::vector<column_device_view> h_args;

    for (auto& in : inputs) {
      if (auto* col = std::get_if<column_view>(&in)) {
        auto handle = column_device_view::create(*col, stream, mr);
        h_args.push_back(*handle);
        handles.push_back(std::move(handle));
      } else if (auto& scalar = std::get<scalar_column_view>(in)) {
        auto handle = column_device_view::create(scalar.as_column_view(), stream, mr);
        h_args.push_back(*handle);
        handles.push_back(std::move(handle));
      }
    }

    for (auto& out : outputs) {
      if (auto* col = std::get_if<mutable_column_view>(&out)) {
        auto handle = mutable_column_device_view::create(*col, stream, mr);
        h_args.push_back(*handle);
        handles.push_back(std::move(handle));
      } else if (auto& sv_column = std::get<mutable_string_view_column_view>(out)) {
        auto handle = sv_column.to_device();
        h_args.push_back(*handle);
        handles.push_back(std::move(handle));
      }
    }

    rmm::device_uvector<column_device_view> d_args{h_args.size(), stream, mr};

    detail::cuda_memcpy_async_impl(d_args.data(),
                                   h_args.data(),
                                   h_args.size() * sizeof(column_device_view),
                                   detail::host_memory_kind::PAGEABLE,
                                   stream);

    return std::make_tuple(std::move(handles), std::move(d_args));
  }

  static void run(null_aware is_null_aware,
                  bool may_evaluate_null,
                  bool has_user_data,
                  size_type row_size,
                  void* user_data,
                  InputViews inputs,
                  bitmask_type const* d_null_mask_and,
                  OutputsView outputs,
                  std::string const& udf,
                  udf_source_type source_type,
                  rmm::cuda_stream_view stream,
                  rmm::device_async_resource_ref mr)
  {
    auto in_types = jitify2::reflection::Template("cudf::jit::type_list")
                      .instantiate(jit::reflect_input_accessors(inputs));
    auto out_types = jitify2::reflection::Template("cudf::jit::type_list")
                       .instantiate(jit::reflect_output_accessors(outputs));

    auto kernel = instantiate(
      is_null_aware, may_evaluate_null, has_user_data, in_types, out_types, udf, source_type);

    auto [args, handles] = to_device_args(inputs, outputs, stream, mr);

    auto* d_inputs = args.data();
    auto* d_outputs =
      reinterpret_cast<mutable_column_device_view const*>(args.data() + inputs.size());

    launch(kernel, row_size, user_data, d_inputs, d_null_mask_and, d_outputs, stream);
  }
};

std::tuple<rmm::device_buffer, size_type> and_null_mask(size_type row_size,
                                                        InputViews inputs,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  // collect the non-scalar elements that contribute to the resulting bitmask
  std::vector<column_view> bitmask_columns;

  // to handle null masks for scalars, we just check if the scalar element is null. If it is null,
  // then all the rows of the transform output will be null. This helps us prevent creating
  // column-sized bitmasks for each scalar.
  for (auto const& in : inputs) {
    if (auto* scalar = std::get_if<scalar_column_view>(&in)) {
      // all nulls
      if (scalar->has_nulls()) {
        return std::make_tuple(create_null_mask(row_size, mask_state::ALL_NULL, stream, mr),
                               row_size);
      }
    } else {
      auto& col = std::get<column_view>(in);
      bitmask_columns.emplace_back(col);
    }
  }

  if (bitmask_columns.empty()) {
    // if there are no non-scalar columns contributing to the null-mask, then the output is all
    // valid (scalar projection) given that the scalar is not null (checked above)
    return std::make_tuple(create_null_mask(row_size, mask_state::UNALLOCATED, stream, mr), 0);
  }

  return cudf::bitmask_and(table_view{bitmask_columns}, stream, mr);
}

bool may_evaluate_null(InputViews inputs, null_aware is_null_aware, output_nullability null_out)
{
  // null-aware UDFs will evaluate nulls unless explicitly marked as not producing nulls
  if (is_null_aware == null_aware::YES) {
    return null_out != output_nullability::ALL_VALID;
  } else {
    /// null-unaware UDFs will evaluate nulls if any input is nullable unless explicitly marked
    /// as not producing nulls
    bool any_nullable = std::any_of(inputs.begin(), inputs.end(), [](auto const& input) {
      return std::visit([](auto const& col) { return col.nullable(); }, input);
    });
    return any_nullable && null_out == output_nullability::PRESERVE;
  }
}

auto finalize(std::vector<OutputColumn> outputs,
              rmm::cuda_stream_view stream,
              rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<column>> results;

  for (auto& out : outputs) {
    if (auto* col = std::get_if<std::unique_ptr<column>>(&out)) {
      results.push_back(std::move(*col));
    } else if (auto& str = std::get<std::unique_ptr<string_view_column>>(out)) {
      auto result = detail::make_strings_column(
        str->_data, std::move(str->_null_mask), std::nullopt, stream, mr);
      results.push_back(std::move(result));
    }
  }

  return results;
}

std::unique_ptr<table> transform_operation(size_type row_size,
                                           InputViews inputs,
                                           std::string const& udf,
                                           std::span<data_type const> output_types,
                                           udf_source_type source_type,
                                           std::optional<void*> user_data,
                                           null_aware is_null_aware,
                                           std::span<output_nullability const> null_policies,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  std::vector<OutputColumn> outputs;

  auto may_return_nulls =
    std::any_of(null_policies.begin(), null_policies.end(), [](auto null_policy) {
      return null_policy != output_nullability::ALL_VALID;
    }) may_evaluate_null(inputs, is_null_aware, null_policy);

  for (size_t i = 0; i < output_types.size(); i++) {
    auto type        = output_types[i];
    auto null_policy = null_policies[i];

    if (is_fixed_width(type)) {
      auto col = make_fixed_width_column(type, row_size, stream, mr);
    } else if (type == type_id::STRING) {
      auto col = string_view_column::make(
        row_size,
        may_return_nulls ? null_mask_state::UNINITIALIZED : null_mask_state::UNALLOCATED,
        stream,
        mr);

      outputs.push_back(std::make_unique<string_view_column>(std::move(col)));
    } else {
      CUDF_UNREACHABLE("Unsupported output type for transform");
    }
  }

  TransformKernel::run(is_null_aware,
                       may_return_nulls,
                       user_data.has_value(),
                       row_size,
                       user_data.value_or(nullptr),
                       inputs,
                       nullptr,  // TODO: compute and pass the and of the null masks if needed
                       outputs,
                       udf,
                       source_type,
                       stream,
                       mr);

  return std::make_unique<table>(finalize(std::move(outputs), stream, mr));
}

void perform_checks(null_aware is_null_aware,
                    udf_source_type source_type,
                    std::optional<size_type> in_row_size,
                    std::span<data_type const> output_types,
                    InputViews inputs)
{
  // TODO: what to do when mutable_column_view arguments violate expected flow?
  // i.e. stencil generation, row-size determination, etc.
  CUDF_EXPECTS(
    !inputs.empty(), "Transform must have at least 1 input column", std::invalid_argument);
  CUDF_EXPECTS(!(is_null_aware == null_aware::YES && source_type == udf_source_type::PTX),
               "Optional types are not supported in PTX UDFs",
               std::invalid_argument);

  CUDF_EXPECTS(is_fixed_width(output_type) || output_type.id() == type_id::STRING,
               "Transforms only support output of fixed-width or string types",
               std::invalid_argument);

  auto get_type = [](auto const& in) {
    return std::visit([](auto const& col) { return col.type(); }, in);
  };

  CUDF_EXPECTS(
    std::all_of(thrust::make_transform_iterator(inputs.begin(), get_type),
                thrust::make_transform_iterator(inputs.end(), get_type),
                [](data_type t) { return is_fixed_width(t) || (t.id() == type_id::STRING); }),
    "Transforms only support input of fixed-width or string types",
    std::invalid_argument);


  if (!in_row_size.has_value()) {
    CUDF_EXPECTS(
      std::any_of(inputs.begin(),
                  inputs.end(),
                  [](auto const& input) { return std::holds_alternative<column_view>(input); }),
      "At least one input of a transform must be a non-scalar column if row size is not provided",
      std::invalid_argument);
  }

  auto row_size = in_row_size.value_or(jit::get_projection_size(inputs));
  CUDF_EXPECTS(std::all_of(inputs.begin(),
                           inputs.end(),
                           [&](auto & in) {
                             if (auto * col = std::get_if<column_view>(&input)) {
                               return col->size() == row_size;
                             }
                             return true;
                           }),
               "All transform input columns must have the same size",
               std::invalid_argument);
}

}  // namespace

std::vector<std::unique_ptr<column>> transform_extended2(std::string const& udf,
                                                         udf_source_type source_type,
                                                         null_aware is_null_aware,
                                                         std::optional<size_type> row_size,
                                                         std::optional<void*> user_data,
                                                         std::span<transform_input const> inputs,
                                                         std::span<transform_output const> outputs,
                                                         rmm::cuda_stream_view stream,
                                                         rmm::device_async_resource_ref mr)
{
}

std::unique_ptr<column> transform_extended(
  std::span<std::variant<column_view, scalar_column_view> const> inputs,
  std::string const& udf,
  data_type output_type,
  udf_source_type source_type,
  std::optional<void*> user_data,
  null_aware is_null_aware,
  std::optional<size_type> row_size,
  output_nullability null_policy,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
}

std::unique_ptr<column> transform(std::vector<column_view> const& columns,
                                  std::string const& transform_udf,
                                  data_type output_type,
                                  bool is_ptx,
                                  std::optional<void*> user_data,
                                  null_aware is_null_aware,
                                  output_nullability null_policy,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  // legacy behavior was to detect which column were scalars based on their sizes
  std::vector<std::variant<column_view, scalar_column_view>> inputs;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  auto base_column = jit::get_transform_base_column(columns);
  for (auto const& col : columns) {
    if (jit::is_scalar(base_column->size(), col.size())) {
#pragma GCC diagnostic pop
      inputs.emplace_back(scalar_column_view{col});
    } else {
      inputs.emplace_back(col);
    }
  }

  // TODO: take inputs of mutable column view strings and pass their data to the kernel
  // TODO: take sizer for the strings, needs to allow zero-sized outputs

  return transform_extended(inputs,
                            transform_udf,
                            output_type,
                            is_ptx ? udf_source_type::PTX : udf_source_type::CUDA,
                            user_data,
                            is_null_aware,
                            base_column->size(),
                            null_policy,
                            stream,
                            mr);
}

std::unique_ptr<column> compute_column_jit(table_view const& table,
                                           ast::expression const& expr,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  detail::row_ir::ast_args ast_args{.table = table};
  auto args = detail::row_ir::ast_converter::compute_column(
    detail::row_ir::target::CUDA, expr, ast_args, stream, mr);
  return transform_extended(args.inputs,
                            args.udf,
                            args.output_type,
                            args.source_type,
                            args.user_data,
                            args.is_null_aware,
                            args.row_size,
                            args.null_policy,
                            stream,
                            mr);
}

}  // namespace cudf
