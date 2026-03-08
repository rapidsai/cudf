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
struct mut_vector_column_view {
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
  using mut_view_type = mut_vector_column_view<T>;

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
    return mut_view_type{_data.data(),
                         _data.size(),
                         static_cast<bitmask_type*>(_null_mask.data()),
                         _offset,
                         _null_count};
  }
};

struct mut_strings_column_view {
  mutable_column_view view;
};

struct mut_strings_column {
  using mut_view_type = mut_strings_column_view;

  std::unique_ptr<column> strings = nullptr;

  static auto make(size_type size,
                   int64_t chars_size,
                   std::unique_ptr<column> offsets,
                   null_mask_state null_state,
                   rmm::cuda_stream_view stream,
                   rmm::device_async_resource_ref mr)
  {
    // TODO: null-count needs to be updated
    auto null_mask = create_null_mask(size, null_state, stream, mr);
    auto chars     = rmm::device_buffer{chars_size, stream, mr};
    return mut_strings_column{.strings{detail::make_strings_column(size,
                                                                   std::move(offsets),
                                                                   std::move(chars),
                                                                   /* null_count = */ size,
                                                                   std::move(null_mask),
                                                                   stream,
                                                                   mr)}};
  }

  auto mutable_view() { return mut_view_type{.view{strings->mutable_view()}}; }
};

using string_views_column          = vector_column<string_view>;
using mut_string_views_column_view = typename string_views_column::mut_view_type;

using OutputColumn = std::variant<std::unique_ptr<column>, string_views_column, mut_strings_column>;

using InputView = std::variant<column_view, scalar_column_view>;
using OutputView =
  std::variant<mutable_column_view, mut_string_views_column_view, mut_strings_column_view>;

using InputViews  = std::span<InputView const>;
using OutputViews = std::span<OutputView const>;

using InHandle = std::unique_ptr<column_device_view, std::function<void(column_device_view*)>>;
using OutHandle =
  std::unique_ptr<mutable_column_device_view, std::function<void(mutable_column_device_view*)>>;

using Handle = std::variant<InHandle, OutHandle>;

namespace jit {
namespace transform {

jitify2::Kernel instantiate(null_aware is_null_aware,
                            bool has_stencil,
                            bool has_user_data,
                            std::string const& ins,
                            std::string const& outs,
                            std::string const& udf,
                            udf_source_type source_type)
{
  CUDF_FUNC_RANGE();
  auto cuda_source = (source_type == udf_source_type::PTX)
                       ? jit::parse_single_function_ptx(
                           udf,
                           "GENERIC_TRANSFORM_OP",
                           jit::build_ptx_params(output_typenames, input_typenames, has_user_data))
                       : jit::parse_single_function_cuda(udf, "GENERIC_TRANSFORM_OP");

  auto kernel = jitify2::reflection::Template("cudf::jit::transform_kernel")
                  .instantiate(is_null_aware, may_evaluate_null, has_user_data, ins, outs);

  return jit::get_udf_kernel(*transform_jit_kernel_cu_jit, kernel, cuda_source);
}

void launch(jitify2::Kernel const& kernel,
            size_type row_size,
            void* user_data,
            bitmask_type const* stencil,
            detail::column_device_view_base const* columns,
            rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  void* args[] = {&row_size, &user_data, &stencil, &columns};
  kernel->configure_1d_max_occupancy(0, 0, nullptr, stream.value())->launch_raw(args);
}

std::pair<std::string, std::string> reflect(InputViews inputs,
                                            OutputViews outputs,
                                            std::span<uint8_t const> input_may_be_nullable,
                                            std::span<uint8_t const> output_may_be_nullable)
{
  std::vector<std::string> ins;

  for (size_t i = 0; i < inputs.size(); i++) {
    auto& input           = inputs[i];
    auto column           = "cudf::column_device_view_core";
    auto element          = std::visit([](auto& a) { return type_to_name(a.type()); }, input);
    auto optional_element = std::format("cuda::std::optional<{}>", element);
    bool as_scalar        = std::holds_alternative<scalar_column_view>(input);
    bool may_be_nullable  = input_may_be_nullable[i];
    auto accessor =
      jitify2::Template("cudf::jit::column_accessor")
        .instantiate(i, column, element, optional_element, as_scalar, may_be_nullable);
    ins.push_back(accessor);
  }

  std::vector<std::string> outs;

  for (size_t i = 0; i < outputs.size(); i++) {
    auto& output = outputs[i];
    auto column =
      std::holds_alternative<mut_string_views_column_view>(output)
        ? "cudf::jit::column_device_view_span_wrapper<cudf::mutable_column_device_view_core>"
        : "cudf::mutable_column_device_view_core";
    auto element          = std::visit([](auto& a) { return type_to_name(a.type()); }, output);
    auto optional_element = std::format("cuda::std::optional<{}>", element);
    bool as_scalar        = false;  // output can never be a scalar
    bool may_be_nullable  = output_may_be_nullable[i];
    auto accessor =
      jitify2::Template("cudf::jit::column_accessor")
        .instantiate(
          inputs.size() + i, column, element, optional_element, as_scalar, may_be_nullable);

    outs.push_back(accessor);
  }
}

auto to_args(InputViews inputs,
             OutputViews outputs,
             rmm::cuda_stream_view stream,
             rmm::device_async_resource_ref mr)
{
  std::vector<Handle> handles;
  std::vector<detail::column_device_view_base> h_args;

  for (auto& in : inputs) {
    if (auto* col = std::get_if<column_view>(&in)) {
      auto handle = column_device_view::create(*col, stream, mr);
      h_args.push_back(*handle);
      handles.push_back(std::move(handle));
    } else {
      auto& scalar = std::get<scalar_column_view>(in);
      auto handle  = column_device_view::create(scalar.as_column_view(), stream, mr);
      h_args.push_back(*handle);
      handles.push_back(std::move(handle));
    }
  }

  for (auto& out : outputs) {
    if (auto* col = std::get_if<mutable_column_view>(&out)) {
      auto handle = mutable_column_device_view::create(*col, stream, mr);
      h_args.push_back(*handle);
      handles.push_back(std::move(handle));
    } else {
      auto& sv_column = std::get<mut_string_views_column_view>(out);
      auto handle     = sv_column.to_device();
      h_args.push_back(*handle);
      handles.push_back(std::move(handle));
    }
  }

  rmm::device_uvector<detail::column_device_view_base> d_args{h_args.size(), stream, mr};

  detail::cuda_memcpy_async_impl(d_args.data(),
                                 h_args.data(),
                                 h_args.size() * sizeof(detail::column_device_view_base),
                                 detail::host_memory_kind::PAGEABLE,
                                 stream);

  return std::make_tuple(std::move(d_args), std::move(handles));
}

void run(null_aware is_null_aware,
         bool has_stencil,
         bool has_user_data,
         size_type row_size,
         void* user_data,
         bitmask_type const* d_stencil,
         InputViews inputs,
         OutputsView outputs,
         std::span<uint8_t const> input_may_be_nullable,
         std::span<uint8_t const> output_may_be_nullable,
         std::string const& udf,
         udf_source_type source_type,
         rmm::cuda_stream_view stream,
         rmm::device_async_resource_ref mr)
{
  auto [in_types, out_types] =
    reflect(inputs, outputs, input_may_be_nullable, output_may_be_nullable);
  auto kernel =
    instantiate(is_null_aware, has_stencil, has_user_data, in_types, out_types, udf, source_type);
  auto [cols, handles] = to_args(inputs, outputs, stream, mr);
  return launch(kernel, row_size, user_data, d_stencil, cols.data(), stream);
}

}  // namespace transform
}  // namespace jit

std::tuple<rmm::device_buffer, size_type> null_mask_and(size_type row_size,
                                                        InputViews inputs,
                                                        rmm::cuda_stream_view stream,
                                                        rmm::device_async_resource_ref mr)
{
  // TODO(lamarrr): handle non-nullable inputs? or is it handled?
  // what if we only have scalars?
  // what if some of the inputs are non-nullable
  // what if none of the inputs are nullable? will an allocated null mask be created?

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

  // TODO: do we depend on the nullness of the outputs?
  if (bitmask_columns.empty()) {
    // if there are no non-scalar columns contributing to the null-mask, then the output is all
    // valid (scalar projection) given that the scalar is not null (checked above)
    return std::make_tuple(create_null_mask(row_size, mask_state::UNALLOCATED, stream, mr), 0);
  }

  return cudf::bitmask_and(table_view{bitmask_columns}, stream, mr);
}

std::pair<std::vector<uint8_t>, std::vector<uint8_t>> get_nullabilities(
  null_aware is_null_aware, InputViews inputs, std::span<transform_output const> outputs)
{
  std::vector<uint8_t> input_may_be_nullable;

  for (auto& in : inputs) {
    input_may_be_nullable.push_back(true);
  }

  std::vector<uint8_t> output_may_be_nullable;

  bool any_input_nullable = std::any_of(inputs.begin(), inputs.end(), [](auto& input) {
    return std::visit([](auto const& col) { return col.nullable(); }, input);
  });

  for (auto& out : outputs) {
    bool may_eval_null = true;
    if (is_null_aware == null_aware::YES) {
    // null-aware UDFs may evaluate nulls unless the output is explicitly marked as all valid
      may_eval_null = out.nullability != output_nullability::ALL_VALID;
    } else {
      // null-unaware UDFs may evaluate nulls if any input is nullable unless explicitly marked as not producing nulls
      may_eval_null = any_input_nullable && (out.nullability == output_nullability::PRESERVE);
    }

    output_may_be_nullable.push_back(may_eval_null);
  }

  return {input_may_be_nullable, output_may_be_nullable};
}

auto finalize(std::vector<OutputColumn> outputs,
              rmm::cuda_stream_view stream,
              rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<column>> results;

  // TODO: finish
  for (auto& out : outputs) {
    if (auto* col = std::get_if<std::unique_ptr<column>>(&out)) {
      results.push_back(std::move(*col));
    } else if (auto& str = std::get<std::unique_ptr<string_views_column>>(out)) {
      auto result = detail::make_strings_column(
        str->_data, std::move(str->_null_mask), std::nullopt, stream, mr);
      results.push_back(std::move(result));
    }
  }

  return results;
}

void perform_checks(udf_source_type source_type,
                    null_aware is_null_aware,
                    std::optional<size_type> in_row_size,
                    InputViews inputs,
                    std::span<transform_output const> outputs,
                    std::span<std::unique_ptr<column> const> string_offsets)
{
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
                           [&](auto& in) {
                             if (auto* col = std::get_if<column_view>(&input)) {
                               return col->size() == row_size;
                             }
                             return true;
                           }),
               "All transform input columns must have the same size",
               std::invalid_argument);

  // TODO: if string offset is provided, make sure it is a string column
}

std::unique_ptr<table> execute_transform(std::string const& udf,
                                         udf_source_type source_type,
                                         null_aware is_null_aware,
                                         std::optional<size_type> in_row_size,
                                         std::optional<void*> user_data,
                                         InputViews inputs,
                                         std::span<transform_output const> outputs,
                                         std::vector<std::unique_ptr<column>> string_offsets,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  auto row_size = in_row_size.value_or(jit::get_projection_size(inputs));
  // TODO: account for MayBeNullable

  auto [input_may_be_nullable, output_may_be_nullable] =
    get_nullabilities(is_null_aware, inputs, outputs);

  for (size_t i = 0; i < outputs.size(); i++) {
    auto output          = outputs[i];
    auto may_be_nullable = output_may_be_nullable[i];
    auto null_state =
      may_be_nullable ? null_mask_state::UNINITIALIZED : null_mask_state::UNALLOCATED;

    if (is_fixed_width(type)) {
      auto col = make_fixed_width_column(type, row_size, null_state, stream, mr);
    } else if (type == type_id::STRING) {
      if (string_offsets[i] == nullptr) {
        auto col = string_views_column::make(row_size, null_state, stream, mr);
        outputs.push_back(std::make_unique<string_views_column>(std::move(col)));
      } else {
        auto col = mut_strings_column::make(row_size,
                                            /*TODO: chars_size*/ 0,
                                            std::move(string_offsets[i]),
                                            null_state,
                                            stream,
                                            mr);
      }
    } else {
      CUDF_UNREACHABLE("Unsupported output type for transform");
    }
  }

  // stencil in-place
  // inplace_bitmask_and
  bool any_output_nullable = std::any_of(
    output_may_be_nullable.begin(), output_may_be_nullable.end(), [](auto b) { return b; });
  bool can_use_stencil = (is_null_aware == null_aware::NO);

  std::vector<OutputColumn> outputs;
  std::optional<bitmask_type*> stencil = std::nullopt;
  // TODO: create a null mask and copy to all outputs, don't create stencil if all the outputs are
  // non-nullable
  // TODO: how will this affect kernel and outputs?
  // TODO: Null-mask-and might return nullptr
  // TODO: if no output is nullable, the bitmask and should be nullptr
  // TODO: when copying bitmasks stencil, only copy to the nullable outputs

  jit::transform::run(is_null_aware,
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

}  // namespace

std::unique_ptr<table> transform_extended2(std::string const& udf,
                                           udf_source_type source_type,
                                           null_aware is_null_aware,
                                           std::optional<size_type> row_size,
                                           std::optional<void*> user_data,
                                           std::span<transform_input const> inputs,
                                           std::span<transform_output const> outputs,
                                           std::vector<std::unique_ptr<column>> string_offsets,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  perform_checks(source_type, is_null_aware, row_size, inputs, outputs, string_offsets);
  return execute_transform(udf,
                           source_type,
                           is_null_aware,
                           row_size,
                           user_data,
                           inputs,
                           outputs,
                           std::move(string_offsets),
                           stream,
                           mr);
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
  transform_output outputs[] = {{.type = output_type, .null_policy = null_policy}};
  auto table                 = transform_extended2(
    udf, source_type, is_null_aware, row_size, user_data, inputs, outputs, {}, stream, mr);
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
