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
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/detail/utilities.hpp>
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

struct mut_fixed_width_column_view {
  mutable_column_view _view;

  auto to_device(rmm::cuda_stream_view stream) const
  {
    return mutable_column_device_view::create(_view, stream);
  }
};

struct fixed_width_column {
  std::unique_ptr<column> _col = nullptr;

  static auto make(data_type type,
                   size_type size,
                   rmm::device_buffer null_mask,
                   size_type null_count,
                   rmm::cuda_stream_view stream,
                   rmm::device_async_resource_ref mr)
  {
    return fixed_width_column{
      make_fixed_width_column(type, size, std::move(null_mask), null_count, stream, mr)};
  }

  auto mutable_view() const { return mut_fixed_width_column_view{_col->mutable_view()}; }

  void set_null_count(size_type count) { _col->set_null_count(count); }

  bool nullable() const { return _col->nullable(); }

  bitmask_type* null_mask() { return _col->mutable_view().null_mask(); }
};

struct mut_string_views_column_view {
  void* _data{nullptr};
  size_type _size{0};
  bitmask_type const* _null_mask{nullptr};
  size_type _offset{0};
  size_type _null_count{0};

  auto to_device(rmm::cuda_stream_view stream) const
  {
    using view = mutable_column_device_view;
    return std::unique_ptr<view, std::function<void(view*)>>(
      new view{
        view::from_parts(data_type{type_id::EMPTY}, _size, _data, _null_mask, _offset, nullptr, 0)},
      [](auto* p) { delete p; });
  }
};

struct string_views_column {
  rmm::device_buffer _data;
  size_type _size{0};
  rmm::device_buffer _null_mask{};
  size_type _null_count{0};

  static auto make(size_type size,
                   rmm::device_buffer null_mask,
                   size_type null_count,
                   rmm::cuda_stream_view stream,
                   rmm::device_async_resource_ref mr)
  {
    rmm::device_buffer data{static_cast<size_t>(size) * sizeof(string_view), stream, mr};
    return string_views_column{std::move(data), size, std::move(null_mask), null_count};
  }

  auto mutable_view() const
  {
    return mut_string_views_column_view{
      const_cast<void*>(_data.data()),
      static_cast<size_type>(_data.size()),
      static_cast<bitmask_type*>(const_cast<void*>(_null_mask.data())),
      0,
      _null_count};
  }

  void set_null_count(size_type count) { _null_count = count; }

  bool nullable() const { return !_null_mask.is_empty(); }

  bitmask_type* null_mask() { return static_cast<bitmask_type*>(_null_mask.data()); }
};

struct mut_strings_column_view {
  mutable_column_view _view;

  auto to_device(rmm::cuda_stream_view stream) const
  {
    return mutable_column_device_view::create(_view, stream);
  }
};

struct mut_strings_column {
  std::unique_ptr<column> _col = nullptr;

  static auto make(size_type size,
                   rmm::device_buffer chars,
                   std::unique_ptr<cudf::column> offsets,
                   rmm::device_buffer null_mask,
                   size_type null_count)
  {
    return mut_strings_column{make_strings_column(
      size, std::move(offsets), std::move(chars), null_count, std::move(null_mask))};
  }

  auto mutable_view() const { return mut_strings_column_view{_col->mutable_view()}; }

  void set_null_count(size_type count) { _col->set_null_count(count); }

  bool nullable() const { return _col->nullable(); }

  bitmask_type* null_mask() { return _col->mutable_view().null_mask(); }
};

using input_column_view = transform_input;
using output_column     = std::variant<fixed_width_column, string_views_column, mut_strings_column>;
using handle            = std::variant<
             std::unique_ptr<column_device_view, std::function<void(column_device_view*)>>,
             std::unique_ptr<mutable_column_device_view, std::function<void(mutable_column_device_view*)>>>;

namespace jit_transform {

jitify2::Kernel instantiate(null_aware is_null_aware,
                            bool has_stencil,
                            bool has_user_data,
                            std::string const& ins,
                            std::string const& outs,
                            std::vector<std::string> const& ptx_input_types,
                            std::vector<std::string> const& ptx_output_types,
                            std::string const& udf,
                            udf_source_type source_type)
{
  CUDF_FUNC_RANGE();
  auto cuda_source = (source_type == udf_source_type::PTX)
                       ? jit::parse_single_function_ptx(
                           udf,
                           "GENERIC_TRANSFORM_OP",
                           jit::build_ptx_params(ptx_output_types, ptx_input_types, has_user_data))
                       : jit::parse_single_function_cuda(udf, "GENERIC_TRANSFORM_OP");

  auto kernel = jitify2::reflection::Template("cudf::jit::transform_kernel")
                  .instantiate(is_null_aware, has_stencil, has_user_data, ins, outs);

  return jit::get_udf_kernel(*transform_jit_kernel_cu_jit, kernel, cuda_source, {"--restrict"});
}

void launch(jitify2::Kernel const& kernel,
            size_type row_size,
            bitmask_type const* stencil,
            void* user_data,
            column_device_view_core const* incols,
            mutable_column_device_view_core const* outcols,
            rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  void* args[] = {&row_size, &stencil, &user_data, &incols, &outcols};
  kernel->configure_1d_max_occupancy(0, 0, nullptr, stream.value())->launch_raw(args);
}

std::string reflect_input_element(column_view const& c) { return type_to_name(c.type()); }

std::string reflect_input_element(scalar_column_view const& c) { return type_to_name(c.type()); }

std::string reflect_output_element(fixed_width_column const& c)
{
  return type_to_name(c._col->type());
}

std::string reflect_output_element(string_views_column const&) { return "cudf::string_view"; }

std::string reflect_output_element(mut_strings_column const&) { return "cuda::std::span<char>"; }

std::string reflect_input_column(column_view const&) { return "cudf::column_device_view_core"; }

std::string reflect_input_column(scalar_column_view const&)
{
  return "cudf::column_device_view_core";
}

std::string reflect_output_column(fixed_width_column const&)
{
  return "cudf::mutable_column_device_view_core";
}

std::string reflect_output_column(string_views_column const&)
{
  return "cudf::jit::vector_column_device_view";
}

std::string reflect_output_column(mut_strings_column const&)
{
  return "cudf::jit::mut_strings_column_device_view";
}

auto reflect(udf_source_type source_type,
             std::span<input_column_view const> inputs,
             std::span<output_column const> outputs,
             std::span<uint8_t const> input_may_be_nullable,
             std::span<uint8_t const> output_may_be_nullable)
{
  std::vector<std::string> ins;

  for (size_t i = 0; i < inputs.size(); i++) {
    auto& in               = inputs[i];
    auto column            = std::visit([](auto& c) { return reflect_input_column(c); }, in);
    auto element           = std::visit([](auto& c) { return reflect_input_element(c); }, in);
    auto optional_element  = std::format("cuda::std::optional<{}>", element);
    bool as_scalar         = std::holds_alternative<scalar_column_view>(in);
    bool may_be_nullable   = input_may_be_nullable[i];
    auto is_strings_output = false;
    auto accessor =
      jitify2::reflection::Template("cudf::jit::column_accessor")
        .instantiate(
          i, column, element, optional_element, as_scalar, may_be_nullable, is_strings_output);
    ins.push_back(accessor);
  }

  std::vector<std::string> outs;

  for (size_t i = 0; i < outputs.size(); i++) {
    auto& out              = outputs[i];
    auto column            = std::visit([](auto& c) { return reflect_output_column(c); }, out);
    auto element           = std::visit([](auto& c) { return reflect_output_element(c); }, out);
    auto optional_element  = std::format("cuda::std::optional<{}>", element);
    bool as_scalar         = false;  // never scalar
    bool may_be_nullable   = output_may_be_nullable[i];
    auto is_strings_output = std::holds_alternative<mut_strings_column>(out);
    auto accessor =
      jitify2::reflection::Template("cudf::jit::column_accessor")
        .instantiate(
          i, column, element, optional_element, as_scalar, may_be_nullable, is_strings_output);

    outs.push_back(accessor);
  }

  auto in_list  = jitify2::reflection::Template("cudf::jit::type_list").instantiate(ins);
  auto out_list = jitify2::reflection::Template("cudf::jit::type_list").instantiate(outs);

  std::vector<std::string> ptx_input_types;
  std::vector<std::string> ptx_output_types;

  if (source_type == udf_source_type::PTX) {
    for (auto& in : inputs) {
      ptx_input_types.push_back(std::visit([](auto& c) { return reflect_input_element(c); }, in));
    }

    for (auto& out : outputs) {
      ptx_output_types.push_back(
        std::visit([](auto& c) { return reflect_output_element(c); }, out));
    }
  }

  return std::make_tuple(in_list, out_list, ptx_input_types, ptx_output_types);
}

auto to_args(std::span<input_column_view const> inputs,
             std::span<output_column const> outputs,
             rmm::cuda_stream_view stream,
             rmm::device_async_resource_ref mr)
{
  std::vector<handle> handles;
  std::vector<detail::column_device_view_base> h_args;

  for (auto& in : inputs) {
    if (auto* col = std::get_if<column_view>(&in)) {
      auto hnd = column_device_view::create(*col, stream);
      h_args.push_back(*hnd);
      handles.push_back(std::move(hnd));
    } else {
      auto& scalar = std::get<scalar_column_view>(in);
      auto hnd     = column_device_view::create(scalar.as_column_view(), stream);
      h_args.push_back(*hnd);
      handles.push_back(std::move(hnd));
    }
  }

  for (auto& out : outputs) {
    std::visit(
      [&](auto& col) {
        auto hnd = col.mutable_view().to_device(stream);
        h_args.push_back(*hnd);
        handles.push_back(std::move(hnd));
      },
      out);
  }

  rmm::device_buffer d_args{h_args.size() * sizeof(detail::column_device_view_base), stream, mr};

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
         bitmask_type const* d_stencil,
         void* user_data,
         std::span<input_column_view const> inputs,
         std::span<output_column const> outputs,
         std::span<uint8_t const> input_may_be_nullable,
         std::span<uint8_t const> output_may_be_nullable,
         std::string const& udf,
         udf_source_type source_type,
         rmm::cuda_stream_view stream,
         rmm::device_async_resource_ref mr)
{
  auto [in_types, out_types, ptx_in_types, ptx_out_types] =
    reflect(source_type, inputs, outputs, input_may_be_nullable, output_may_be_nullable);
  auto kernel          = instantiate(is_null_aware,
                            has_stencil,
                            has_user_data,
                            in_types,
                            out_types,
                            ptx_in_types,
                            ptx_out_types,
                            udf,
                            source_type);
  auto [cols, handles] = to_args(inputs, outputs, stream, mr);
  auto* incols         = reinterpret_cast<column_device_view_core const*>(cols.data());
  auto* outcols = reinterpret_cast<mutable_column_device_view_core const*>(incols + inputs.size());
  return launch(kernel, row_size, d_stencil, user_data, incols, outcols, stream);
}

}  // namespace jit_transform

size_type inplace_null_mask_and(bitmask_type* null_mask,
                                size_type row_size,
                                std::span<transform_input const> inputs,
                                rmm::cuda_stream_view stream)
{
  auto is_nullable = null_mask != nullptr;

  if (!is_nullable) { return 0; }

  auto num_words = num_bitmask_words(row_size);
  auto num_bytes = num_words * sizeof(bitmask_type);

  CUDF_EXPECTS(!inputs.empty(), "Inputs must not be empty", std::logic_error);

  auto has_scalars = std::any_of(inputs.begin(), inputs.end(), [](auto& in) {
    return std::holds_alternative<scalar_column_view>(in);
  });

  if (has_scalars) {
    auto scalar_is_null = std::any_of(inputs.begin(), inputs.end(), [](auto& in) {
      if (auto* scalar = std::get_if<scalar_column_view>(&in)) { return scalar->has_nulls(); }
      return false;
    });

    if (scalar_is_null) {
      // scalar is null, all rows will be null
      CUDF_CUDA_TRY(cudaMemsetAsync(null_mask, 0x00, num_bytes, stream.value()));

      return row_size;
    }
  }

  auto has_cols = std::any_of(
    inputs.begin(), inputs.end(), [](auto& in) { return std::holds_alternative<column_view>(in); });

  if (has_cols) {
    // no non-scalar columns, so all rows are valid
    CUDF_CUDA_TRY(cudaMemsetAsync(null_mask, 0xFF, num_bytes, stream.value()));

    return 0;
  }

  // collect the non-scalar nullable columns that contribute to the output nullmask
  std::vector<bitmask_type const*> nullable_masks;
  std::vector<size_type> nullable_offsets;
  std::vector<size_type> nullable_null_counts;

  for (auto& in : inputs) {
    if (auto* c = std::get_if<column_view>(&in)) {
      if (c->nullable()) {
        nullable_masks.push_back(c->null_mask());
        nullable_offsets.push_back(c->offset());
        nullable_null_counts.push_back(c->null_count());
      }
    }
  }

  if (nullable_masks.empty()) {
    // we only have non-nullable columns, so all rows are valid
    CUDF_CUDA_TRY(cudaMemsetAsync(null_mask, 0xFF, num_bytes, stream.value()));

    return 0;
  }

  if (nullable_masks.size() == 1) {
    // only 1 mask provided, copy it directly to the output

    detail::cuda_memcpy_async_impl(
      null_mask, nullable_masks[0], num_bytes, detail::host_memory_kind::PINNED, stream);
    return nullable_null_counts[0];
  }

  return detail::inplace_bitmask_and(
    device_span<bitmask_type>{null_mask, static_cast<size_t>(num_words)},
    nullable_masks,
    nullable_offsets,
    row_size,
    stream);
}

/**
 * @brief Get the null-mask transformation for the transform operation based on the UDF's parameters
 * and inputs
 *
 * @return input and output null-policies for the UDF kernel
 */
auto get_null_transformation(null_aware is_null_aware,
                             std::span<transform_input const> inputs,
                             std::span<transform_output const> outputs)
{
  std::vector<uint8_t> input_may_be_nullable;

  for (auto& in : inputs) {
    input_may_be_nullable.push_back(true);
  }

  bool any_input_nullable = std::any_of(inputs.begin(), inputs.end(), [](auto& in) {
    return std::visit([](auto& c) { return c.nullable(); }, in);
  });

  std::vector<uint8_t> output_may_be_nullable;
  for (auto& out : outputs) {
    bool may_eval_nulls = true;
    if (is_null_aware == null_aware::YES) {
      // null-aware UDFs may evaluate nulls unless the output is explicitly marked as all valid
      may_eval_nulls = out.nullability != output_nullability::ALL_VALID;
    } else {
      // null-unaware UDFs may evaluate nulls if any input is nullable unless explicitly marked as
      // not producing nulls
      may_eval_nulls = any_input_nullable && (out.nullability == output_nullability::PRESERVE);
    }

    output_may_be_nullable.push_back(may_eval_nulls);
  }

  return std::make_tuple(std::move(input_may_be_nullable), std::move(output_may_be_nullable));
}

void perform_checks(udf_source_type source_type,
                    null_aware is_null_aware,
                    std::optional<size_type> in_row_size,
                    std::span<transform_input const> inputs,
                    std::span<transform_output const> outputs,
                    std::span<std::unique_ptr<column> const> string_offsets)
{
  CUDF_EXPECTS(
    !inputs.empty(), "Transform must have at least 1 input column", std::invalid_argument);
  CUDF_EXPECTS(!(is_null_aware == null_aware::YES && source_type == udf_source_type::PTX),
               "Optional types are not supported in PTX UDFs",
               std::invalid_argument);

  CUDF_EXPECTS(std::all_of(outputs.begin(),
                           outputs.end(),
                           [](auto& out) {
                             return is_fixed_width(out.type) || (out.type.id() == type_id::STRING);
                           }),
               "Transforms only support output of fixed-width or string types",
               std::invalid_argument);

  CUDF_EXPECTS(std::all_of(inputs.begin(),
                           inputs.end(),
                           [&](auto& in) {
                             auto type = std::visit([](auto& c) { return c.type(); }, in);
                             return is_fixed_width(type) || (type.id() == type_id::STRING);
                           }),
               "Transforms only support input of fixed-width or string types",
               std::invalid_argument);

  if (!in_row_size.has_value()) {
    CUDF_EXPECTS(
      std::any_of(inputs.begin(),
                  inputs.end(),
                  [](auto const& in) { return std::holds_alternative<column_view>(in); }),
      "At least one input of a transform must be a non-scalar column if row size is not provided",
      std::invalid_argument);
  }

  auto row_size = in_row_size.value_or(jit::get_projection_size(inputs));
  CUDF_EXPECTS(std::all_of(inputs.begin(),
                           inputs.end(),
                           [&](auto& in) {
                             if (auto* col = std::get_if<column_view>(&in)) {
                               return col->size() == row_size;
                             }
                             return true;
                           }),
               "All transform input columns must have the same size",
               std::invalid_argument);

  CUDF_EXPECTS(string_offsets.empty() || (string_offsets.size() == outputs.size()),
               "Number of string offsets must be empty or match the number of outputs",
               std::invalid_argument);

  CUDF_EXPECTS(std::all_of(thrust::make_counting_iterator(size_t{0}),
                           thrust::make_counting_iterator(string_offsets.size()),
                           [&](auto i) {
                             if (outputs[i].type.id() == type_id::STRING) { return true; }
                             return string_offsets.empty() || string_offsets[i] != nullptr;
                           }),
               "String offsets must only be provided for string outputs",
               std::invalid_argument);
}

std::optional<bitmask_type const*> prepare_stencil(null_aware is_null_aware,
                                                   bool null_strict,
                                                   size_type row_size,
                                                   std::span<transform_input const> inputs,
                                                   std::span<output_column> outputs,
                                                   std::span<bitmask_type* const> output_masks,
                                                   rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  // null-aware, no stencil needed
  if (is_null_aware != null_aware::NO) { return std::nullopt; }

  std::optional<bitmask_type*> stencil = std::nullopt;

  for (auto* mask : output_masks) {
    if (mask != nullptr) {
      stencil = mask;
      break;
    }
  }

  // no nullable outputs
  if (!stencil.has_value()) {
    if (!null_strict) {
      // function is not null-strict, we don't need to use the stencil in the kernel
      return std::nullopt;
    }

    return nullptr;
  }

  auto null_count = inplace_null_mask_and(*stencil, row_size, inputs, stream);

  for (size_t i = 0; i < outputs.size(); i++) {
    auto* mask = output_masks[i];

    if (mask != nullptr) {
      if (mask != *stencil) {
        detail::cuda_memcpy_async_impl(mask,
                                       *stencil,
                                       bitmask_allocation_size_bytes(row_size),
                                       detail::host_memory_kind::PINNED,
                                       stream);
      }

      std::visit(
        [&](auto& c) {
          if (c.nullable()) {
            c.set_null_count(null_count);
          } else {
            c.set_null_count(0);
          }
        },
        outputs[i]);
    }
  }

  if (!null_strict) {
    // function is not null-strict, we don't need to use the stencil in the kernel
    return std::nullopt;
  }

  return stencil;
}

auto allocate_outputs(size_type row_size,
                      std::span<transform_output const> outputs,
                      std::span<uint8_t const> is_nullable,
                      std::vector<std::unique_ptr<column>> string_offsets,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  std::vector<output_column> cols;
  std::vector<bitmask_type*> masks;

  for (size_t i = 0; i < outputs.size(); i++) {
    auto output     = outputs[i];
    auto nullable   = is_nullable[i];
    auto null_state = nullable ? mask_state::UNINITIALIZED : mask_state::UNALLOCATED;
    auto null_mask  = create_null_mask(row_size, null_state, stream, mr);
    masks.push_back(static_cast<bitmask_type*>(null_mask.data()));

    if (is_fixed_width(output.type)) {
      auto col =
        fixed_width_column::make(output.type, row_size, std::move(null_mask), 0, stream, mr);
      cols.push_back(std::move(col));
    } else if (output.type.id() == type_id::STRING) {
      if (string_offsets.empty() || string_offsets[i] == nullptr) {
        auto col = string_views_column::make(row_size, std::move(null_mask), 0, stream, mr);
        cols.push_back(std::move(col));
      } else {
        auto chars_size =
          strings::detail::get_offset_value(string_offsets[i]->view(), row_size, stream);
        auto chars = rmm::device_buffer{static_cast<size_t>(chars_size), stream, mr};
        auto col   = mut_strings_column::make(
          row_size, std::move(chars), std::move(string_offsets[i]), std::move(null_mask), 0);
        cols.push_back(std::move(col));
      }
    } else {
      CUDF_UNREACHABLE("Unsupported output type for transform");
    }
  }

  return std::make_tuple(std::move(cols), std::move(masks));
}

auto prepare_outputs(null_aware is_null_aware,
                     size_type row_size,
                     std::span<transform_input const> inputs,
                     std::span<transform_output const> outputs,
                     std::span<uint8_t const> output_is_nullable,
                     std::vector<std::unique_ptr<column>> string_offsets,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  auto [output_columns, output_column_masks] =
    allocate_outputs(row_size, outputs, output_is_nullable, std::move(string_offsets), stream, mr);
  auto stencil = prepare_stencil(
    is_null_aware, true, row_size, inputs, output_columns, output_column_masks, stream);

  return std::make_tuple(std::move(output_columns), stencil);
}

void update_null_counts(std::span<output_column> outputs,
                        null_aware is_null_aware,
                        size_type row_size,
                        rmm::cuda_stream_view stream)
{
  // update null counts if the function is not null-aware, since we haven't processed nullability
  // ahead of time
  if (is_null_aware == null_aware::YES) {
    std::vector<bitmask_type const*> bitmasks;
    std::vector<int32_t> indices;

    for (size_t i = 0; i < outputs.size(); i++) {
      std::visit(
        [&](auto& c) {
          if (c.nullable()) {
            indices.push_back(i);
            bitmasks.push_back(c.null_mask());
          }
        },
        outputs[i]);
    }

    auto null_counts = batch_null_count(bitmasks, 0, row_size, stream);

    for (size_t i = 0; i < bitmasks.size(); i++) {
      std::visit([&](auto& c) { c.set_null_count(null_counts[i]); }, outputs[indices[i]]);
    }
  }
}

auto finalize_output(fixed_width_column&& c, rmm::cuda_stream_view, rmm::device_async_resource_ref)
{
  return std::move(c._col);
}

auto finalize_output(mut_strings_column&& c, rmm::cuda_stream_view, rmm::device_async_resource_ref)
{
  return std::move(c._col);
}

auto finalize_output(string_views_column&& c,
                     rmm::cuda_stream_view stream,
                     rmm::device_async_resource_ref mr)
{
  return strings::detail::make_strings_column(
    device_span<string_view const>{static_cast<string_view const*>(c._data.data()),
                                   static_cast<size_t>(c._size)},
    std::move(c._null_mask),
    c._null_count,
    stream,
    mr);
}

auto finalize_outputs(null_aware is_null_aware,
                      size_type row_size,
                      std::vector<output_column> outputs,
                      rmm::cuda_stream_view stream,
                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  update_null_counts(outputs, is_null_aware, row_size, stream);

  std::vector<std::unique_ptr<column>> results;

  for (auto& out : outputs) {
    std::visit([&](auto& c) { results.push_back(finalize_output(std::move(c), stream, mr)); }, out);
  }

  return results;
}

std::unique_ptr<table> execute_transform(std::string const& udf,
                                         udf_source_type source_type,
                                         null_aware is_null_aware,
                                         std::optional<size_type> in_row_size,
                                         std::optional<void*> user_data,
                                         std::span<transform_input const> inputs,
                                         std::span<transform_output const> outputs,
                                         std::vector<std::unique_ptr<column>> string_offsets,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  auto row_size = in_row_size.value_or(jit::get_projection_size(inputs));
  auto [input_may_be_nullable, output_may_be_nullable] =
    get_null_transformation(is_null_aware, inputs, outputs);
  auto [output_columns, stencil] = prepare_outputs(is_null_aware,
                                                   row_size,
                                                   inputs,
                                                   outputs,
                                                   output_may_be_nullable,
                                                   std::move(string_offsets),
                                                   stream,
                                                   mr);
  jit_transform::run(is_null_aware,
                     stencil.has_value(),
                     user_data.has_value(),
                     row_size,
                     stencil.value_or(nullptr),
                     user_data.value_or(nullptr),
                     inputs,
                     output_columns,
                     input_may_be_nullable,
                     output_may_be_nullable,
                     udf,
                     source_type,
                     stream,
                     mr);

  auto finalized = finalize_outputs(is_null_aware, row_size, std::move(output_columns), stream, mr);
  return std::make_unique<table>(std::move(finalized));
}

}  // namespace

std::unique_ptr<table> multi_transform(std::string const& udf,
                                       udf_source_type source_type,
                                       null_aware is_null_aware,
                                       std::optional<void*> user_data,
                                       std::span<transform_input const> inputs,
                                       std::span<transform_output const> outputs,
                                       std::vector<std::unique_ptr<column>> string_offsets,
                                       std::optional<size_type> row_size,
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

std::unique_ptr<column> transform_extended(std::span<transform_input const> inputs,
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
  transform_output outputs[] = {{.type = output_type, .nullability = null_policy}};
  auto table                 = multi_transform(
    udf, source_type, is_null_aware, user_data, inputs, outputs, {}, row_size, stream, mr);
  auto cols = table->release();
  return std::move(cols[0]);
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
  std::vector<transform_input> inputs;

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
