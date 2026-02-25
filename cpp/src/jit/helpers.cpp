/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "helpers.hpp"

#include <cudf/column/column_device_view_base.cuh>
#include <cudf/detail/nvtx/ranges.hpp>

#include <jit/cache.hpp>

#include <format>

namespace cudf {
namespace jit {

bool is_scalar(cudf::size_type base_column_size, cudf::size_type column_size)
{
  return column_size == 1 && column_size != base_column_size;
}

typename std::vector<column_view>::const_iterator get_transform_base_column(
  std::vector<column_view> const& inputs)
{
  if (inputs.empty()) { return inputs.end(); }

  auto [smallest, largest] = std::minmax_element(
    inputs.begin(), inputs.end(), [](auto const& a, auto const& b) { return a.size() < b.size(); });

  /// when the largest size is 1, the size-1 column could be a scalar or an actual column, it would
  /// be a scalar if it has columns that are zero-sized
  if (largest->size() != 1) { return largest; }

  if (smallest->size() == 0) { return smallest; }

  return largest;
}

size_type get_projection_size(column_view const& col) { return col.size(); }

/// @brief Scalar columns don't contribute to the row-size of a transform.
size_type get_projection_size(scalar_column_view const& col) { return 0; }

size_type get_projection_size(std::span<std::variant<column_view, scalar_column_view> const> inputs)
{
  CUDF_EXPECTS(
    !inputs.empty(), "Transform must have at least 1 input column", std::invalid_argument);

  auto get_size = [](auto const& var) {
    return std::visit([](auto& a) { return get_projection_size(a); }, var);
  };

  return *std::max_element(thrust::make_transform_iterator(inputs.begin(), get_size),
                           thrust::make_transform_iterator(inputs.end(), get_size));
}

std::string input_reflection::accessor(int32_t index) const
{
  auto column_accessor =
    jitify2::reflection::Template("cudf::jit::column_accessor").instantiate(type_name, index);

  return is_scalar ? jitify2::reflection::Template("cudf::jit::scalar_accessor")
                       .instantiate(column_accessor)
                   : column_accessor;
}

std::map<uint32_t, std::string> build_ptx_params(std::span<std::string const> output_typenames,
                                                 std::span<std::string const> input_typenames,
                                                 bool has_user_data)
{
  std::map<uint32_t, std::string> params;
  uint32_t index = 0;

  if (has_user_data) {
    params.emplace(index++, "void *");
    params.emplace(index++, jitify2::reflection::reflect<cudf::size_type>());
  }

  for (auto& name : output_typenames) {
    params.emplace(index++, name + "*");
  }

  for (auto& name : input_typenames) {
    params.emplace(index++, name);
  }

  return params;
}

std::vector<std::string> output_type_names(std::span<mutable_column_view const> views)
{
  std::vector<std::string> names;

  std::transform(views.begin(), views.end(), std::back_inserter(names), [](auto const& view) {
    return type_to_name(view.type());
  });

  return names;
}

std::vector<std::string> input_type_names(
  std::span<std::variant<column_view, scalar_column_view> const> views)
{
  std::vector<std::string> names;
  auto get_type_name = [](auto const& var) {
    return std::visit([](auto& a) { return type_to_name(a.type()); }, var);
  };

  std::transform(views.begin(), views.end(), std::back_inserter(names), [&](auto const& view) {
    return get_type_name(view);
  });

  return names;
}

std::string get_jit_element_type_name(column_view const& view)
{
  if (is_fixed_width(view.type()) || view.type().id() == type_id::STRING) {
    return type_to_name(view.type());
  } else if (view.type().id() == type_id::DICTIONARY32) {
    return std::format(
      "cudf::dictionary_element<{}, {}>",
      get_jit_element_type_name(
        view.child(column_device_view_core::dictionary_offsets_column_index)),
      get_jit_element_type_name(view.child(column_device_view_core::dictionary_keys_column_index)));
  } else {
    CUDF_FAIL("Unsupported type for JIT compilation: " + type_to_name(view.type()));
  }
}

std::string get_jit_element_type_name(scalar_column_view const& view)
{
  return get_jit_element_type_name(view.as_column_view());
}

input_reflection reflect_input(std::variant<column_view, scalar_column_view> const& input)
{
  auto type_name = std::visit([](auto& a) { return get_jit_element_type_name(a); }, input);
  return input_reflection{type_name, std::holds_alternative<scalar_column_view>(input)};
}

std::vector<input_reflection> reflect_inputs(
  std::span<std::variant<column_view, scalar_column_view> const> inputs)
{
  std::vector<input_reflection> reflections;
  std::transform(
    inputs.begin(), inputs.end(), std::back_inserter(reflections), [&](auto const& view) {
      return reflect_input(view);
    });

  return reflections;
}

jitify2::Kernel get_udf_kernel(jitify2::PreprocessedProgramData const& preprocessed_program_data,
                               std::string const& kernel_name,
                               std::string const& cuda_source)
{
  CUDF_FUNC_RANGE();

  int runtime_version;
  CUDF_CUDA_TRY(cudaRuntimeGetVersion(&runtime_version));
  int constexpr min_pch_runtime_version = 12800;  // CUDA 12.8

  std::vector<std::string> options;
  options.emplace_back("-arch=sm_.");

  if (runtime_version >= min_pch_runtime_version) { options.emplace_back("-pch"); }

  return cudf::jit::get_program_cache(preprocessed_program_data)
    .get_kernel(kernel_name, {}, {{"cudf/detail/operation-udf.hpp", cuda_source}}, options);
}

}  // namespace jit
}  // namespace cudf
