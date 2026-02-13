/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "helpers.hpp"

namespace cudf {
namespace jit {

size_type get_major_size(column_view const& col) { return col.size(); }

/// @brief Scalar columns don't contribute to the row-size of a transform.
size_type get_major_size(scalar_column_view const& col) { return 0; }

size_type get_transform_major_size(
  std::span<std::variant<column_view, scalar_column_view> const> inputs)
{
  CUDF_EXPECTS(
    !inputs.empty(), "Transform must have at least 1 input column", std::invalid_argument);

  auto get_size = [](auto const& var) {
    return std::visit([](auto& a) { return get_major_size(a); }, var);
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

input_reflection reflect_input(std::variant<column_view, scalar_column_view> const& input)
{
  auto get_type_name = [](auto const& var) {
    return std::visit([](auto& a) { return type_to_name(a.type()); }, var);
  };

  return input_reflection{get_type_name(input), std::holds_alternative<scalar_column_view>(input)};
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

}  // namespace jit
}  // namespace cudf
