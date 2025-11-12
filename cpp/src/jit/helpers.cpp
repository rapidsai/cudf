/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "helpers.hpp"

namespace cudf {
namespace jit {

typename std::vector<column_view>::const_iterator get_transform_base_column(
  std::vector<column_view> const& inputs)
{
  // TODO(lamarrr): remove ambiguous row-size-related logic for processing scalars in transforms and
  // filters and use strongly-typed scalars

  if (inputs.empty()) { return inputs.end(); }

  auto [smallest, largest] = std::minmax_element(
    inputs.begin(), inputs.end(), [](auto const& a, auto const& b) { return a.size() < b.size(); });

  /// when the largest size is 1, the size-1 column could be a scalar or an actual column, it would
  /// be a scalar if it has columns that are zero-sized
  if (largest->size() != 1) { return largest; }

  if (smallest->size() == 0) { return smallest; }

  return largest;
}

std::map<uint32_t, std::string> build_ptx_params(std::vector<std::string> const& output_typenames,
                                                 std::vector<std::string> const& input_typenames,
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

input_column_reflection reflect_input_column(size_type base_column_size, column_view column)
{
  return input_column_reflection{type_to_name(column.type()),
                                 is_scalar(base_column_size, column.size())};
}

std::vector<input_column_reflection> reflect_input_columns(size_type base_column_size,
                                                           std::vector<column_view> const& inputs)
{
  std::vector<input_column_reflection> reflections;
  std::transform(
    inputs.begin(), inputs.end(), std::back_inserter(reflections), [&](auto const& view) {
      return reflect_input_column(base_column_size, view);
    });

  return reflections;
}

}  // namespace jit
}  // namespace cudf
