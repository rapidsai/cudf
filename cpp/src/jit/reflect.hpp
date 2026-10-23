/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/transform.hpp>

#include <jit/output_columns.hpp>

#include <memory>
#include <span>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

namespace cudf::jit {

struct transform_reflection {
  std::string input_accessors;
  std::string output_accessors;
  std::vector<std::string> ptx_input_value_types;
  std::vector<std::string> ptx_output_value_types;

  bool operator==(transform_reflection const& other) const
  {
    return input_accessors == other.input_accessors && output_accessors == other.output_accessors &&
           ptx_input_value_types == other.ptx_input_value_types &&
           ptx_output_value_types == other.ptx_output_value_types;
  }
};

/** @brief Reflects input/output accessors and PTX value types. */
transform_reflection reflect(std::variant<udf_source_type, lto_binary_type> source_type,
                             std::span<transform_input_spec const> inputs,
                             std::span<transform_output_spec const> outputs,
                             std::span<int32_t const> input_table_sources  = {},
                             std::span<int32_t const> output_table_sources = {});

/** @brief Reflects accessors and PTX value types from column storage. */
transform_reflection reflect(std::variant<udf_source_type, lto_binary_type> source_type,
                             std::span<transform_input const> inputs,
                             std::span<output_column const> outputs,
                             std::span<int32_t const> input_table_sources  = {},
                             std::span<int32_t const> output_table_sources = {});

/** @brief Derives input type specifications from columns or scalars. */
std::vector<transform_input_spec> make_input_specs(std::span<transform_input const> inputs);

/** @brief Derives output type specifications from column storage. */
std::vector<transform_output_spec> make_output_specs(std::span<output_column const> outputs);

/** @brief Derives output type specifications from requested types and offsets. */
std::vector<transform_output_spec> make_output_specs(
  std::span<transform_output const> outputs,
  std::span<std::unique_ptr<column> const> string_offsets);

/** @brief Reflects the UDF function signature from type specifications. */
std::string reflect_udf_signature(bool is_null_aware,
                                  bool has_user_data,
                                  std::span<transform_input_spec const> inputs,
                                  std::span<transform_output_spec const> outputs,
                                  bool use_physical_types);

/** @brief Reflects the UDF function signature from column storage. */
std::string reflect_udf_signature(bool is_null_aware,
                                  bool has_user_data,
                                  std::span<transform_input const> inputs,
                                  std::span<output_column const> outputs,
                                  bool use_physical_types);

}  // namespace cudf::jit
