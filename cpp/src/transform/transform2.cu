
/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/transform.hpp>

#include <span>
#include <variant>
#include <vector>

namespace cudf {

enum class [[nodiscard]] udf_source_type : uint8_t { LTOIR_BINARY = 0, PTX_BINARY = 1, CUDA = 2 };

struct [[nodiscard]] transform_operation_params {
  using input = std::variant<column_view, scalar, void*>;

  std::vector<input> inputs = {};

  std::vector<bool> include_nullness = {};

  std::vector<size_type> output_sizes = {};

  std::vector<data_type> output_types = {};

  std::vector<output_nullability> output_nullability = {};

  std::span<char const> udf = {};

  udf_source_type type = udf_source_type::LTOIR_BINARY;

  rmm::cuda_stream_view stream = cudf::get_default_stream();

  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref();
};

std::unique_ptr<table> transform_operation(transform_operation_params const& params)
{
  return nullptr;
}

}  // namespace cudf
