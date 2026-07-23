/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "core/exceptions.hpp"

#include <cudf/binaryop.h>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstdint>
#include <memory>

namespace {
std::unique_ptr<cudf::column>& get_column_owner(cudfColumn_t col)
{
  return *reinterpret_cast<std::unique_ptr<cudf::column>*>(col->addr);
}

cudf::data_type to_data_type(cudfDataType_t type)
{
  return cudf::data_type{static_cast<cudf::type_id>(type.id), type.scale};
}
}  // namespace

extern "C" cudfError_t cudfBinaryOpColumns(cudfColumn_t lhs,
                                           cudfColumn_t rhs,
                                           cudfBinaryOp_t op,
                                           cudfDataType_t output_type,
                                           cudaStream_t stream,
                                           cudfColumn_t* result)
{
  return cudf::c::translate_exceptions([=] {
    CUDF_EXPECTS(result != nullptr, "output cudfColumn_t pointer cannot be null");

    auto result_column = cudf::binary_operation(get_column_owner(lhs)->view(),
                                                get_column_owner(rhs)->view(),
                                                static_cast<cudf::binary_operator>(op),
                                                to_data_type(output_type),
                                                rmm::cuda_stream_view{stream});

    auto owner = new std::unique_ptr<cudf::column>(std::move(result_column));
    *result    = new cudfColumn{reinterpret_cast<uintptr_t>(owner)};
  });
}
