/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once
#include <cudf/column/column_view.hpp>

#include <jit/cache.hpp>

#include <optional>
#include <string>
namespace cudf::jit {
enum class row_execution { TRANSFORM, JOIN };
void validate_row_column(column_view column);
kernel link_row_kernel(std::span<uint8_t const> row,
                       row_execution execution,
                       std::string_view source                                      = {},
                       std::span<rtcx::memory_fragment const> dependencies          = {},
                       std::optional<std::span<std::string const>> required_symbols = std::nullopt);
kernel get_row_kernel(std::string const& source,
                      row_execution execution,
                      std::span<rtcx::memory_fragment const> dependencies          = {},
                      std::optional<std::span<std::string const>> required_symbols = std::nullopt);
}  // namespace cudf::jit
