/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <nvbench/nvbench.cuh>

std::unique_ptr<cudf::table> create_lists_data(nvbench::state& state,
                                               cudf::size_type const num_columns = 1,
                                               cudf::size_type const min_val     = 0,
                                               cudf::size_type const max_val     = 5);

std::unique_ptr<cudf::table> create_structs_data(nvbench::state& state,
                                                 cudf::size_type const n_cols = 1);
