/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/io/types.hpp>
#include <cudf/utilities/export.hpp>

namespace CUDF_EXPORT cudf {
namespace test {

void expect_metadata_equal(cudf::io::table_input_metadata in_meta,
                           cudf::io::table_metadata out_meta);

/**
 * @brief Ensures that the metadata of two tables matches for the root columns as well as for all
 * descendents (recursively)
 */
void expect_metadata_equal(cudf::io::table_metadata lhs_meta, cudf::io::table_metadata rhs_meta);

}  // namespace test
}  // namespace CUDF_EXPORT cudf
