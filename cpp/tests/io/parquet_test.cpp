/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/testing_main.hpp>

// NOTE: this file exists to define the parquet test's `main()` function.
// `main()` is kept in its own compilation unit to keep the compilation time for
// PARQUET_TEST at a minimum.
//
// Do not add any test definitions to this file.

CUDF_TEST_PROGRAM_MAIN()
