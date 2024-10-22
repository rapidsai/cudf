/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudf_test/testing_main.hpp>

// NOTE: this file exists to define the parquet test's `main()` function.
// `main()` is kept in its own compilation unit to keep the compilation time for
// PARQUET_TEST at a minimum.
//
// Do not add any test definitions to this file.

CUDF_TEST_PROGRAM_MAIN()
