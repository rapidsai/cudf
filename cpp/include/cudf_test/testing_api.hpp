/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#pragma once

/**
 * @file testing_api.hpp
 * @brief exposes the testing APIs, GTEST can be disabled by defining CUDF_TEST_EXCLUDE_GTEST=1, and
 * then including a testing API that conforms to GTEST's API before including any cudf_test headers.
 * The testing API must define CUDF_TEST_TESTING_API_IMPL to signal cudf_test that it conforms to
 * the GTest API.
 *
 */

#if !(defined(CUDF_TEST_EXCLUDE_GTEST) && CUDF_TEST_EXCLUDE_GTEST)
#include <cudf_test/testing_api_gtest.hpp>
#endif

#if !defined(CUDF_TEST_TESTING_API_IMPL)
#error \
  "No CUDF Testing API implementation found, Include a testing API that conforms to the GoogleTest API before including libcudftestutil headers"
#endif
