/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

#include <cudf_test/cudf_gtest.hpp>

#include <cudf/utilities/default_stream.hpp>

#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
TEST(DefaultStreamTest, PtdsIsEnabled) { EXPECT_TRUE(cudf::is_ptds_enabled()); }
#else
TEST(DefaultStreamTest, PtdsIsNotEnabled) { EXPECT_FALSE(cudf::is_ptds_enabled()); }
#endif
