/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf_test/cudf_gtest.hpp>

#include <cudf/utilities/default_stream.hpp>

#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
TEST(DefaultStreamTest, PtdsIsEnabled) { EXPECT_TRUE(cudf::is_ptds_enabled()); }
#else
TEST(DefaultStreamTest, PtdsIsNotEnabled) { EXPECT_FALSE(cudf::is_ptds_enabled()); }
#endif
