/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

#include <cudf.h>
#include <gtest/gtest.h>

#include <ftw.h>

#include <rmm/rmm.h>

#define ASSERT_CUDA_SUCCEEDED(expr) ASSERT_EQ(cudaSuccess, expr)
#define EXPECT_CUDA_SUCCEEDED(expr) EXPECT_EQ(cudaSuccess, expr)

#define ASSERT_RMM_SUCCEEDED(expr)  ASSERT_EQ(RMM_SUCCESS, expr)
#define EXPECT_RMM_SUCCEEDED(expr)  EXPECT_EQ(RMM_SUCCESS, expr)

#define ASSERT_CUDF_SUCCEEDED(gdf_error_expression) \
do { \
    gdf_error _assert_cudf_success_eval_result;\
    ASSERT_NO_THROW(_assert_cudf_success_eval_result = gdf_error_expression); \
    const char* _assertion_failure_message = #gdf_error_expression; \
    ASSERT_EQ(_assert_cudf_success_eval_result, GDF_SUCCESS) << "Failing expression: " << _assertion_failure_message; \
} while (0)


// Base class fixture for GDF google tests that initializes / finalizes the
// RAPIDS memory manager
struct GdfTest : public ::testing::Test
{
    static void SetUpTestCase() {
        ASSERT_RMM_SUCCEEDED( rmmInitialize(nullptr) );
    }

    static void TearDownTestCase() {
        ASSERT_RMM_SUCCEEDED( rmmFinalize() );
    }
};

/**
* @brief Environment for google tests that creates/deletes temporary directory 
* for each test program and provides path of filenames
* 
* TempDirTestEnvironment* const temp_env = static_cast<TempDirTestEnvironment*>(
*   ::testing::AddGlobalTestEnvironment(new TempDirTestEnvironment));
*/
struct TempDirTestEnvironment : public ::testing::Environment
{
    std::string tmpdir;

    void SetUp() {
        char tmp_format[]="/tmp/gtest.XXXXXX";
        tmpdir = mkdtemp(tmp_format);
        tmpdir += "/";
    }

    void TearDown() {
        //TODO: should use std::filesystem instead, once C++17 support added
        nftw(tmpdir.c_str(), rm_files, 10, FTW_DEPTH|FTW_MOUNT|FTW_PHYS);
    }

    static int rm_files(const char *pathname, const struct stat *sbuf, int type, struct FTW *ftwb)
    {
        return remove(pathname);
    }

    /**
    * @brief get temporary path of filename for this test program
    *
    * @return temporary directory path
    */
    std::string get_temp_dir()
    {
        return tmpdir;
    }
};
