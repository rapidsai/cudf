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

#ifndef CUDF_TEST_FIXTURES_H
#define CUDF_TEST_FIXTURES_H

#include <cudf/cudf.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include <rmm/rmm.h>

#include <ftw.h>
#include "cudf_test_utils.cuh"

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
     * @brief Get directory path to use for temporary files
     *
     * @return std::string The temporary directory path
     */
    std::string get_temp_dir() { return tmpdir; }

    /**
     * @brief Get a temporary filepath to use for the specified filename
     *
     * @return std::string The temporary filepath
     */
    std::string get_temp_filepath(std::string filename)
    {
        return tmpdir + filename;
    }
};
#endif // CUDF_TEST_FIXTURES_H
