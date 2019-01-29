/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

// unit_test.cpp : Defines the entry point for the console application.
//

#include "gdf_interface.h"

#ifndef GDF_ORC_NO_FILE_TEST

TEST(gdf_orc_read_others, zero_orc)
{
    orc_read_arg arg;
    gdf_error ret = load_orc(&arg, "examples/zero.orc");
    ASSERT_EQ(GDF_VALIDITY_UNSUPPORTED, ret);   // check the error code

    release_orc_read_arg(&arg);
}

TEST(gdf_orc_read_others, version1999)
{
    orc_read_arg arg;
    gdf_error ret = load_orc(&arg, "examples/version1999.orc");

    // return code is "Success" even though no data
    ASSERT_EQ(GDF_SUCCESS, ret);

    EXPECT_EQ(0, arg.num_cols_out);
    EXPECT_EQ(0, arg.num_rows_out);
    EXPECT_EQ(nullptr, arg.data);

    release_orc_read_arg(&arg);
}

TEST(gdf_orc_read_others, file_not_found)
{
    orc_read_arg arg;
    gdf_error ret = load_orc(&arg, "examples/file_not_found.orc");  // this file does not exist

    ASSERT_EQ(GDF_VALIDITY_UNSUPPORTED, ret);      // GDF_ORC_FILE_NOT_FOUND is converted into GDF_VALIDITY_UNSUPPORTED

    release_orc_read_arg(&arg);
}

// --------------------------------------------------------------------------
#ifdef DO_TEST_OTHERS

void UnsupportedCompressedFileLoad(const char* filename)
{
    orc_read_arg arg;
    gdf_error ret = load_orc(&arg, filename);

    // return code is "GDF_VALIDITY_UNSUPPORTED" if the compression mode is not supported.
    ASSERT_EQ(GDF_VALIDITY_UNSUPPORTED, ret);

    EXPECT_EQ(0, arg.num_cols_out);
    EXPECT_EQ(0, arg.num_rows_out);
    EXPECT_EQ(nullptr, arg.data);

    release_orc_read_arg(&arg);
}

TEST(gdf_orc_read_snappy, nulls_at_end_snappy)
{
    UnsupportedCompressedFileLoad("examples/nulls-at-end-snappy.orc");
}

TEST(gdf_orc_read_snappy, testSnappy)
{
    UnsupportedCompressedFileLoad("examples/TestOrcFile.testSnappy.orc");
}

TEST(gdf_orc_read_snappy, testWithoutIndex)
{
    UnsupportedCompressedFileLoad("examples/TestOrcFile.testWithoutIndex.orc");
}

TEST(gdf_orc_read_lz4, testLz4)
{
    UnsupportedCompressedFileLoad("examples/TestOrcFile.testLz4.orc");
}

TEST(gdf_orc_read_lzo, testLzo)
{
    UnsupportedCompressedFileLoad("examples/TestOrcFile.testLzo.orc");
}

#endif


#endif // #ifndef GDF_ORC_NO_FILE_TEST