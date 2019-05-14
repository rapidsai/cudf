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


#include <tests/utilities/column_wrapper.cuh>
#include <binary/jit/core/cache.h>
#include <gtest/gtest.h>
#include <ftw.h>

namespace cudf {
namespace test {


struct JitCacheTest : public ::testing::Test
                    , public cudf::jit::cudfJitCache {
    JitCacheTest() {
    }

    virtual ~JitCacheTest() {
    }

    virtual void SetUp() {
        // Prime up the cache so that the in-memory and file cache is populated
        
        purgeFileCache();

        // Single value column
        auto column = cudf::test::column_wrapper<int>{{4,0}};
        auto expect = cudf::test::column_wrapper<int>{{64,0}};

        // make program
        auto program = getProgram("MemoryCacheTestProg", program_source);
        // make kernel
        auto kernel = getKernelInstantiation("my_kernel",
                                                    program,
                                                    {"3", "int"});
        (*std::get<1>(kernel)).configure_1d_max_occupancy()
                 .launch(column.get()->data);

        ASSERT_EQ(expect, column);
    }

    virtual void TearDown() {
        purgeFileCache();
    }

    void purgeFileCache() {
        #if defined(JITIFY_USE_CACHE)
            std::string cachedir = cudf::jit::getCacheDir();
            nftw(cachedir.c_str(), rm_files, 10, FTW_DEPTH|FTW_MOUNT|FTW_PHYS);
        #endif
    }

    // TODO (dm): remove if/when #1642 gets merged and use the following:
    // https://github.com/rapidsai/cudf/pull/1642/files#diff-aee4e59e3139423d13edaa200e8c82b3R64
    static int rm_files(const char *pathname, const struct stat *sbuf, int type, struct FTW *ftwb)
    {
        return remove(pathname);
    }

    const char* program_source =
        "my_program\n"
        "template<int N, typename T>\n"
        "__global__\n"
        "void my_kernel(T* data) {\n"
        "    T data0 = data[0];\n"
        "    for( int i=0; i<N-1; ++i ) {\n"
        "        data[0] *= data0;\n"
        "    }\n"
        "}\n";

    const char* program2_source =
        "my_program\n"
        "template<int N, typename T>\n"
        "__global__\n"
        "void my_kernel(T* data) {\n"
        "    T data0 = data[0];\n"
        "    for( int i=0; i<N-1; ++i ) {\n"
        "        data[0] += data0;\n"
        "    }\n"
        "}\n";

};

TEST_F(JitCacheTest, CacheExceptionTest) {
    EXPECT_NO_THROW(auto program = getProgram("MemoryCacheTestProg"));
    EXPECT_ANY_THROW(auto program1 = getProgram("MemoryCacheTestProg1"));
}

// Test the in memory caching ability
TEST_F(JitCacheTest, MemoryCacheKernelTest) {
    // Check the kernel caching

    // Single value column
    // TODO (dm): should be a scalar tho
    auto column = cudf::test::column_wrapper<int>{{5,0}};
    auto expect = cudf::test::column_wrapper<int>{{125,0}};

    // make new program and rename it to match old program
    auto program = getProgram("MemoryCacheTestProg1", program2_source);
    // TODO: when I convert this pair to a class, make an inherited test class that can edit names
    std::get<0>(program) = "MemoryCacheTestProg";

    // remove any file cache so below kernel should not be obtained from file
    purgeFileCache();

    // make kernel that if the cache tried to compile, will use a different
    // program than intended and give wrong result. 
    auto kernel = getKernelInstantiation("my_kernel",
                                         program,
                                         {"3", "int"});

    (*std::get<1>(kernel)).configure_1d_max_occupancy()
                .launch(column.get()->data);

    ASSERT_TRUE(expect == column) << "Expected col: " << expect.to_str()
                                  << "  Actual col: " << column.to_str();
}

TEST_F(JitCacheTest, MemoryCacheProgramTest) {
    // Check program source caching

    // Single value column
    // TODO (dm): should be a scalar tho
    auto column = cudf::test::column_wrapper<int>{{5,0}};
    auto expect = cudf::test::column_wrapper<int>{{625,0}};

    // remove any file cache so below program should not be obtained from file
    purgeFileCache();

    auto program = getProgram("MemoryCacheTestProg");
    // make kernel that HAS to be compiled
    auto kernel = getKernelInstantiation("my_kernel",
                                         program,
                                         {"4", "int"});

    (*std::get<1>(kernel)).configure_1d_max_occupancy()
                .launch(column.get()->data);

    ASSERT_TRUE(expect == column) << "Expected col: " << expect.to_str()
                                  << "  Actual col: " << column.to_str();
}

// Test the file caching ability
#if defined(JITIFY_USE_CACHE)
TEST_F(JitCacheTest, FileCacheProgramTest) {
    // Brand new cache object that has nothing in in-memory cache
    cudf::jit::cudfJitCache cache;

    // Single value column
    auto column = cudf::test::column_wrapper<int>{{5,0}};
    auto expect = cudf::test::column_wrapper<int>{{625,0}};

    // make program
    auto program = cache.getProgram("MemoryCacheTestProg");
    // make kernel that HAS to be compiled
    auto kernel = cache.getKernelInstantiation("my_kernel",
                                                program,
                                                {"4", "int"});
    (*std::get<1>(kernel)).configure_1d_max_occupancy()
                .launch(column.get()->data);

    ASSERT_EQ(expect, column) << "Expected col: " << expect.to_str()
                              << "  Actual col: " << column.to_str();
}

TEST_F(JitCacheTest, FileCacheKernelTest) {
    // Brand new cache object that has nothing in in-memory cache
    cudf::jit::cudfJitCache cache;

    // Single value column
    auto column = cudf::test::column_wrapper<int>{{5,0}};
    auto expect = cudf::test::column_wrapper<int>{{125,0}};

    // make program
    auto program = cache.getProgram("MemoryCacheTestProg1", program2_source);
    // make kernel that should NOT need to be compiled
    // TODO (dm): convert this pair to a class, so the name can be edited
    std::get<0>(program) = "MemoryCacheTestProg";
    auto kernel = cache.getKernelInstantiation("my_kernel",
                                                program,
                                                {"3", "int"});
    (*std::get<1>(kernel)).configure_1d_max_occupancy()
                .launch(column.get()->data);

    ASSERT_EQ(expect, column) << "Expected col: " << expect.to_str()
                              << "  Actual col: " << column.to_str();
}
#endif

// Test the thread safety
// just create a bunch of std threads and make sure they all give correct result




} // namespace test
} // namespace gdf
