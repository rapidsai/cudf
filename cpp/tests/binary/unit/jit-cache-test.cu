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

namespace cudf {
namespace test {


struct JitCacheTest : public ::testing::Test
                    , public cudf::jit::cudfJitCache {
    JitCacheTest() {
    }

    virtual ~JitCacheTest() {
    }

    virtual void SetUp() {
        // Single value column
        auto column = cudf::test::column_wrapper<int>{{5,0}};
        auto expect = cudf::test::column_wrapper<int>{{125,0}};

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
            for (auto&& cache_file : cache_files_to_clear) {
                auto path = getFilePath(cache_file);
                std::remove(path.c_str());
            }
        #endif
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

    std::vector<std::string> cache_files_to_clear;
};

// Test the in memory caching ability
TEST_F(JitCacheTest, MemoryCacheTest) {

    { // Check the kernel caching
        // Cleanup
        cache_files_to_clear.push_back("MemoryCacheTestProg.my_kernel_3_int");
        purgeFileCache();

        // Single value column
        // TODO (dm): should be a scalar tho
        auto column = cudf::test::column_wrapper<int>{{5,0}};
        auto expect = cudf::test::column_wrapper<int>{{125,0}};

        // give a different program and check that still the old kernel is used
        auto program = getProgram("MemoryCacheTestProg", program2_source);
        auto kernel = getKernelInstantiation("my_kernel",
                                                    program,
                                                    {"3", "int"});

        (*std::get<1>(kernel)).configure_1d_max_occupancy()
                 .launch(column.get()->data);

        ASSERT_TRUE(expect == column) << "Expected col: " << expect.to_str()
                                      << "  Actual col: " << column.to_str();
    }

    { // Check program source caching
        // Cleanup
        cache_files_to_clear.push_back("MemoryCacheTestProg");
        cache_files_to_clear.push_back("MemoryCacheTestProg.my_kernel_4_int");
        purgeFileCache();

        // Single value column
        // TODO (dm): should be a scalar tho
        auto column = cudf::test::column_wrapper<int>{{5,0}};
        auto expect = cudf::test::column_wrapper<int>{{625,0}};

        // give a different program and check that still the old kernel is used
        auto program = getProgram("MemoryCacheTestProg", program2_source);
        auto kernel = getKernelInstantiation("my_kernel",
                                                    program,
                                                    {"4", "int"});

        (*std::get<1>(kernel)).configure_1d_max_occupancy()
                 .launch(column.get()->data);

        ASSERT_TRUE(expect == column) << "Expected col: " << expect.to_str()
                                      << "  Actual col: " << column.to_str();
    }
}

// Test the file caching ability
#if defined(JITIFY_USE_CACHE)
TEST_F(JitCacheTest, FileCacheTest) {

}
#endif

// Test the thread safety
// just create a bunch of std threads and make sure they all give correct result




} // namespace test
} // namespace gdf
